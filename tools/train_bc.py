#!/usr/bin/env python3
"""Train an episode-split behavior-cloning policy for the CRANE+ grasp task."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class NormalizedPolicy(nn.Module):
    """MLP with train-set normalization embedded for portable inference."""

    def __init__(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.register_buffer("state_mean", state_mean.clone())
        self.register_buffer("state_std", state_std.clone())
        self.register_buffer("action_mean", action_mean.clone())
        self.register_buffer("action_std", action_std.clone())
        self.network = nn.Sequential(
            nn.Linear(state_mean.numel(), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_mean.numel()),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        normalized_state = (state - self.state_mean) / self.state_std
        normalized_action = self.network(normalized_state)
        return normalized_action * self.action_std + self.action_mean

    def forward_normalized(self, normalized_state: torch.Tensor) -> torch.Tensor:
        return self.network(normalized_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Merged transitions.jsonl")
    parser.add_argument("output", type=Path, help="Directory for model and reports")
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"),
                        default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_rows(path: Path) -> tuple[list[dict], str]:
    digest = hashlib.sha256()
    rows = []
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not all(key in row for key in ("source_episode", "state", "action")):
                raise ValueError(f"{path}:{line_number}: incomplete transition")
            rows.append(row)
    if not rows:
        raise ValueError(f"No transitions in {path}")
    return rows, digest.hexdigest()


def episode_split(episodes: Iterable[str], seed: int) -> dict[str, list[str]]:
    names = sorted(set(episodes))
    if len(names) < 5:
        raise ValueError("At least five episodes are required for train/val/test splits")
    random.Random(seed).shuffle(names)
    train_count = max(1, int(len(names) * 0.8))
    val_count = max(1, int(len(names) * 0.1))
    if train_count + val_count >= len(names):
        train_count = len(names) - 2
        val_count = 1
    return {
        "train": names[:train_count],
        "validation": names[train_count:train_count + val_count],
        "test": names[train_count + val_count:],
    }


def arrays_for(rows: list[dict], episodes: set[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["source_episode"] in episodes]
    states = np.asarray([row["state"] for row in selected], dtype=np.float32)
    actions = np.asarray([row["action"] for row in selected], dtype=np.float32)
    references = np.asarray(
        [row.get("current_reference", row["action"]) for row in selected],
        dtype=np.float32,
    )
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError("State and action records must be fixed-length vectors")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Dataset contains non-finite values")
    return states, actions, references


def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    action_order: list[str],
) -> dict:
    error = prediction - target
    mae = np.abs(error).mean(axis=0)
    rmse = np.sqrt(np.square(error).mean(axis=0))
    return {
        "samples": int(target.shape[0]),
        "mae_rad": float(np.abs(error).mean()),
        "rmse_rad": float(np.sqrt(np.square(error).mean())),
        "per_joint": {
            name: {"mae_rad": float(mae[index]), "rmse_rad": float(rmse[index])}
            for index, name in enumerate(action_order)
        },
    }


def predict(model: NormalizedPolicy, states: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for offset in range(0, len(states), 2048):
            batch = torch.from_numpy(states[offset:offset + 2048]).to(device)
            output.append(model(batch).cpu().numpy())
    return np.concatenate(output, axis=0)


def write_loss_svg(path: Path, history: list[dict]) -> None:
    width, height = 900, 420
    left, right, top, bottom = 70, 20, 30, 55
    chart_width = width - left - right
    chart_height = height - top - bottom
    values = [row[key] for row in history for key in ("train_loss", "validation_loss")]
    positive = [max(value, 1e-9) for value in values]
    lower = math.log10(min(positive))
    upper = math.log10(max(positive))
    if math.isclose(lower, upper):
        upper = lower + 1.0

    def point(index: int, value: float) -> tuple[float, float]:
        x = left + chart_width * index / max(1, len(history) - 1)
        fraction = (math.log10(max(value, 1e-9)) - lower) / (upper - lower)
        y = top + chart_height * (1.0 - fraction)
        return x, y

    train_points = " ".join(
        f"{x:.1f},{y:.1f}" for x, y in
        (point(index, row["train_loss"]) for index, row in enumerate(history))
    )
    val_points = " ".join(
        f"{x:.1f},{y:.1f}" for x, y in
        (point(index, row["validation_loss"]) for index, row in enumerate(history))
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="20" text-anchor="middle" font-family="sans-serif" font-size="16">BC training loss (log scale)</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#333"/>
<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#333"/>
<polyline points="{train_points}" fill="none" stroke="#1565c0" stroke-width="2"/>
<polyline points="{val_points}" fill="none" stroke="#ef6c00" stroke-width="2"/>
<text x="{left + chart_width / 2}" y="{height - 12}" text-anchor="middle" font-family="sans-serif" font-size="13">epoch</text>
<text x="16" y="{top + chart_height / 2}" transform="rotate(-90 16 {top + chart_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">Smooth L1 loss</text>
<line x1="{width - 205}" y1="42" x2="{width - 175}" y2="42" stroke="#1565c0" stroke-width="3"/><text x="{width - 168}" y="47" font-family="sans-serif" font-size="12">train</text>
<line x1="{width - 105}" y1="42" x2="{width - 75}" y2="42" stroke="#ef6c00" stroke-width="3"/><text x="{width - 68}" y="47" font-family="sans-serif" font-size="12">validation</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not dataset.is_file():
        raise SystemExit(f"Dataset not found: {dataset}")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output is not empty (use --overwrite): {output}")
    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    rows, dataset_sha256 = read_rows(dataset)
    split = episode_split((row["source_episode"] for row in rows), args.seed)
    arrays = {
        name: arrays_for(rows, set(episode_names))
        for name, episode_names in split.items()
    }
    input_dim = arrays["train"][0].shape[1]
    action_dim = arrays["train"][1].shape[1]
    if any(values[0].shape[1] != input_dim or values[1].shape[1] != action_dim
           for values in arrays.values()):
        raise ValueError("Inconsistent state or action dimensions")

    collection_summary = dataset.with_name("summary.json")
    if collection_summary.is_file():
        source_summary = json.loads(collection_summary.read_text(encoding="utf-8"))
        state_order = list(source_summary["state_order"])
        action_order = list(source_summary["action_order"])
    else:
        state_order = [f"state_{index}" for index in range(input_dim)]
        action_order = [f"action_{index}" for index in range(action_dim)]

    train_states, train_actions, _ = arrays["train"]
    state_mean_np = train_states.mean(axis=0)
    state_std_np = np.maximum(train_states.std(axis=0), 1e-6)
    action_mean_np = train_actions.mean(axis=0)
    action_std_np = np.maximum(train_actions.std(axis=0), 1e-6)
    state_mean = torch.from_numpy(state_mean_np)
    state_std = torch.from_numpy(state_std_np)
    action_mean = torch.from_numpy(action_mean_np)
    action_std = torch.from_numpy(action_std_np)

    model = NormalizedPolicy(
        state_mean, state_std, action_mean, action_std, args.hidden_size
    ).to(device)
    normalized_train_states = (train_states - state_mean_np) / state_std_np
    normalized_train_actions = (train_actions - action_mean_np) / action_std_np
    val_states, val_actions, _ = arrays["validation"]
    normalized_val_states = (val_states - state_mean_np) / state_std_np
    normalized_val_actions = (val_actions - action_mean_np) / action_std_np

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized_train_states),
            torch.from_numpy(normalized_train_actions),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_state_tensor = torch.from_numpy(normalized_val_states).to(device)
    val_action_tensor = torch.from_numpy(normalized_val_actions).to(device)
    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
    )

    history = []
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        train_samples = 0
        for state_batch, action_batch in loader:
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.forward_normalized(state_batch)
            loss = criterion(prediction, action_batch)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * len(state_batch)
            train_samples += len(state_batch)
        train_loss = train_total / train_samples
        model.eval()
        with torch.inference_mode():
            val_loss = float(criterion(
                model.forward_normalized(val_state_tensor), val_action_tensor
            ))
        scheduler.step(val_loss)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}",
                  flush=True)
        if stale_epochs >= args.patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)

    metrics = {}
    for name, (states, actions, references) in arrays.items():
        prediction = predict(model, states, device)
        metrics[name] = regression_metrics(prediction, actions, action_order)
        metrics[name]["persistence_baseline"] = regression_metrics(
            references, actions, action_order
        )

    model = model.to("cpu").eval()
    scripted = torch.jit.script(model)
    scripted.save(str(output / "policy.ts"))
    torch.save(model.state_dict(), output / "policy_state.pt")
    with (output / "loss_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    write_loss_svg(output / "loss_curve.svg", history)

    report = {
        "format": "crane_grasp_bc_policy_v1",
        "dataset": str(dataset),
        "dataset_sha256": dataset_sha256,
        "seed": args.seed,
        "device_used": str(device),
        "state_order": state_order,
        "action_order": action_order,
        "architecture": [input_dim, args.hidden_size, args.hidden_size, action_dim],
        "normalization_source": "training episodes only",
        "normalization": {
            "state_mean": state_mean_np.tolist(),
            "state_std": state_std_np.tolist(),
            "action_mean": action_mean_np.tolist(),
            "action_std": action_std_np.tolist(),
        },
        "split": split,
        "split_samples": {name: int(values[0].shape[0]) for name, values in arrays.items()},
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_loss": best_loss,
        "metrics": metrics,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    test = metrics["test"]
    baseline = test["persistence_baseline"]
    print(f"saved: {output}")
    print(f"test MAE={test['mae_rad']:.6f} rad "
          f"(persistence baseline={baseline['mae_rad']:.6f} rad)")


if __name__ == "__main__":
    main()
