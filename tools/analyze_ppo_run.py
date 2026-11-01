#!/usr/bin/env python3
"""Create EDA tables and plots from a residual PPO training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


JOINT_LABELS = ["joint 1", "joint 2", "joint 3", "joint 4", "gripper"]
RESIDUAL_LIMITS_RAD = [0.06, 0.06, 0.06, 0.06, 0.10]
COMPONENTS = [
    "reward_distance_progress",
    "reward_lift_progress",
    "reward_action_penalty",
    "reward_time_penalty",
    "reward_candidate_bonus",
    "reward_success_bonus",
    "reward_timeout_penalty",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Directory containing step CSV files")
    parser.add_argument("--rolling-episodes", type=int, default=10)
    parser.add_argument("--rolling-steps", type=int, default=200)
    parser.add_argument("--heatmap-bins", type=int, default=5)
    return parser.parse_args()


def episode_table(steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for episode, frame in steps.groupby("episode", sort=True):
        last = frame.iloc[-1]
        row = {
            "episode": int(episode),
            "target_x": float(last.target_x),
            "target_y": float(last.target_y),
            "steps": int(last.step),
            "episode_reward": float(last.episode_return),
            "max_lift_m": float(frame.lift_m.max()),
            "min_distance_m": float(frame.distance_m.min()),
            "success": int(frame.success.max()),
            "terminated": int(last.terminated),
            "truncated": int(last.truncated),
            "complete": int(bool(last.terminated) or bool(last.truncated)),
        }
        for index in range(1, 6):
            row[f"mean_abs_residual_rad_joint_{index}"] = float(
                frame[f"residual_rad_joint_{index}"].abs().mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def reward_curve(episodes: pd.DataFrame, output: Path, window: int) -> None:
    fig, axis = plt.subplots(figsize=(10, 5.5))
    x = episodes.episode.to_numpy()
    rewards = episodes.episode_reward
    axis.plot(x, rewards.to_numpy(), color="#9bbbd4", linewidth=1.0, alpha=0.8,
              label="episode reward")
    rolling = rewards.rolling(window, min_periods=1).mean()
    axis.plot(x, rolling.to_numpy(), color="#005a9c", linewidth=2.3,
              label=f"rolling mean ({window} episodes)")
    success_reward = episodes.loc[episodes.success == 1]
    failure_reward = episodes.loc[episodes.success == 0]
    axis.scatter(success_reward.episode.to_numpy(),
                 success_reward.episode_reward.to_numpy(),
                 s=18, color="#238636", label="success", zorder=3)
    axis.scatter(failure_reward.episode.to_numpy(),
                 failure_reward.episode_reward.to_numpy(),
                 s=18, color="#d1242f", label="failure", zorder=3)
    axis.set(xlabel="Training episode", ylabel="Episode reward")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4)
    fig.suptitle("Residual PPO reward curve", y=0.99)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    fig.savefig(output, dpi=160)
    plt.close(fig)


def reward_components(steps: pd.DataFrame, output: Path, window: int) -> None:
    fig, axis = plt.subplots(figsize=(11, 6))
    x = np.arange(1, len(steps) + 1)
    for component in COMPONENTS:
        smoothed = steps[component].rolling(window, min_periods=1).mean()
        axis.plot(x, smoothed.to_numpy(), linewidth=1.5,
                  label=component.removeprefix("reward_").replace("_", " "))
    axis.set(title=f"Reward components ({window}-step rolling mean)",
             xlabel="Environment step", ylabel="Reward contribution")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def residual_distribution(steps: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5), sharey=True)
    for index, (axis, label) in enumerate(zip(axes, JOINT_LABELS), start=1):
        values = steps[f"residual_rad_joint_{index}"]
        axis.hist(values.to_numpy(), bins=30, color="#6f42c1", alpha=0.82)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_title(label)
        axis.set_xlabel("residual (rad)")
    axes[0].set_ylabel("steps")
    fig.suptitle("PPO residual action distribution")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def success_heatmap(episodes: pd.DataFrame, output: Path, bins: int,
                    title_prefix: str) -> None:
    x_edges = np.linspace(-0.010, 0.010, bins + 1)
    y_edges = np.linspace(-0.110, -0.090, bins + 1)
    total, _, _ = np.histogram2d(
        episodes.target_y, episodes.target_x, bins=[y_edges, x_edges]
    )
    successful, _, _ = np.histogram2d(
        episodes.loc[episodes.success == 1, "target_y"],
        episodes.loc[episodes.success == 1, "target_x"],
        bins=[y_edges, x_edges],
    )
    rate = np.divide(successful, total, out=np.full_like(total, np.nan), where=total > 0)
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(rate, origin="lower", vmin=0.0, vmax=1.0,
                        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                        aspect="equal", cmap="RdYlGn")
    for y_index in range(bins):
        for x_index in range(bins):
            if total[y_index, x_index] > 0:
                x = (x_edges[x_index] + x_edges[x_index + 1]) / 2
                y = (y_edges[y_index] + y_edges[y_index + 1]) / 2
                axis.text(x, y, f"{rate[y_index, x_index]:.0%}\nn={int(total[y_index, x_index])}",
                          ha="center", va="center", fontsize=8)
    axis.set(title=f"{title_prefix} success rate by cube position",
             xlabel="Cube x (m)", ylabel="Cube y (m)")
    fig.colorbar(image, ax=axis, label="Success rate")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run = args.run.expanduser().resolve()
    training_path = run / "training_steps.csv"
    if not training_path.is_file():
        raise SystemExit(f"Missing step log: {training_path}")
    training = pd.read_csv(training_path)
    if training.empty:
        raise SystemExit("Training step log is empty")
    all_episodes = episode_table(training)
    all_episodes.to_csv(run / "training_episode_summary.csv", index=False)
    episodes = all_episodes.loc[all_episodes.complete == 1].copy()
    if episodes.empty:
        raise SystemExit("No completed training episodes in step log")
    reward_curve(episodes, run / "reward_curve.png", args.rolling_episodes)
    reward_components(
        training, run / "reward_components.png", args.rolling_steps
    )
    residual_distribution(training, run / "residual_distribution.png")
    success_heatmap(
        episodes, run / "training_success_heatmap.png", args.heatmap_bins, "Training"
    )

    evaluation_path = run / "evaluation_steps.csv"
    evaluation_summary = None
    if evaluation_path.is_file():
        evaluation = pd.read_csv(evaluation_path)
        if not evaluation.empty:
            evaluation_episodes = episode_table(evaluation)
            evaluation_episodes.to_csv(
                run / "evaluation_episode_summary.csv", index=False
            )
            success_heatmap(
                evaluation_episodes,
                run / "evaluation_success_heatmap.png",
                args.heatmap_bins,
                "Evaluation",
            )
            evaluation_summary = {
                "episodes": int(len(evaluation_episodes)),
                "successes": int(evaluation_episodes.success.sum()),
                "success_rate": float(evaluation_episodes.success.mean()),
                "mean_episode_reward": float(evaluation_episodes.episode_reward.mean()),
            }

    summary = {
        "format": "crane_residual_ppo_eda_v1",
        "training_steps": int(len(training)),
        "training_episodes": int(len(episodes)),
        "training_successes": int(episodes.success.sum()),
        "training_success_rate": float(episodes.success.mean()),
        "mean_episode_reward": float(episodes.episode_reward.mean()),
        "first_10_episode_mean_reward": float(
            episodes.episode_reward.head(10).mean()
        ),
        "last_10_episode_mean_reward": float(
            episodes.episode_reward.tail(10).mean()
        ),
        "first_10_episode_success_rate": float(episodes.success.head(10).mean()),
        "last_10_episode_success_rate": float(episodes.success.tail(10).mean()),
        "final_rolling_episode_reward": float(
            episodes.episode_reward.rolling(
                args.rolling_episodes, min_periods=1
            ).mean().iloc[-1]
        ),
        "joint_limit_clip_rate": float(
            (training.joint_limit_clip_count > 0).mean()
        ),
        "rate_limit_clip_rate": float(
            (training.rate_limit_clip_count > 0).mean()
        ),
        "mean_abs_residual_rad": {
            f"joint_{index}": float(training[f"residual_rad_joint_{index}"].abs().mean())
            for index in range(1, 6)
        },
        "residual_boundary_rate": {
            f"joint_{index}": float((
                training[f"residual_rad_joint_{index}"].abs()
                >= 0.99 * RESIDUAL_LIMITS_RAD[index - 1]
            ).mean())
            for index in range(1, 6)
        },
        "evaluation": evaluation_summary,
    }
    (run / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
