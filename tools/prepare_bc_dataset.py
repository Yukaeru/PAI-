#!/usr/bin/env python3
"""Convert one CRANE+ capture episode into compact BC transitions.

The converter keeps short context windows around commanded reference motion
and phase changes. Long idle gaps become separate segments, so a future target
is never selected across an artificial time jump. Measured contact jitter is
not treated as a new command.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="Directory containing samples.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--motion-threshold", type=float, default=1e-4,
                        help="Minimum per-sample joint/reference change in radians")
    parser.add_argument("--context-seconds", type=float, default=0.5,
                        help="Time retained before and after motion/phase boundaries")
    parser.add_argument("--horizon-seconds", type=float, default=0.2,
                        help="Future controller-reference horizon used as the BC action")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def vector(row: dict, field: str, joint_order: list[str]) -> list[float]:
    values = row.get(field, {})
    result = []
    for joint in joint_order:
        value = float(values[joint])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {field}.{joint} at sim_time={row.get('sim_time')}")
        result.append(value)
    return result


def state_vector(row: dict, joint_order: list[str], include_object: bool) -> list[float]:
    result = vector(row, "joint_positions", joint_order)
    if include_object:
        position = row["object_pose"]["position"]
        values = [float(position[index]) for index in range(3)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite object pose at sim_time={row.get('sim_time')}")
        result.extend(values)
    return result


def max_delta(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def mean_std(vectors: list[list[float]]) -> tuple[list[float], list[float]]:
    columns = list(zip(*vectors))
    means = [statistics.fmean(column) for column in columns]
    stds = [statistics.pstdev(column) or 1.0 for column in columns]
    return means, stds


def build_keep_mask(rows: list[dict], references: list[list[float]], threshold: float,
                    context_seconds: float) -> list[bool]:
    anchor_times = []
    for index in range(1, len(rows)):
        moved = max_delta(references[index], references[index - 1]) > threshold
        phase_changed = rows[index].get("phase") != rows[index - 1].get("phase")
        if moved or phase_changed:
            anchor_times.append(float(rows[index]["sim_time"]))
    if not anchor_times:
        return [False] * len(rows)
    anchor_times.sort()

    keep = []
    for row in rows:
        stamp = float(row["sim_time"])
        position = bisect.bisect_left(anchor_times, stamp)
        distances = []
        if position < len(anchor_times):
            distances.append(abs(anchor_times[position] - stamp))
        if position:
            distances.append(abs(anchor_times[position - 1] - stamp))
        keep.append(min(distances) <= context_seconds)
    return keep


def contiguous_segments(keep: list[bool]) -> list[list[int]]:
    segments: list[list[int]] = []
    current: list[int] = []
    for index, selected in enumerate(keep):
        if selected:
            current.append(index)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return [segment for segment in segments if len(segment) >= 2]


def main() -> None:
    args = parse_args()
    episode = args.episode.expanduser().resolve()
    metadata_path = episode / "metadata.json"
    samples_path = episode / "samples.jsonl"
    if not metadata_path.is_file() or not samples_path.is_file():
        raise SystemExit(f"Not a capture episode: {episode}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("label") != "success":
        raise SystemExit("BC conversion requires an episode labelled success")
    joint_order = list(metadata["joint_order"])
    rows = read_jsonl(samples_path)
    if len(rows) < 2:
        raise SystemExit("Need at least two samples")

    include_object = all(
        row.get("object_pose") and len(row["object_pose"].get("position", [])) == 3
        for row in rows
    )
    state_order = joint_order + (["object_x", "object_y", "object_z"] if include_object else [])
    states = [state_vector(row, joint_order, include_object) for row in rows]
    references = [vector(row, "controller_reference", joint_order) for row in rows]
    keep = build_keep_mask(rows, references, args.motion_threshold,
                           args.context_seconds)
    segments = contiguous_segments(keep)
    if not segments:
        raise SystemExit("No motion segments found; lower --motion-threshold")

    output_dir = (args.output_dir or (episode / "bc_dataset")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    transitions_path = output_dir / "transitions.jsonl"
    transition_states: list[list[float]] = []
    transition_actions: list[list[float]] = []
    segment_summaries = []

    with transitions_path.open("w", encoding="utf-8") as stream:
        for segment_id, indices in enumerate(segments):
            segment_times = [float(rows[index]["sim_time"]) for index in indices]
            for segment_step, index in enumerate(indices):
                target_stamp = float(rows[index]["sim_time"]) + args.horizon_seconds
                target_offset = bisect.bisect_left(segment_times, target_stamp)
                target_offset = min(target_offset, len(indices) - 1)
                target_index = indices[target_offset]
                state = states[index]
                action = references[target_index]
                record = {
                    "source_episode": episode.name,
                    "segment_id": segment_id,
                    "segment_step": segment_step,
                    "sim_time": float(rows[index]["sim_time"]),
                    "phase": rows[index].get("phase"),
                    "state": state,
                    "current_reference": references[index],
                    "action": action,
                    "action_sim_time": float(rows[target_index]["sim_time"]),
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                transition_states.append(state)
                transition_actions.append(action)
            segment_summaries.append({
                "segment_id": segment_id,
                "samples": len(indices),
                "start_sim_time": segment_times[0],
                "end_sim_time": segment_times[-1],
                "start_phase": rows[indices[0]].get("phase"),
                "end_phase": rows[indices[-1]].get("phase"),
            })

    state_mean, state_std = mean_std(transition_states)
    action_mean, action_std = mean_std(transition_actions)
    summary = {
        "format": "crane_grasp_bc_v1",
        "source_episode": str(episode),
        "source_label": metadata.get("label"),
        "joint_order": joint_order,
        "state_order": state_order,
        "state": ("measured joint positions and Gazebo object position"
                  if include_object else "measured joint positions at the current sample"),
        "action": f"controller reference approximately {args.horizon_seconds}s in the future",
        "motion_threshold_rad": args.motion_threshold,
        "context_seconds": args.context_seconds,
        "original_samples": len(rows),
        "output_samples": len(transition_states),
        "removed_samples": len(rows) - len(transition_states),
        "segments": segment_summaries,
        "normalization": {
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
        },
        "warning": "One success episode is enough to validate the pipeline, not to train a robust policy.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(transition_states)} transitions in {len(segments)} segments")
    print(f"Removed {len(rows) - len(transition_states)} idle samples")
    print(output_dir)


if __name__ == "__main__":
    main()
