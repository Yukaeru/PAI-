#!/usr/bin/env python3
"""Prepare and merge all successful capture episodes in one collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    args = parser.parse_args()
    collection = args.collection.expanduser().resolve()
    converter = Path(__file__).with_name("prepare_bc_dataset.py")
    successes = []
    failures = []
    for episode in sorted(collection.glob("episode_*")):
        metadata_path = episode / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("label") == "success":
            subprocess.run([sys.executable, str(converter), str(episode)], check=True)
            successes.append(episode)
        else:
            failures.append(episode)
    if not successes:
        raise SystemExit("No successful episodes found")

    output = collection / "bc_collection"
    output.mkdir(exist_ok=True)
    merged_path = output / "transitions.jsonl"
    total = 0
    state_order = None
    episode_summaries = []
    with merged_path.open("w", encoding="utf-8") as merged:
        for collection_episode_id, episode in enumerate(successes):
            summary = json.loads((episode / "bc_dataset" / "summary.json").read_text(
                encoding="utf-8"))
            if state_order is None:
                state_order = summary["state_order"]
            elif summary["state_order"] != state_order:
                raise RuntimeError("Episodes have incompatible observation layouts")
            episode_summaries.append({
                "episode": episode.name,
                "transitions": summary["output_samples"],
                "segments": len(summary["segments"]),
            })
            for line in (episode / "bc_dataset" / "transitions.jsonl").read_text(
                    encoding="utf-8").splitlines():
                if not line:
                    continue
                row = json.loads(line)
                row["collection_episode_id"] = collection_episode_id
                merged.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1
    summary = {
        "format": "crane_grasp_bc_collection_v1",
        "source_collection": str(collection),
        "success_episodes": len(successes),
        "failed_episodes": len(failures),
        "transitions": total,
        "state_order": state_order,
        "action_order": [f"crane_plus_joint{i}" for i in (1, 2, 3, 4)]
                        + ["crane_plus_joint_hand"],
        "episodes": episode_summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Merged {total} transitions from {len(successes)} successful episodes")
    print(output)


if __name__ == "__main__":
    main()
