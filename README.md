# CRANE+ grasp capture helpers

## Reset the robot and cube

Stop any active recording in the **CRANE+ Grasp** panel, then run on the Mac:

```bash
./tools/reset_grasp_scene.sh
```

The command opens the gripper, returns all four arm joints to zero, and puts
`wood_cube_5cm` back at the initial world position. It deliberately refuses to
run while `ros2 bag record` is active, because resetting during a capture would
make that episode invalid.

After it finishes, select `approach` in the panel and press **START NEW
EPISODE**.

## Build compact behavior-cloning transitions

```bash
python3 tools/prepare_bc_dataset.py grasp_episode_20261101_161013_4446cd
```

Outputs are written under the episode's `bc_dataset/` directory:

- `transitions.jsonl`: measured joint state and a controller-reference action
  approximately 0.2 seconds in the future
- `summary.json`: joint order, normalization values, segment boundaries, and
  conversion settings

Long stationary waits are omitted. Disconnected motion windows are separate
segments so a target never crosses a removed time gap.

## Collect varied successful demonstrations automatically

Stop any active recording in the GUI panel, then run:

```bash
./tools/run_auto_collection.sh 30
```

Each episode varies the cube position within a small reachable area, records
the cube pose with the five joint values, and labels success only when Gazebo
reports a lift of at least 3 cm. Raw rosbag files are not created for automated
episodes; the lightweight JSONL data is copied to `datasets/` and converted
into one merged `bc_collection/transitions.jsonl` file.

Position variation is useful only because the automatic observations include
`object_x`, `object_y`, and `object_z`. A joint-only policy cannot infer where a
moved cube is located.

The current primary collection is
`datasets/grasp_auto_20261102_012854`: 30/30 successful grasps and 5,036 merged
transitions. See [`datasets/README.md`](datasets/README.md) for measured ranges.

## GitHub

This workspace uses the `main` branch and the following remote:

```text
origin  https://github.com/Yukaeru/PAI-.git
```

Raw rosbag directories are ignored because they grow quickly. Lightweight
JSONL samples, metadata, conversion tools, and merged BC transitions are kept
in Git.
