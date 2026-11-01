# CRANE+ grasp behavior cloning

This repository contains the successful Gazebo demonstrations, a PyTorch
behavior-cloning policy, closed-loop ROS 2 evaluation tools, and MP4 recording
for the CRANE+ cube-grasp task.

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

## Train the behavior-cloning policy

The split is made by episode, not by individual transition. With the current
30 demonstrations this gives 24 training, 3 validation, and 3 test episodes.
Normalization statistics are computed from the training episodes only.

```bash
python3 tools/train_bc.py \
  datasets/grasp_auto_20261102_012854/bc_collection/transitions.jsonl \
  models/bc_grasp_retrained \
  --device cpu
```

The command writes a portable TorchScript policy, normalization and split
metadata, per-joint metrics, loss history, and a loss chart. The checked-in
model is `models/bc_grasp_v1/policy.ts`.

## Evaluate in Gazebo and record MP4

Keep the `pai_ros2` container and CRANE+ Gazebo simulation running, then run:

```bash
./tools/run_bc_evaluation.sh 10
```

Every episode resets the robot and cube, chooses a new cube position, runs the
learned policy at 0.2-second simulation-time intervals, and labels success only
after a measured lift of at least 3 cm. Joint limits and a 1.5 rad/s rate limit
are enforced. Once the learned action starts closing the gripper, it remains
closed for the rest of that rollout, matching all training demonstrations and
preventing a slipped cube from causing the stateless policy to reopen.

The output path is printed when the run finishes. It contains:

- `run/summary.json`: positions, lift heights, errors, and success rate
- `run/traces/`: every observation and applied action
- `gazebo.mp4`: H.264 recording of the Gazebo desktop

Generated evaluation directories are ignored by Git. A compact representative
video and the validated 10-episode summaries are kept under
`results/bc_grasp_v1/`.

## Watch Gazebo live

Open the following URL while the container is running:

<http://localhost:6080/vnc.html?autoconnect=1&resize=scale>

Select the **Gazebo** window in the bottom task bar before starting an
evaluation. The MP4 recorder captures exactly this VNC desktop, so camera zoom
or viewpoint changes made there also appear in the video.

## Current measured result

- offline held-out test MAE: 0.00630 rad
- persistence baseline MAE: 0.03212 rad
- Gazebo evaluation: 9/10 successful grasps at unseen sampled positions
- success threshold: 30 mm; mean maximum lift over all 10 trials: 47.6 mm

See [`results/bc_grasp_v1/README.md`](results/bc_grasp_v1/README.md) for the
exact files and limitations. The observation uses the cube XYZ position
reported directly by Gazebo; this is not yet a camera-image policy.

## GitHub

This workspace uses the `main` branch and the following remote:

```text
origin  https://github.com/Yukaeru/PAI-.git
```

Raw rosbag directories are ignored because they grow quickly. Lightweight
JSONL samples, metadata, conversion tools, and merged BC transitions are kept
in Git.
