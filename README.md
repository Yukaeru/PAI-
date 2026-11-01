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

## Fine-tune with residual PPO

PPO is used here as a small correction policy, not as a replacement trained
from random motion. The frozen BC model proposes the nominal five joint
targets; PPO observes joints, velocities, cube-to-end-effector displacement,
lift height, and elapsed time, then adds at most 0.06 rad to each arm target
and 0.10 rad to the gripper target. Physical joint and velocity limits remain
active.

The container environment can be recreated with:

```bash
./tools/setup_ppo_env.sh
```

With Gazebo and the `pai_ros2` container running, first check the environment:

```bash
PPO_CHECK_ONLY=1 ./tools/run_ppo_training.sh 8
```

Then run a short smoke training or a longer experiment:

```bash
./tools/run_ppo_training.sh 128
./tools/run_ppo_training.sh 10000
```

Set `PPO_RECORD_VIDEO=1` to record the visible Gazebo/VNC desktop during a
run. This is best used for a short run because a full training video can become
large:

```bash
PPO_RECORD_VIDEO=1 ./tools/run_ppo_training.sh 128
```

Outputs are written under ignored `rl_runs/` directories. Each completed run
contains `ppo_residual.zip`, `summary.json`, `training.monitor.csv`,
`training_steps.csv`, and `evaluation_steps.csv`. It also generates a reward
curve, reward-component chart, residual-action distribution, position success
heatmaps, and per-episode CSV tables. Video runs contain
`gazebo_training.mp4`. The initial 128-step smoke test is a pipeline check, not
enough evidence to claim that PPO improves over BC. Compare many fixed-seed
evaluation episodes before selecting a new production policy.
The checked-in measured smoke result is documented in
[`results/ppo_residual_smoke_v1/`](results/ppo_residual_smoke_v1/README.md).

The measured 5,056-step run, plots, and limitations are documented in
[`results/ppo_residual_5k_v1/`](results/ppo_residual_5k_v1/README.md). Its final
model is `models/ppo_residual_5k_v1/ppo_residual.zip`.

## GitHub

This workspace uses the `main` branch and the following remote:

```text
origin  https://github.com/Yukaeru/PAI-.git
```

Raw rosbag directories are ignored because they grow quickly. Lightweight
JSONL samples, metadata, conversion tools, and merged BC transitions are kept
in Git.
