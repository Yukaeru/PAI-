# Residual PPO smoke test

This result verifies the complete Gazebo training path: reset, observation,
bounded residual action, PPO update, model save/load, deterministic evaluation,
and optional MP4 capture.

- requested and actual timesteps: 128
- PPO rollout size: 64 steps
- frozen nominal policy: `models/bc_grasp_v1/policy.ts`
- deterministic evaluation: 1/1 success
- maximum measured lift: 33.8 mm
- success threshold: 30 mm
- training wall time: 58.9 seconds

This is deliberately labeled a smoke test. One evaluation episode and 128
training steps do not establish that PPO improves the 9/10 BC result. A useful
comparison needs fixed held-out cube positions, several random seeds, and many
more evaluation episodes.

The full local run, including the loadable PPO model and Monitor CSV, is under
the ignored directory `rl_runs/ppo_residual_20261102_034045/`. A separate
video-enabled check produced a valid 38.05-second H.264 recording at 1512x762
and 20 fps under `rl_runs/ppo_residual_20261102_034346/`.
