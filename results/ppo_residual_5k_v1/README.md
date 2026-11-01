# Residual PPO 5k run and EDA

This is the first full residual PPO measurement run around the frozen BC
policy. The requested 5,000 steps were rounded to the 64-step rollout boundary,
so Stable-Baselines3 executed 5,056 environment steps.

## Result

- completed training episodes: 80
- exploration-on training successes: 39/80 (48.75%)
- mean training episode reward: 10.68
- first 10 episode mean reward: 16.51
- last 10 episode mean reward: 14.58
- deterministic post-training evaluation: 10/10 successes
- mean deterministic evaluation reward: 19.48
- success threshold: 30 mm lift
- training wall time: 2,330.7 seconds (38.8 minutes)

## Post-training video

The final saved policy was evaluated again at 10 deterministic, seeded cube
positions while recording the Gazebo desktop. All 10 trials succeeded, with a
mean maximum lift of 35.0 mm. The full local recording is 366.75 seconds long;
the checked-in `gazebo_evaluation_2min.mp4` is its first 120.25 seconds.

The complete video and step log remain under
`evaluations/ppo_residual_eval_20261102_052903/`. The exact ten outcomes are in
`video_evaluation_summary.json`.

The reward curve did not improve monotonically. Its 10-episode mean fell into
the 4-8 range in the middle of training, then recovered to 14.58 at the end.
The last-10 mean remained below the first-10 mean, so this run is not evidence
that PPO improved the nominal BC policy.

The training action distribution explains much of the instability. Depending
on the joint, 35.5-47.8% of stochastic residual actions reached a configured
boundary. The learned Gaussian standard deviation remained about 0.986 at the
end. A follow-up experiment should reduce initial action variance and the
entropy coefficient, then compare BC and PPO at identical fixed cube positions.

The 10/10 deterministic result is encouraging, but it is not directly
comparable with the earlier 9/10 BC result because the sampled positions were
not paired and both samples are small.

## Files

- `reward_curve.png`: episode reward and 10-episode rolling mean
- `reward_components.png`: 200-step rolling mean of each reward term
- `residual_distribution.png`: per-joint residuals in radians
- `training_success_heatmap.png`: exploration-on training outcomes by position
- `evaluation_success_heatmap.png`: deterministic outcomes by position
- `gazebo_evaluation_2min.mp4`: two-minute representative post-training video
- `video_evaluation_summary.json`: outcomes from the full recorded evaluation
- `training_episode_summary.csv`: one row per training episode
- `evaluation_episode_summary.csv`: one row per deterministic evaluation
- `summary.json`: training configuration and evaluation outcomes
- `eda_summary.json`: aggregate EDA measurements

The final loadable model is
`models/ppo_residual_5k_v1/ppo_residual.zip`. The full 5,056-row step log and
seven intermediate checkpoints remain in the ignored local run directory
`rl_runs/ppo_residual_20261102_041127/`.
