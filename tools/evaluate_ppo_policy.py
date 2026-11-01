#!/usr/bin/env python3
"""Evaluate a saved residual PPO policy in Gazebo and save step metrics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from train_ppo import ResidualGraspEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-model", type=Path, required=True)
    parser.add_argument("--ppo-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--control-period", type=float, default=0.20)
    parser.add_argument("--max-velocity", type=float, default=1.5)
    parser.add_argument("--max-episode-steps", type=int, default=75)
    parser.add_argument("--residual-arm-scale", type=float, default=0.06)
    parser.add_argument("--residual-hand-scale", type=float, default=0.10)
    parser.add_argument("--success-height", type=float, default=0.030)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be positive")
    bc_model = args.bc_model.expanduser().resolve()
    ppo_model = args.ppo_model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for model in (bc_model, ppo_model):
        if not model.is_file():
            raise SystemExit(f"Model not found: {model}")
    output.mkdir(parents=True, exist_ok=False)

    env_args = SimpleNamespace(
        control_period=args.control_period,
        max_velocity=args.max_velocity,
        max_episode_steps=args.max_episode_steps,
        success_height=args.success_height,
        residual_arm_scale=args.residual_arm_scale,
        residual_hand_scale=args.residual_hand_scale,
    )
    rclpy.init()
    raw_env = ResidualGraspEnv(bc_model, env_args)
    raw_env.enable_step_logging(output / "evaluation_steps.csv", "evaluation")
    env = Monitor(raw_env, filename=str(output / "evaluation"))
    results = []
    try:
        model = PPO.load(ppo_model, env=env, device="cpu")
        for episode in range(args.episodes):
            observation, reset_info = env.reset(seed=args.seed + episode)
            terminated = truncated = False
            reward_total = 0.0
            max_lift = 0.0
            last_info = reset_info
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, last_info = env.step(action)
                reward_total += float(reward)
                max_lift = max(max_lift, float(last_info["lift_m"]))
            result = {
                "episode": episode,
                "seed": args.seed + episode,
                "target_xy": [reset_info["target_x"], reset_info["target_y"]],
                "steps": raw_env.episode_step,
                "reward": reward_total,
                "max_lift_height_m": max_lift,
                "success": bool(last_info.get("is_success", False)),
            }
            results.append(result)
            print(
                f"[{episode + 1}/{args.episodes}] success={result['success']} "
                f"lift={max_lift:.4f}m reward={reward_total:.3f}",
                flush=True,
            )
    finally:
        raw_env.close()
        rclpy.shutdown()

    successes = sum(item["success"] for item in results)
    summary = {
        "format": "crane_residual_ppo_gazebo_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bc_model": str(bc_model),
        "ppo_model": str(ppo_model),
        "seed": args.seed,
        "episodes": args.episodes,
        "successes": successes,
        "failures": args.episodes - successes,
        "success_rate": successes / args.episodes,
        "mean_reward": float(np.mean([item["reward"] for item in results])),
        "mean_max_lift_height_m": float(np.mean([
            item["max_lift_height_m"] for item in results
        ])),
        "success_height_m": args.success_height,
        "results": results,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "successes": successes,
        "episodes": args.episodes,
        "success_rate": successes / args.episodes,
    }), flush=True)


if __name__ == "__main__":
    main()
