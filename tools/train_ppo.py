#!/usr/bin/env python3
"""Train a small residual PPO policy around the frozen BC grasp policy."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from crane_plus_commander.kinematics import forward_kinematics
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import torch
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM = [f"crane_plus_joint{i}" for i in (1, 2, 3, 4)]
HAND = "crane_plus_joint_hand"
JOINTS = ARM + [HAND]
ARM_ACTION = "/crane_plus_arm_controller/follow_joint_trajectory"
HAND_ACTION = "/crane_plus_gripper_controller/follow_joint_trajectory"
ARM_TOPIC = "/crane_plus_arm_controller/joint_trajectory"
HAND_TOPIC = "/crane_plus_gripper_controller/joint_trajectory"
POSE_TOPIC = "/world/default/dynamic_pose/info"

LOWER_LIMITS = np.asarray(
    [-2.61799, -1.82451, -2.55656, -1.84039, -0.70895], dtype=np.float32
)
UPPER_LIMITS = np.asarray(
    [2.61799, 1.78380, 2.45357, 1.76843, 0.66706], dtype=np.float32
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=75)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--checkpoint-freq", type=int, default=640)
    parser.add_argument("--control-period", type=float, default=0.20)
    parser.add_argument("--max-velocity", type=float, default=1.5)
    parser.add_argument("--residual-arm-scale", type=float, default=0.06)
    parser.add_argument("--residual-hand-scale", type=float, default=0.10)
    parser.add_argument("--success-height", type=float, default=0.030)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def set_cube_pose(x: float, y: float) -> None:
    request = (
        f'name: "wood_cube_5cm", position: {{x: {x}, y: {y}, z: 1.05}}, '
        "orientation: {w: 1.0}"
    )
    errors = []
    for attempt in range(1, 6):
        result = subprocess.run(
            [
                "ign", "service", "-s", "/world/default/set_pose",
                "--reqtype", "ignition.msgs.Pose",
                "--reptype", "ignition.msgs.Boolean", "--timeout", "5000",
                "--req", request,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and "data: true" in result.stdout:
            return
        errors.append(
            f"attempt {attempt}: returncode={result.returncode}, "
            f"stdout={result.stdout.strip()!r}, stderr={result.stderr.strip()!r}"
        )
        time.sleep(0.5 * attempt)
    raise RuntimeError(
        "Gazebo rejected cube reset after 5 attempts: " + "; ".join(errors)
    )


def finite_map(names, values) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in zip(names, values)
        if math.isfinite(float(value))
    }


class RosCraneInterface(Node):
    """Small synchronous facade over the ROS topics/actions used by Gym."""

    def __init__(self, control_period: float):
        super().__init__("residual_ppo_grasp_training")
        self.control_period = control_period
        self.joint: dict[str, float] = {}
        self.velocity: dict[str, float] = {}
        self.object_position: np.ndarray | None = None
        self.sim_time: float | None = None
        self.arm_client = ActionClient(self, FollowJointTrajectory, ARM_ACTION)
        self.hand_client = ActionClient(self, FollowJointTrajectory, HAND_ACTION)
        self.arm_publisher = self.create_publisher(JointTrajectory, ARM_TOPIC, 10)
        self.hand_publisher = self.create_publisher(JointTrajectory, HAND_TOPIC, 10)
        self.create_subscription(
            JointState, "/joint_states", self.on_joint, qos_profile_sensor_data
        )
        self.create_subscription(
            Clock, "/clock", self.on_clock, qos_profile_sensor_data
        )
        self.create_subscription(
            TFMessage, POSE_TOPIC, self.on_pose, qos_profile_sensor_data
        )

    def on_joint(self, message: JointState) -> None:
        self.joint = finite_map(message.name, message.position)
        self.velocity = finite_map(message.name, message.velocity)

    def on_clock(self, message: Clock) -> None:
        self.sim_time = message.clock.sec + message.clock.nanosec * 1e-9

    def on_pose(self, message: TFMessage) -> None:
        for transform in message.transforms:
            position = transform.transform.translation
            if position.z > 0.9 and math.hypot(position.x, position.y) > 0.04:
                self.object_position = np.asarray(
                    [position.x, position.y, position.z], dtype=np.float32
                )
                return

    def spin_step(self, timeout: float = 0.02) -> None:
        rclpy.spin_once(self, timeout_sec=timeout)

    def wait_wall(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.spin_step(min(0.02, max(0.0, deadline - time.monotonic())))

    def wait_sim(self, seconds: float) -> None:
        if self.sim_time is None:
            raise RuntimeError("Simulation clock is unavailable")
        target = self.sim_time + seconds
        deadline = time.monotonic() + seconds * 8.0 + 10.0
        while self.sim_time is None or self.sim_time < target:
            if time.monotonic() >= deadline:
                raise RuntimeError("Gazebo simulation clock stalled")
            self.spin_step()

    def wait_until_ready(self, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.spin_step()
            if (
                self.sim_time is not None
                and self.object_position is not None
                and all(name in self.joint for name in JOINTS)
                and self.arm_publisher.get_subscription_count() > 0
                and self.hand_publisher.get_subscription_count() > 0
            ):
                return
        raise RuntimeError("ROS joint state, cube pose, or controller is unavailable")

    def wait_for_cube(self, target_x: float, target_y: float, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.spin_step()
            if self.object_position is None:
                continue
            x, y, z = self.object_position
            if abs(x - target_x) < 0.015 and abs(y - target_y) < 0.015 and z > 0.99:
                return
        raise RuntimeError(f"Cube did not reach requested pose ({target_x}, {target_y})")

    def send_reset_goal(
        self, client, joints: list[str], positions: list[float], seconds: float
    ) -> None:
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Trajectory action server is unavailable")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joints
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        whole = int(seconds)
        point.time_from_start.sec = whole
        point.time_from_start.nanosec = int((seconds - whole) * 1e9)
        goal.trajectory.points = [point]
        future = client.send_goal_async(goal)
        deadline = time.monotonic() + seconds + 10.0
        while not future.done() and time.monotonic() < deadline:
            self.spin_step()
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("Reset trajectory was rejected")
        result = handle.get_result_async()
        while not result.done() and time.monotonic() < deadline:
            self.spin_step()
        if not result.done() or result.result().result.error_code != 0:
            raise RuntimeError("Reset trajectory failed")

    def reset_robot(self) -> None:
        self.send_reset_goal(self.hand_client, [HAND], [-0.5236], 1.2)
        self.send_reset_goal(self.arm_client, ARM, [0.0, 0.0, 0.0, 0.0], 2.5)

    @staticmethod
    def trajectory(
        joints: list[str], positions: np.ndarray, seconds: float
    ) -> JointTrajectory:
        message = JointTrajectory()
        message.joint_names = joints
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        whole = int(seconds)
        point.time_from_start.sec = whole
        point.time_from_start.nanosec = int((seconds - whole) * 1e9)
        message.points = [point]
        return message

    def publish_action(self, action: np.ndarray) -> None:
        self.arm_publisher.publish(
            self.trajectory(ARM, action[:4], self.control_period)
        )
        self.hand_publisher.publish(
            self.trajectory([HAND], action[4:], self.control_period)
        )


class ResidualGraspEnv(gym.Env):
    """PPO learns bounded corrections while BC supplies the nominal action."""

    metadata = {"render_modes": []}

    def __init__(self, bc_model: Path, args: argparse.Namespace):
        super().__init__()
        torch.set_num_threads(1)
        self.bc_model = torch.jit.load(str(bc_model), map_location="cpu").eval()
        self.interface = RosCraneInterface(args.control_period)
        self.interface.wait_until_ready()
        self.control_period = args.control_period
        self.max_step = args.max_velocity * args.control_period
        self.max_episode_steps = args.max_episode_steps
        self.success_height = args.success_height
        self.residual_scale = np.asarray(
            [args.residual_arm_scale] * 4 + [args.residual_hand_scale],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.asarray(
                [*LOWER_LIMITS, *([-5.0] * 5), *([-1.0] * 3), -0.2, 0.0],
                dtype=np.float32,
            ),
            high=np.asarray(
                [*UPPER_LIMITS, *([5.0] * 5), *([1.0] * 3), 0.3, 1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.episode_step = 0
        self.initial_cube_z = 1.05
        self.previous_distance = 0.0
        self.previous_lift = 0.0
        self.success_hold = 0
        self.gripper_latched = False
        self.target_x = 0.0
        self.target_y = -0.1
        self.episode_index = -1
        self.episode_return = 0.0
        self.step_log_file = None
        self.step_log_writer = None
        self.log_phase = "disabled"

    def enable_step_logging(self, path: Path, phase: str) -> None:
        """Start a fresh step-level CSV for training or evaluation."""
        self.close_step_log()
        self.step_log_file = path.open("w", encoding="utf-8", newline="")
        fields = [
            "phase", "episode", "step", "sim_time", "target_x", "target_y",
            "reward", "episode_return", "reward_distance_progress",
            "reward_lift_progress", "reward_action_penalty", "reward_time_penalty",
            "reward_candidate_bonus", "reward_success_bonus", "reward_timeout_penalty",
            "distance_m", "lift_m", "cube_z", "hand_joint", "candidate",
            "success", "terminated", "truncated", "gripper_latched",
            "joint_limit_clip_count", "rate_limit_clip_count",
        ]
        for prefix in ("normalized_residual", "residual_rad", "nominal", "applied"):
            fields.extend(f"{prefix}_joint_{index}" for index in range(1, 6))
        self.step_log_writer = csv.DictWriter(self.step_log_file, fieldnames=fields)
        self.step_log_writer.writeheader()
        self.step_log_file.flush()
        self.log_phase = phase
        self.episode_index = -1

    def close_step_log(self) -> None:
        if self.step_log_file is not None:
            self.step_log_file.flush()
            self.step_log_file.close()
        self.step_log_file = None
        self.step_log_writer = None

    def measured_joints(self) -> np.ndarray:
        return np.asarray(
            [self.interface.joint[name] for name in JOINTS], dtype=np.float32
        )

    def observation(self) -> tuple[np.ndarray, dict[str, float]]:
        joints = self.measured_joints()
        velocities = np.asarray(
            [self.interface.velocity.get(name, 0.0) for name in JOINTS],
            dtype=np.float32,
        )
        velocities = np.clip(velocities, -5.0, 5.0)
        if self.interface.object_position is None:
            raise RuntimeError("Cube pose disappeared")
        cube_world = self.interface.object_position.copy()
        cube_robot = cube_world.copy()
        cube_robot[2] -= 1.0
        ee = np.asarray(forward_kinematics(joints[:4])[:3], dtype=np.float32)
        relative = np.clip(cube_robot - ee, -1.0, 1.0)
        lift = float(cube_world[2] - self.initial_cube_z)
        elapsed = min(1.0, self.episode_step / self.max_episode_steps)
        observation = np.concatenate(
            (joints, velocities, relative, [lift, elapsed])
        ).astype(np.float32)
        metrics = {
            "distance_m": float(np.linalg.norm(relative)),
            "lift_m": lift,
            "cube_z": float(cube_world[2]),
            "hand_joint": float(joints[4]),
        }
        return observation, metrics

    def nominal_action(self, measured: np.ndarray) -> np.ndarray:
        if self.interface.object_position is None:
            raise RuntimeError("Cube pose disappeared")
        bc_state = np.concatenate((measured, self.interface.object_position))
        with torch.inference_mode():
            action = (
                self.bc_model(torch.from_numpy(bc_state).unsqueeze(0))
                .squeeze(0)
                .numpy()
            )
        if not np.isfinite(action).all():
            raise RuntimeError("BC policy returned a non-finite action")
        return action.astype(np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.interface.reset_robot()
        self.target_x = float(self.np_random.uniform(-0.010, 0.010))
        self.target_y = float(self.np_random.uniform(-0.110, -0.090))
        set_cube_pose(self.target_x, self.target_y)
        self.interface.wait_for_cube(self.target_x, self.target_y)
        self.interface.wait_wall(0.4)
        if self.interface.object_position is None:
            raise RuntimeError("Cube pose unavailable after reset")
        self.episode_step = 0
        self.initial_cube_z = float(self.interface.object_position[2])
        self.previous_lift = 0.0
        self.success_hold = 0
        self.gripper_latched = False
        self.episode_index += 1
        self.episode_return = 0.0
        observation, metrics = self.observation()
        self.previous_distance = metrics["distance_m"]
        info = {
            **metrics,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "is_success": False,
        }
        return observation, info

    def step(self, residual: np.ndarray):
        residual = np.clip(np.asarray(residual, dtype=np.float32), -1.0, 1.0)
        measured = self.measured_joints()
        nominal = self.nominal_action(measured)
        desired_unclipped = nominal + residual * self.residual_scale
        desired = np.clip(desired_unclipped, LOWER_LIMITS, UPPER_LIMITS)
        joint_limit_clip_count = int(np.count_nonzero(
            ~np.isclose(desired, desired_unclipped, atol=1e-7)
        ))
        desired_before_rate_limit = desired.copy()
        desired = np.clip(desired, measured - self.max_step, measured + self.max_step)
        rate_limit_clip_count = int(np.count_nonzero(
            ~np.isclose(desired, desired_before_rate_limit, atol=1e-7)
        ))

        if nominal[4] >= 0.0 or desired[4] >= 0.0:
            self.gripper_latched = True
        if self.gripper_latched:
            desired[4] = min(0.1745, measured[4] + self.max_step)

        self.interface.publish_action(desired)
        self.interface.wait_sim(self.control_period)
        self.episode_step += 1
        observation, metrics = self.observation()

        distance_progress = 10.0 * (self.previous_distance - metrics["distance_m"])
        lift_progress = 150.0 * (metrics["lift_m"] - self.previous_lift)
        action_penalty = -0.002 * float(np.mean(np.square(residual)))
        time_penalty = -0.01
        reward = distance_progress + lift_progress + action_penalty + time_penalty

        candidate = (
            metrics["lift_m"] >= self.success_height
            and metrics["hand_joint"] > 0.05
            and metrics["distance_m"] < 0.12
        )
        self.success_hold = self.success_hold + 1 if candidate else 0
        terminated = self.success_hold >= 2
        truncated = self.episode_step >= self.max_episode_steps
        candidate_bonus = 1.0 if candidate else 0.0
        success_bonus = 10.0 if terminated else 0.0
        timeout_penalty = -0.5 if truncated and not terminated else 0.0
        reward += candidate_bonus
        if terminated:
            reward += success_bonus
        elif truncated:
            reward += timeout_penalty

        self.episode_return += float(reward)

        self.previous_distance = metrics["distance_m"]
        self.previous_lift = metrics["lift_m"]
        info = {
            **metrics,
            "is_success": terminated,
            "nominal_action": nominal.tolist(),
            "applied_action": desired.tolist(),
            "residual_action": residual.tolist(),
            "reward_distance_progress": distance_progress,
            "reward_lift_progress": lift_progress,
            "reward_action_penalty": action_penalty,
        }
        if self.step_log_writer is not None:
            row = {
                "phase": self.log_phase,
                "episode": self.episode_index,
                "step": self.episode_step,
                "sim_time": self.interface.sim_time,
                "target_x": self.target_x,
                "target_y": self.target_y,
                "reward": reward,
                "episode_return": self.episode_return,
                "reward_distance_progress": distance_progress,
                "reward_lift_progress": lift_progress,
                "reward_action_penalty": action_penalty,
                "reward_time_penalty": time_penalty,
                "reward_candidate_bonus": candidate_bonus,
                "reward_success_bonus": success_bonus,
                "reward_timeout_penalty": timeout_penalty,
                "distance_m": metrics["distance_m"],
                "lift_m": metrics["lift_m"],
                "cube_z": metrics["cube_z"],
                "hand_joint": metrics["hand_joint"],
                "candidate": int(candidate),
                "success": int(terminated),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "gripper_latched": int(self.gripper_latched),
                "joint_limit_clip_count": joint_limit_clip_count,
                "rate_limit_clip_count": rate_limit_clip_count,
            }
            for prefix, values in (
                ("normalized_residual", residual),
                ("residual_rad", residual * self.residual_scale),
                ("nominal", nominal),
                ("applied", desired),
            ):
                row.update({
                    f"{prefix}_joint_{index}": float(value)
                    for index, value in enumerate(values, start=1)
                })
            self.step_log_writer.writerow(row)
            self.step_log_file.flush()
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        self.close_step_log()
        self.interface.destroy_node()


def evaluate(model: PPO, env: Monitor, episodes: int) -> list[dict]:
    results = []
    for episode in range(episodes):
        observation, reset_info = env.reset()
        terminated = truncated = False
        episode_reward = 0.0
        max_lift = 0.0
        last_info = reset_info
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, last_info = env.step(action)
            episode_reward += float(reward)
            max_lift = max(max_lift, float(last_info["lift_m"]))
        result = {
            "episode": episode,
            "reward": episode_reward,
            "steps": int(last_info.get("episode", {}).get("l", env.env.episode_step)),
            "max_lift_height_m": max_lift,
            "success": bool(last_info.get("is_success", False)),
            "target_xy": [reset_info["target_x"], reset_info["target_y"]],
        }
        results.append(result)
        print(
            f"[eval {episode + 1}/{episodes}] success={result['success']} "
            f"lift={max_lift:.4f}m reward={episode_reward:.3f}",
            flush=True,
        )
    return results


def main() -> None:
    args = parse_args()
    if args.timesteps < 1 or args.eval_episodes < 1:
        raise SystemExit("--timesteps and --eval-episodes must be positive")
    if args.n_steps < 2 or args.batch_size < 2:
        raise SystemExit("--n-steps and --batch-size must be at least 2")
    if args.checkpoint_freq < 1:
        raise SystemExit("--checkpoint-freq must be positive")
    bc_model = args.bc_model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not bc_model.is_file():
        raise SystemExit(f"BC model not found: {bc_model}")
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rclpy.init()
    raw_env = ResidualGraspEnv(bc_model, args)
    try:
        print("Checking Gymnasium environment...", flush=True)
        check_env(raw_env, warn=True, skip_render_check=True)
        if args.check_only:
            result = {
                "format": "crane_residual_ppo_environment_check_v1",
                "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                "result": "passed",
                "observation_shape": list(raw_env.observation_space.shape),
                "action_shape": list(raw_env.action_space.shape),
            }
            (output / "environment_check.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(result), flush=True)
            return

        raw_env.enable_step_logging(output / "training_steps.csv", "training")
        env = Monitor(raw_env, filename=str(output / "training"))
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=5,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.001,
            policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
            seed=args.seed,
            device="cpu",
            verbose=1,
        )
        started = time.monotonic()
        checkpoint_dir = output / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_residual",
        )
        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=checkpoint,
                progress_bar=False,
            )
        except BaseException as exception:
            model.save(output / "ppo_residual_interrupted")
            interruption = {
                "format": "crane_residual_ppo_interruption_v1",
                "interrupted_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_timesteps": model.num_timesteps,
                "error_type": type(exception).__name__,
                "error": str(exception),
            }
            (output / "interruption.json").write_text(
                json.dumps(interruption, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            raise
        training_seconds = time.monotonic() - started
        model.save(output / "ppo_residual")
        raw_env.enable_step_logging(output / "evaluation_steps.csv", "evaluation")
        evaluation = evaluate(model, env, args.eval_episodes)
        successes = sum(item["success"] for item in evaluation)
        summary = {
            "format": "crane_residual_ppo_training_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "bc_model": str(bc_model),
            "requested_timesteps": args.timesteps,
            "actual_timesteps": model.num_timesteps,
            "seed": args.seed,
            "training_wall_seconds": training_seconds,
            "observation_size": 15,
            "action_size": 5,
            "residual_scale_rad": args.residual_arm_scale,
            "residual_hand_scale_rad": args.residual_hand_scale,
            "control_period_seconds": args.control_period,
            "success_height_m": args.success_height,
            "evaluation_episodes": args.eval_episodes,
            "evaluation_successes": successes,
            "evaluation_success_rate": successes / args.eval_episodes,
            "evaluation": evaluation,
            "versions": {
                "gymnasium": gym.__version__,
                "stable_baselines3": __import__("stable_baselines3").__version__,
                "torch": torch.__version__,
            },
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "timesteps": model.num_timesteps,
            "eval_successes": successes,
            "eval_episodes": args.eval_episodes,
        }), flush=True)
    finally:
        raw_env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
