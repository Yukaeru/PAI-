#!/usr/bin/env python3
"""Run a TorchScript BC grasp policy in Gazebo and measure lift success."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
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

# Limits from crane_plus.urdf.xacro, in radians.
LOWER_LIMITS = np.asarray([-2.61799, -1.82451, -2.55656, -1.84039, -0.70895],
                          dtype=np.float32)
UPPER_LIMITS = np.asarray([2.61799, 1.78380, 2.45357, 1.76843, 0.66706],
                          dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--x-range", type=float, default=0.010)
    parser.add_argument("--y-center", type=float, default=-0.100)
    parser.add_argument("--y-range", type=float, default=0.010)
    parser.add_argument("--control-period", type=float, default=0.20)
    parser.add_argument("--rollout-seconds", type=float, default=15.0)
    parser.add_argument("--max-velocity", type=float, default=1.5,
                        help="Per-joint safety rate limit in rad/s")
    parser.add_argument("--success-height", type=float, default=0.030)
    parser.add_argument("--disable-gripper-latch", action="store_true",
                        help="Allow the policy to reopen after it starts closing")
    return parser.parse_args()


def set_cube_pose(x: float, y: float) -> None:
    request = (f'name: "wood_cube_5cm", position: {{x: {x}, y: {y}, z: 1.05}}, '
               'orientation: {w: 1.0}')
    result = subprocess.run([
        "ign", "service", "-s", "/world/default/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean", "--timeout", "5000",
        "--req", request,
    ], check=True, capture_output=True, text=True)
    if "data: true" not in result.stdout:
        raise RuntimeError("Gazebo rejected cube reset: " + result.stdout.strip())


def finite_joint_map(names, positions) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in zip(names, positions)
        if math.isfinite(float(value))
    }


class PolicyRunner(Node):
    def __init__(self, model_path: Path, control_period: float, max_velocity: float,
                 gripper_latch: bool):
        super().__init__("bc_grasp_policy_evaluator")
        torch.set_num_threads(1)
        self.model = torch.jit.load(str(model_path), map_location="cpu").eval()
        self.control_period = control_period
        self.max_step = max_velocity * control_period
        self.use_gripper_latch = gripper_latch
        self.gripper_latched = False
        self.joint: dict[str, float] = {}
        self.object_position: list[float] | None = None
        self.sim_time: float | None = None
        self.arm_client = ActionClient(self, FollowJointTrajectory, ARM_ACTION)
        self.hand_client = ActionClient(self, FollowJointTrajectory, HAND_ACTION)
        self.arm_publisher = self.create_publisher(JointTrajectory, ARM_TOPIC, 10)
        self.hand_publisher = self.create_publisher(JointTrajectory, HAND_TOPIC, 10)
        self.create_subscription(
            JointState, "/joint_states", self.on_joint, qos_profile_sensor_data
        )
        self.create_subscription(Clock, "/clock", self.on_clock,
                                 qos_profile_sensor_data)
        self.create_subscription(
            TFMessage, POSE_TOPIC, self.on_pose, qos_profile_sensor_data
        )

    def on_joint(self, message: JointState) -> None:
        self.joint = finite_joint_map(message.name, message.position)

    def on_clock(self, message: Clock) -> None:
        self.sim_time = message.clock.sec + message.clock.nanosec * 1e-9

    def on_pose(self, message: TFMessage) -> None:
        # Dynamic poses are ordered with the cube first in this world. The
        # geometric filter excludes the robot base and links near the origin.
        for transform in message.transforms:
            position = transform.transform.translation
            if position.z > 0.9 and math.hypot(position.x, position.y) > 0.04:
                self.object_position = [float(position.x), float(position.y),
                                        float(position.z)]
                return

    def spin_step(self, timeout: float = 0.02) -> None:
        rclpy.spin_once(self, timeout_sec=timeout)

    def wait(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.spin_step(min(0.02, max(0.0, deadline - time.monotonic())))

    def wait_until_ready(self, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.spin_step()
            if (self.sim_time is not None and self.object_position is not None
                    and all(name in self.joint for name in JOINTS)):
                if self.arm_publisher.get_subscription_count() > 0 \
                        and self.hand_publisher.get_subscription_count() > 0:
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

    def send_reset_goal(self, client, joints: list[str], positions: list[float],
                        seconds: float) -> None:
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
        goal_future = client.send_goal_async(goal)
        deadline = time.monotonic() + seconds + 10.0
        while not goal_future.done() and time.monotonic() < deadline:
            self.spin_step()
        handle = goal_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("Reset trajectory was rejected")
        result_future = handle.get_result_async()
        while not result_future.done() and time.monotonic() < deadline:
            self.spin_step()
        if not result_future.done() or result_future.result().result.error_code != 0:
            raise RuntimeError("Reset trajectory failed")

    def reset_robot(self) -> None:
        self.gripper_latched = False
        self.send_reset_goal(self.hand_client, [HAND], [-0.5236], 1.2)
        self.send_reset_goal(self.arm_client, ARM, [0.0, 0.0, 0.0, 0.0], 2.5)

    @staticmethod
    def trajectory(joints: list[str], positions: np.ndarray,
                   seconds: float) -> JointTrajectory:
        message = JointTrajectory()
        message.joint_names = joints
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        whole = int(seconds)
        point.time_from_start.sec = whole
        point.time_from_start.nanosec = int((seconds - whole) * 1e9)
        message.points = [point]
        return message

    def policy_action(self) -> tuple[np.ndarray, np.ndarray]:
        measured = np.asarray([self.joint[name] for name in JOINTS], dtype=np.float32)
        if self.object_position is None:
            raise RuntimeError("Cube pose disappeared")
        state = np.concatenate((measured, np.asarray(self.object_position, dtype=np.float32)))
        with torch.inference_mode():
            raw = self.model(torch.from_numpy(state).unsqueeze(0)).squeeze(0).numpy()
        if not np.isfinite(raw).all():
            raise RuntimeError("Policy returned a non-finite action")
        limited = np.clip(raw, LOWER_LIMITS, UPPER_LIMITS)
        limited = np.clip(limited, measured - self.max_step, measured + self.max_step)
        if self.use_gripper_latch and raw[4] >= 0.0:
            self.gripper_latched = True
        if self.gripper_latched:
            # Every demonstration keeps the close reference through the lift.
            # Hysteresis prevents a slipped cube from making the stateless MLP
            # interpret the state as a pre-grasp state and reopen the fingers.
            limited[4] = min(0.1745, measured[4] + self.max_step)
        return state, limited

    def publish_action(self, action: np.ndarray) -> None:
        self.arm_publisher.publish(self.trajectory(ARM, action[:4], self.control_period))
        self.hand_publisher.publish(self.trajectory([HAND], action[4:], self.control_period))

    def rollout(self, seconds: float, trace_path: Path,
                initial_z: float) -> tuple[float, float, int]:
        if self.sim_time is None:
            raise RuntimeError("Simulation clock is unavailable")
        start_sim = self.sim_time
        deadline_sim = start_sim + seconds
        next_control_sim = start_sim
        wall_deadline = time.monotonic() + seconds * 5.0 + 30.0
        max_z = initial_z
        steps = 0
        with trace_path.open("w", encoding="utf-8") as trace:
            while self.sim_time is not None and self.sim_time < deadline_sim:
                if time.monotonic() >= wall_deadline:
                    raise RuntimeError("Gazebo simulation clock stalled during rollout")
                if self.sim_time < next_control_sim:
                    self.spin_step(0.02)
                    if self.object_position is not None:
                        max_z = max(max_z, self.object_position[2])
                    continue
                state, action = self.policy_action()
                self.publish_action(action)
                record = {
                    "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                    "sim_time": self.sim_time,
                    "elapsed_sim_seconds": self.sim_time - start_sim,
                    "state": state.tolist(),
                    "action": action.tolist(),
                    "gripper_latched": self.gripper_latched,
                }
                trace.write(json.dumps(record, ensure_ascii=False) + "\n")
                trace.flush()
                steps += 1
                next_control_sim += self.control_period
                if next_control_sim < self.sim_time:
                    next_control_sim = self.sim_time + self.control_period
        self.wait(0.5)
        final_z = self.object_position[2] if self.object_position is not None else float("nan")
        max_z = max(max_z, final_z)
        return max_z, final_z, steps


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be positive")
    model_path = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")
    output.mkdir(parents=True, exist_ok=False)
    trace_dir = output / "traces"
    trace_dir.mkdir()
    rng = random.Random(args.seed)

    rclpy.init()
    runner = PolicyRunner(
        model_path, args.control_period, args.max_velocity,
        gripper_latch=not args.disable_gripper_latch,
    )
    results = []
    try:
        runner.wait_until_ready()
        for index in range(args.episodes):
            target_x = rng.uniform(-args.x_range, args.x_range)
            target_y = rng.uniform(args.y_center - args.y_range,
                                   args.y_center + args.y_range)
            result = {
                "episode": index,
                "cube_target_xy": [target_x, target_y],
                "label": "failure",
                "error": None,
            }
            try:
                runner.reset_robot()
                set_cube_pose(target_x, target_y)
                runner.wait_for_cube(target_x, target_y)
                runner.wait(1.0)
                if runner.object_position is None:
                    raise RuntimeError("Cube pose unavailable after reset")
                initial = list(runner.object_position)
                trace_path = trace_dir / f"episode_{index:03d}.jsonl"
                max_z, final_z, steps = runner.rollout(
                    args.rollout_seconds, trace_path, initial[2]
                )
                lift = max_z - initial[2]
                result.update({
                    "cube_pose_initial": initial,
                    "max_cube_z": max_z,
                    "final_cube_z": final_z,
                    "max_lift_height_m": lift,
                    "control_steps": steps,
                    "label": "success" if lift >= args.success_height else "failure",
                })
            except Exception as exception:  # Keep the rest of the evaluation usable.
                result["error"] = str(exception)
            results.append(result)
            lift = result.get("max_lift_height_m")
            lift_text = "n/a" if lift is None else f"{lift:.4f}m"
            print(f"[{index + 1}/{args.episodes}] {result['label']} "
                  f"xy=({target_x:+.4f},{target_y:+.4f}) lift={lift_text}",
                  flush=True)
    finally:
        runner.destroy_node()
        rclpy.shutdown()

    successes = sum(result["label"] == "success" for result in results)
    summary = {
        "format": "crane_grasp_bc_gazebo_evaluation_v1",
        "model": str(model_path),
        "seed": args.seed,
        "episodes": args.episodes,
        "successes": successes,
        "failures": args.episodes - successes,
        "success_rate": successes / args.episodes,
        "success_height_m": args.success_height,
        "control_period_seconds": args.control_period,
        "rollout_seconds": args.rollout_seconds,
        "max_velocity_rad_s": args.max_velocity,
        "gripper_latch": not args.disable_gripper_latch,
        "position_distribution": {
            "x_center": 0.0,
            "x_range": args.x_range,
            "y_center": args.y_center,
            "y_range": args.y_range,
        },
        "results": results,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "successes": successes,
        "episodes": args.episodes,
        "success_rate": successes / args.episodes,
    }), flush=True)


if __name__ == "__main__":
    main()
