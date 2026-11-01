#!/usr/bin/env python3
"""Automatically collect varied CRANE+ grasp demonstrations in Gazebo.

Run this script inside the ROS 2 container after sourcing the Humble and
workspace setup files. It records lightweight JSONL episodes (no rosbag),
including the cube pose in every observation, and labels success from measured
Gazebo lift height.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time
from datetime import datetime, timezone

import rclpy
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectoryPoint

from crane_plus_commander.kinematics import inverse_kinematics, joint_in_range


ARM = [f"crane_plus_joint{i}" for i in (1, 2, 3, 4)]
HAND = "crane_plus_joint_hand"
JOINTS = ARM + [HAND]
ARM_ACTION = "/crane_plus_arm_controller/follow_joint_trajectory"
HAND_ACTION = "/crane_plus_gripper_controller/follow_joint_trajectory"
POSE_TOPIC = "/world/default/dynamic_pose/info"
PHASES = ("approach", "descend", "close", "lift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--x-range", type=float, default=0.010,
                        help="Uniform cube x variation around zero, in metres")
    parser.add_argument("--y-center", type=float, default=-0.100)
    parser.add_argument("--y-range", type=float, default=0.010,
                        help="Uniform cube y variation around --y-center, in metres")
    parser.add_argument("--success-height", type=float, default=0.030)
    return parser.parse_args()


def joint_map(names, positions) -> dict[str, float]:
    return {str(name): float(value) for name, value in zip(names, positions)
            if math.isfinite(float(value))}


def controller_reference(message) -> dict[str, float]:
    for field in ("reference", "desired"):
        point = getattr(message, field, None)
        values = list(getattr(point, "positions", []))
        if values:
            return joint_map(message.joint_names, values)
    return {}


class Collector(Node):
    def __init__(self, root: Path):
        super().__init__("automatic_grasp_collector")
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        self.joint: dict[str, float] = {}
        self.reference: dict[str, float] = {}
        self.sim_time: float | None = None
        self.object_pose: dict | None = None
        self.phase = "approach"
        self.sample_stream = None
        self.sample_count = 0
        self.last_sample_wall = 0.0
        self.last_sample_sim = None
        self.arm_client = ActionClient(self, FollowJointTrajectory, ARM_ACTION)
        self.hand_client = ActionClient(self, FollowJointTrajectory, HAND_ACTION)
        self.create_subscription(JointState, "/joint_states", self.on_joint,
                                 qos_profile_sensor_data)
        self.create_subscription(Clock, "/clock", self.on_clock,
                                 qos_profile_sensor_data)
        self.create_subscription(JointTrajectoryControllerState,
                                 "/crane_plus_arm_controller/controller_state",
                                 self.on_reference, qos_profile_sensor_data)
        self.create_subscription(JointTrajectoryControllerState,
                                 "/crane_plus_gripper_controller/controller_state",
                                 self.on_reference, qos_profile_sensor_data)
        self.create_subscription(TFMessage, POSE_TOPIC, self.on_pose,
                                 qos_profile_sensor_data)

    def on_joint(self, message):
        self.joint = joint_map(message.name, message.position)

    def on_clock(self, message):
        self.sim_time = message.clock.sec + message.clock.nanosec * 1e-9

    def on_reference(self, message):
        self.reference.update(controller_reference(message))

    def on_pose(self, message):
        candidates = []
        for transform in message.transforms:
            translation = transform.transform.translation
            radial = math.hypot(translation.x, translation.y)
            if translation.z > 0.9 and radial > 0.04:
                candidates.append(transform)
        if not candidates:
            return
        transform = candidates[0].transform
        self.object_pose = {
            "frame_id": "world",
            "child_frame_id": "wood_cube_5cm",
            "position": [transform.translation.x, transform.translation.y,
                         transform.translation.z],
            "orientation": [transform.rotation.x, transform.rotation.y,
                            transform.rotation.z, transform.rotation.w],
        }

    def ready(self) -> bool:
        return (self.sim_time is not None and self.object_pose is not None
                and all(name in self.joint and name in self.reference for name in JOINTS))

    def spin_step(self, timeout: float = 0.02):
        rclpy.spin_once(self, timeout_sec=timeout)
        if self.sample_stream is None or not self.ready():
            return
        now = time.monotonic()
        if now - self.last_sample_wall < 0.1 or self.sim_time == self.last_sample_sim:
            return
        row = {
            "sim_time": self.sim_time,
            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
            "phase": self.phase,
            "joint_positions": {name: self.joint[name] for name in JOINTS},
            "controller_reference": {name: self.reference[name] for name in JOINTS},
            "object_pose": self.object_pose,
        }
        self.sample_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.sample_stream.flush()
        self.sample_count += 1
        self.last_sample_wall = now
        self.last_sample_sim = self.sim_time

    def wait(self, seconds: float):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.spin_step()

    def wait_until_ready(self, timeout: float = 15.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.spin_step()
            if self.ready():
                return
        raise RuntimeError("ROS state or Gazebo cube pose did not become ready")

    def send(self, client, joints, positions, seconds: float):
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Trajectory action server is unavailable")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joints
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        whole_seconds = int(seconds)
        point.time_from_start.sec = whole_seconds
        point.time_from_start.nanosec = int((seconds - whole_seconds) * 1e9)
        goal.trajectory.points = [point]
        goal_future = client.send_goal_async(goal)
        deadline = time.monotonic() + seconds + 12.0
        while not goal_future.done() and time.monotonic() < deadline:
            self.spin_step()
        handle = goal_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("Trajectory goal was rejected")
        result_future = handle.get_result_async()
        while not result_future.done() and time.monotonic() < deadline:
            self.spin_step()
        if not result_future.done():
            raise RuntimeError("Trajectory action timed out")
        result = result_future.result().result
        if result.error_code != 0:
            raise RuntimeError("Trajectory failed: " + result.error_string)

    def move_arm_to(self, target, seconds: float):
        solution = inverse_kinematics(target, True)
        if solution is None or not all(joint_in_range(solution)):
            raise RuntimeError(f"No valid IK solution for {target}")
        self.send(self.arm_client, ARM, solution, seconds)

    def begin_episode(self, path: Path):
        path.mkdir(parents=True, exist_ok=False)
        self.sample_stream = (path / "samples.jsonl").open("x", encoding="utf-8")
        self.sample_count = 0
        self.last_sample_wall = 0.0
        self.last_sample_sim = None

    def end_episode(self):
        if self.sample_stream:
            self.sample_stream.close()
        self.sample_stream = None


def start_pose_bridge() -> subprocess.Popen:
    command = [
        "ros2", "run", "ros_gz_bridge", "parameter_bridge",
        f"{POSE_TOPIC}@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
    ]
    return subprocess.Popen(command, stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT, start_new_session=True)


def set_cube_pose(x: float, y: float):
    request = (f'name: "wood_cube_5cm", position: {{x: {x}, y: {y}, z: 1.05}}, '
               'orientation: {w: 1.0}')
    result = subprocess.run([
        "ign", "service", "-s", "/world/default/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean", "--timeout", "5000",
        "--req", request,
    ], check=True, capture_output=True, text=True)
    if "data: true" not in result.stdout:
        raise RuntimeError("Gazebo rejected cube reset: " + result.stdout)


def write_metadata(path: Path, document: dict):
    temp = path / "metadata.json.tmp"
    temp.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    temp.replace(path / "metadata.json")


def main():
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be positive")
    rng = random.Random(args.seed)
    bridge = start_pose_bridge()
    rclpy.init()
    collector = Collector(args.root.expanduser().resolve())
    results = []
    try:
        collector.wait_until_ready()
        for episode_index in range(args.episodes):
            tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            episode_path = collector.root / f"episode_{tag}_{episode_index:03d}"
            collector.sample_count = 0
            target_x = rng.uniform(-args.x_range, args.x_range)
            target_y = rng.uniform(args.y_center - args.y_range,
                                   args.y_center + args.y_range)
            label = "failure"
            error = None
            initial_pose = final_pose = None
            try:
                collector.send(collector.hand_client, [HAND], [-0.5236], 1.2)
                collector.send(collector.arm_client, ARM, [0.0, 0.0, 0.0, 0.0], 2.5)
                set_cube_pose(target_x, target_y)
                collector.wait(1.0)
                initial_pose = json.loads(json.dumps(collector.object_pose))
                cube_x, cube_y, _ = initial_pose["position"]

                collector.begin_episode(episode_path)
                collector.phase = "approach"
                collector.wait(0.3)
                collector.move_arm_to([cube_x, cube_y, 0.18, 0.0], 2.0)
                collector.phase = "descend"
                collector.move_arm_to([cube_x, cube_y, 0.13, -0.8], 1.4)
                collector.move_arm_to([cube_x, cube_y, 0.09, -1.2], 1.4)
                collector.move_arm_to([cube_x, cube_y, 0.07, -math.pi / 2], 1.4)
                collector.phase = "close"
                collector.send(collector.hand_client, [HAND], [0.1745], 1.4)
                collector.wait(0.4)
                collector.phase = "lift"
                collector.move_arm_to([cube_x, cube_y, 0.11, -1.2], 2.2)
                collector.wait(1.2)
                final_pose = json.loads(json.dumps(collector.object_pose))
                lift_height = final_pose["position"][2] - initial_pose["position"][2]
                label = "success" if lift_height >= args.success_height else "failure"
            except Exception as exception:
                error = str(exception)
                lift_height = None
            finally:
                collector.end_episode()

            episode_path.mkdir(parents=True, exist_ok=True)
            metadata = {
                "format": "crane_grasp_capture_v2_auto",
                "label": label,
                "success_source": "gazebo_cube_lift_height",
                "success_height_m": args.success_height,
                "measured_lift_height_m": lift_height,
                "sample_count": collector.sample_count,
                "joint_order": JOINTS,
                "observation_order": JOINTS + ["object_x", "object_y", "object_z"],
                "phases": PHASES,
                "pose_topic": POSE_TOPIC,
                "cube_target_xy": [target_x, target_y],
                "cube_pose_initial": initial_pose,
                "cube_pose_final": final_pose,
                "episode_index": episode_index,
                "seed": args.seed,
                "error": error,
            }
            write_metadata(episode_path, metadata)
            results.append(metadata)
            lift_text = "n/a" if lift_height is None else f"{lift_height:.4f}m"
            print(f"[{episode_index + 1}/{args.episodes}] {label} "
                  f"xy=({target_x:+.4f},{target_y:+.4f}) lift={lift_text} "
                  f"samples={collector.sample_count}", flush=True)

        summary = {
            "format": "crane_grasp_collection_v1",
            "seed": args.seed,
            "requested_episodes": args.episodes,
            "successes": sum(item["label"] == "success" for item in results),
            "failures": sum(item["label"] != "success" for item in results),
            "x_range": args.x_range,
            "y_center": args.y_center,
            "y_range": args.y_range,
            "success_height_m": args.success_height,
        }
        write_metadata(collector.root, summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        collector.end_episode()
        collector.destroy_node()
        rclpy.shutdown()
        if bridge.poll() is None:
            os.killpg(bridge.pid, signal.SIGTERM)
            try:
                bridge.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(bridge.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
