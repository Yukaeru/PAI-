#!/usr/bin/env bash
set -euo pipefail

container="${ROS2_CONTAINER:-pai_ros2}"
world="${ROS2_WORLD:-default}"
cube_name="${GRASP_CUBE_NAME:-wood_cube_5cm}"
cube_x="${GRASP_CUBE_X:-0.0}"
cube_y="${GRASP_CUBE_Y:--0.1}"
cube_z="${GRASP_CUBE_Z:-1.05}"

if ! docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null | grep -qx true; then
  echo "Container is not running: $container" >&2
  exit 1
fi

if docker exec -u ubuntu "$container" pgrep -af '[r]os2 bag record' >/dev/null; then
  echo "Recording is active. Press STOP in the CRANE+ Grasp panel first." >&2
  exit 2
fi

echo "Opening gripper and returning the arm to the vertical pose..."
docker exec -i -u ubuntu "$container" bash -lc \
  'source /opt/ros/humble/setup.bash && source /home/ubuntu/airobot_ws/install/setup.bash && python3 -' <<'PY'
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint


def send(node, action_name, joints, positions, seconds):
    client = ActionClient(node, FollowJointTrajectory, action_name)
    if not client.wait_for_server(timeout_sec=10.0):
        raise RuntimeError(f"Action server not available: {action_name}")
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = joints
    point = JointTrajectoryPoint()
    point.positions = positions
    point.time_from_start.sec = seconds
    goal.trajectory.points = [point]
    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future)
    handle = future.result()
    if handle is None or not handle.accepted:
        raise RuntimeError(f"Goal rejected: {action_name}")
    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result
    if result.error_code != 0:
        raise RuntimeError(f"Goal failed: {action_name}: {result.error_string}")


rclpy.init()
node = Node("reset_grasp_scene")
try:
    send(node, "/crane_plus_gripper_controller/follow_joint_trajectory",
         ["crane_plus_joint_hand"], [-0.5236], 2)
    send(node, "/crane_plus_arm_controller/follow_joint_trajectory",
         ["crane_plus_joint1", "crane_plus_joint2",
          "crane_plus_joint3", "crane_plus_joint4"],
         [0.0, 0.0, 0.0, 0.0], 5)
finally:
    node.destroy_node()
    rclpy.shutdown()
PY

echo "Returning $cube_name to its initial position..."
pose_result="$({
  docker exec -u ubuntu "$container" ign service \
    -s "/world/$world/set_pose" \
    --reqtype ignition.msgs.Pose \
    --reptype ignition.msgs.Boolean \
    --timeout 5000 \
    --req "name: \"$cube_name\", position: {x: $cube_x, y: $cube_y, z: $cube_z}, orientation: {w: 1.0}"
} 2>&1)"
printf '%s\n' "$pose_result"
if ! grep -q 'data: true' <<<"$pose_result"; then
  echo "Gazebo did not accept the cube reset." >&2
  exit 3
fi

sleep 2
echo "Reset complete. Select phase 'approach', then press START NEW EPISODE."
