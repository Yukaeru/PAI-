#!/usr/bin/env bash
set -euo pipefail

episodes="${1:-10}"
container="${ROS2_CONTAINER:-pai_ros2}"
workspace_root="$(cd "$(dirname "$0")/.." && pwd)"
model="${BC_MODEL:-$workspace_root/models/bc_grasp_v1/policy.ts}"
stamp="$(date +%Y%m%d_%H%M%S)"
name="bc_grasp_eval_${stamp}"
host_output="$workspace_root/evaluations/$name"
container_output="/home/ubuntu/bc_evaluations/$name"
container_model="/home/ubuntu/bc_tools/policy.ts"
container_script="/home/ubuntu/bc_tools/evaluate_bc_policy.py"
video="$container_output/gazebo.mp4"
pid_file="/tmp/${name}_ffmpeg.pid"
bridge_pid_file="/tmp/${name}_pose_bridge.pid"

if ! [[ "$episodes" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 [positive_episode_count]" >&2
  exit 1
fi
if [[ ! -f "$model" ]]; then
  echo "Model not found: $model" >&2
  exit 1
fi
if ! docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null | grep -qx true; then
  echo "Container is not running: $container" >&2
  exit 1
fi

mkdir -p "$workspace_root/evaluations"
docker exec -u ubuntu "$container" mkdir -p /home/ubuntu/bc_tools "$container_output"
docker cp "$workspace_root/tools/evaluate_bc_policy.py" "$container:$container_script"
docker cp "$model" "$container:$container_model"

# A plain CRANE+ Gazebo launch bridges /clock but not the dynamic pose topic.
# Start a temporary pose bridge only when no ROS publisher is already present.
publisher_count="$(docker exec -u ubuntu "$container" bash -lc \
  "source /opt/ros/humble/setup.bash; source /home/ubuntu/airobot_ws/install/setup.bash; \
   ros2 topic info /world/default/dynamic_pose/info 2>/dev/null | \
   sed -n 's/^Publisher count: //p'" || true)"
publisher_count="${publisher_count:-0}"
if [[ "$publisher_count" -lt 1 ]]; then
  docker exec -u ubuntu "$container" bash -lc \
    "source /opt/ros/humble/setup.bash; source /home/ubuntu/airobot_ws/install/setup.bash; \
     ros2 run ros_gz_bridge parameter_bridge \
     '/world/default/dynamic_pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V' \
     >'$container_output/pose_bridge.log' 2>&1 & echo \$! >'$bridge_pid_file'"
  sleep 2
fi

# Record the current VNC/X11 desktop. xrandr is queried because noVNC may resize it.
docker exec -u ubuntu "$container" bash -lc "
  size=\$(DISPLAY=:0 xrandr --current | sed -n 's/.*current \([0-9][0-9]*\) x \([0-9][0-9]*\).*/\1x\2/p' | head -1)
  test -n \"\$size\"
  DISPLAY=:0 ffmpeg -hide_banner -loglevel warning -nostdin \
    -f x11grab -draw_mouse 0 -framerate 20 -video_size \"\$size\" -i :0.0 \
    -c:v libx264 -preset veryfast -crf 24 -pix_fmt yuv420p -y '$video' \
    >'$container_output/ffmpeg.log' 2>&1 &
  echo \$! >'$pid_file'
"

stop_recorder() {
  docker exec "$container" bash -lc \
    "if test -f '$pid_file'; then kill -INT \$(cat '$pid_file') 2>/dev/null || true; fi" \
    >/dev/null 2>&1 || true
  for _ in $(seq 1 30); do
    if ! docker exec "$container" bash -lc \
      "test -f '$pid_file' && kill -0 \$(cat '$pid_file') 2>/dev/null"; then
      break
    fi
    sleep 0.2
  done
}
stop_pose_bridge() {
  docker exec "$container" bash -lc \
    "if test -f '$bridge_pid_file'; then kill -TERM \$(cat '$bridge_pid_file') 2>/dev/null || true; fi" \
    >/dev/null 2>&1 || true
}
cleanup() {
  stop_recorder
  stop_pose_bridge
}
trap cleanup EXIT INT TERM

docker exec -u ubuntu "$container" bash -lc \
  "source /opt/ros/humble/setup.bash; source /home/ubuntu/airobot_ws/install/setup.bash; \
   python3 '$container_script' --model '$container_model' --output '$container_output/run' \
   --episodes '$episodes'"

stop_recorder
stop_pose_bridge
trap - EXIT INT TERM
docker cp "$container:$container_output" "$host_output"
echo "$host_output"
