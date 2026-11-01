#!/usr/bin/env bash
set -euo pipefail

timesteps="${1:-128}"
container="${ROS2_CONTAINER:-pai_ros2}"
workspace_root="$(cd "$(dirname "$0")/.." && pwd)"
bc_model="${BC_MODEL:-$workspace_root/models/bc_grasp_v1/policy.ts}"
stamp="$(date +%Y%m%d_%H%M%S)"
name="ppo_residual_${stamp}"
host_output="$workspace_root/rl_runs/$name"
container_root="/home/ubuntu/ppo_runs/$name"
container_script="/home/ubuntu/ppo_tools/train_ppo.py"
container_model="/home/ubuntu/ppo_tools/bc_policy.ts"
bridge_pid_file="/tmp/${name}_pose_bridge.pid"
video_pid_file="/tmp/${name}_ffmpeg.pid"
video="$container_root/gazebo_training.mp4"

if ! [[ "$timesteps" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 [positive_timestep_count]" >&2
  exit 1
fi
if [[ ! -f "$bc_model" ]]; then
  echo "BC model not found: $bc_model" >&2
  exit 1
fi
if ! docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null | grep -qx true; then
  echo "Container is not running: $container" >&2
  exit 1
fi
if ! docker exec -u ubuntu "$container" bash -lc \
  'source /opt/ros/humble/setup.bash; /home/ubuntu/ppo_venv/bin/python -c \
  "import gymnasium, stable_baselines3, rclpy, torch"' >/dev/null; then
  echo "PPO environment is missing. Install requirements-rl.txt in /home/ubuntu/ppo_venv." >&2
  exit 1
fi

mkdir -p "$workspace_root/rl_runs"
docker exec -u ubuntu "$container" mkdir -p /home/ubuntu/ppo_tools "$container_root"
docker cp "$workspace_root/tools/train_ppo.py" "$container:$container_script"
docker cp "$bc_model" "$container:$container_model"

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
     >'$container_root/pose_bridge.log' 2>&1 & echo \$! >'$bridge_pid_file'"
  sleep 2
fi

stop_pose_bridge() {
  docker exec "$container" bash -lc \
    "if test -f '$bridge_pid_file'; then kill -TERM \$(cat '$bridge_pid_file') 2>/dev/null || true; fi" \
    >/dev/null 2>&1 || true
}
stop_recorder() {
  docker exec "$container" bash -lc \
    "if test -f '$video_pid_file'; then kill -INT \$(cat '$video_pid_file') 2>/dev/null || true; fi" \
    >/dev/null 2>&1 || true
  for _ in $(seq 1 30); do
    if ! docker exec "$container" bash -lc \
      "test -f '$video_pid_file' && kill -0 \$(cat '$video_pid_file') 2>/dev/null"; then
      break
    fi
    sleep 0.2
  done
}
cleanup() {
  stop_recorder
  stop_pose_bridge
}
trap cleanup EXIT INT TERM

if [[ "${PPO_RECORD_VIDEO:-0}" == "1" ]]; then
  docker exec -u ubuntu "$container" bash -lc "
    size=\$(DISPLAY=:0 xrandr --current | sed -n \
      's/.*current \([0-9][0-9]*\) x \([0-9][0-9]*\).*/\1x\2/p' | head -1)
    test -n \"\$size\"
    DISPLAY=:0 ffmpeg -hide_banner -loglevel warning -nostdin \
      -f x11grab -draw_mouse 0 -framerate 20 -video_size \"\$size\" -i :0.0 \
      -c:v libx264 -preset veryfast -crf 24 -pix_fmt yuv420p -y '$video' \
      >'$container_root/ffmpeg.log' 2>&1 &
    echo \$! >'$video_pid_file'
  "
fi

check_arg=""
if [[ "${PPO_CHECK_ONLY:-0}" == "1" ]]; then
  check_arg="--check-only"
fi

docker exec -u ubuntu "$container" bash -lc \
  "source /opt/ros/humble/setup.bash; source /home/ubuntu/airobot_ws/install/setup.bash; \
   /home/ubuntu/ppo_venv/bin/python '$container_script' \
   --bc-model '$container_model' --output '$container_root/run' \
   --timesteps '$timesteps' --eval-episodes '${PPO_EVAL_EPISODES:-1}' $check_arg"

stop_recorder
stop_pose_bridge
trap - EXIT INT TERM
docker cp "$container:$container_root" "$host_output"
echo "$host_output"
