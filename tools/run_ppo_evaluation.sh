#!/usr/bin/env bash
set -euo pipefail

episodes="${1:-10}"
container="${ROS2_CONTAINER:-pai_ros2}"
workspace_root="$(cd "$(dirname "$0")/.." && pwd)"
bc_model="${BC_MODEL:-$workspace_root/models/bc_grasp_v1/policy.ts}"
ppo_model="${PPO_MODEL:-$workspace_root/models/ppo_residual_5k_v1/ppo_residual.zip}"
stamp="$(date +%Y%m%d_%H%M%S)"
name="ppo_residual_eval_${stamp}"
host_output="$workspace_root/evaluations/$name"
container_root="/home/ubuntu/ppo_evaluations/$name"
container_tools="/home/ubuntu/ppo_tools"
video="$container_root/gazebo.mp4"
video_pid_file="/tmp/${name}_ffmpeg.pid"
bridge_pid_file="/tmp/${name}_pose_bridge.pid"

if ! [[ "$episodes" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 [positive_episode_count]" >&2
  exit 1
fi
for model in "$bc_model" "$ppo_model"; do
  if [[ ! -f "$model" ]]; then
    echo "Model not found: $model" >&2
    exit 1
  fi
done
if ! docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null | grep -qx true; then
  echo "Container is not running: $container" >&2
  exit 1
fi

mkdir -p "$workspace_root/evaluations"
docker exec -u ubuntu "$container" mkdir -p "$container_tools" "$container_root"
docker cp "$workspace_root/tools/train_ppo.py" "$container:$container_tools/train_ppo.py"
docker cp "$workspace_root/tools/evaluate_ppo_policy.py" \
  "$container:$container_tools/evaluate_ppo_policy.py"
docker cp "$bc_model" "$container:$container_tools/bc_policy.ts"
docker cp "$ppo_model" "$container:$container_tools/ppo_residual.zip"

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

docker exec -u ubuntu "$container" ign service \
  -s /world/default/control \
  --reqtype ignition.msgs.WorldControl \
  --reptype ignition.msgs.Boolean \
  --timeout 5000 --req 'pause: false' >/dev/null

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

evaluation_status=0
docker exec -u ubuntu "$container" bash -lc \
  "source /opt/ros/humble/setup.bash; source /home/ubuntu/airobot_ws/install/setup.bash; \
   cd '$container_tools'; /home/ubuntu/ppo_venv/bin/python evaluate_ppo_policy.py \
   --bc-model bc_policy.ts --ppo-model ppo_residual.zip \
   --output '$container_root/run' --episodes '$episodes'" \
  || evaluation_status=$?

stop_recorder
stop_pose_bridge
trap - EXIT INT TERM
docker cp "$container:$container_root" "$host_output"
echo "$host_output"
exit "$evaluation_status"
