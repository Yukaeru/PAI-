#!/usr/bin/env bash
set -euo pipefail

episodes="${1:-30}"
container="${ROS2_CONTAINER:-pai_ros2}"
seed="${GRASP_COLLECTION_SEED:-20261102}"
stamp="$(date +%Y%m%d_%H%M%S)"
name="grasp_auto_${stamp}"
container_root="/home/ubuntu/dataset/$name"
workspace_root="$(cd "$(dirname "$0")/.." && pwd)"
host_root="$workspace_root/datasets/$name"

if ! [[ "$episodes" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 [positive_episode_count]" >&2
  exit 1
fi
if ! docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null | grep -qx true; then
  echo "Container is not running: $container" >&2
  exit 1
fi
if docker exec -u ubuntu "$container" pgrep -af '[r]os2 bag record' >/dev/null; then
  echo "Recording is active. Press STOP in the CRANE+ Grasp panel first." >&2
  exit 2
fi

mkdir -p "$workspace_root/datasets"
docker cp "$workspace_root/tools/collect_grasp_episodes.py" \
  "$container:/home/ubuntu/grasp_tools/collect_grasp_episodes.py"

docker exec -u ubuntu "$container" bash -lc \
  "source /opt/ros/humble/setup.bash; source /home/ubuntu/airobot_ws/install/setup.bash; python3 /home/ubuntu/grasp_tools/collect_grasp_episodes.py --root '$container_root' --episodes '$episodes' --seed '$seed'"

docker cp "$container:$container_root" "$host_root"
python3 "$workspace_root/tools/prepare_bc_collection.py" "$host_root"
echo "$host_root"
