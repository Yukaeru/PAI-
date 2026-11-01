#!/usr/bin/env bash
set -euo pipefail

container="${ROS2_CONTAINER:-pai_ros2}"
workspace_root="$(cd "$(dirname "$0")/.." && pwd)"
requirements="$workspace_root/requirements-rl.txt"

if ! docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null | grep -qx true; then
  echo "Container is not running: $container" >&2
  exit 1
fi

docker cp "$requirements" "$container:/tmp/requirements-rl.txt"
if ! docker exec "$container" dpkg-query -W -f='${Status}' python3.10-venv \
  2>/dev/null | grep -q 'install ok installed'; then
  docker exec "$container" apt-get update
  docker exec "$container" apt-get install -y python3.10-venv
fi
docker exec -u ubuntu "$container" bash -lc '
  if [[ ! -x /home/ubuntu/ppo_venv/bin/python ]]; then
    python3 -m venv --system-site-packages /home/ubuntu/ppo_venv
  fi
  /home/ubuntu/ppo_venv/bin/python -m pip install --upgrade pip
  /home/ubuntu/ppo_venv/bin/python -m pip install -r /tmp/requirements-rl.txt
  source /opt/ros/humble/setup.bash
  /home/ubuntu/ppo_venv/bin/python -c \
    "import gymnasium, stable_baselines3, rclpy, torch; \
     print(gymnasium.__version__, stable_baselines3.__version__, torch.__version__)"
'
