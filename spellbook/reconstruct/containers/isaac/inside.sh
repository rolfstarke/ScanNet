#!/usr/bin/env bash
# Container entry for engines/isaac.py: recorder + graph -> SVO end -> save_ply -> teardown.
# Every wait is bounded (#22). Args: <overall_s> <save_ply_s>
# ISAAC_PRIME=1: stop once the first frame plays (ZED models are optimized by then), no mesh.
OVERALL_S="${1:?overall bound}"
SAVE_PLY_S="${2:?save_ply bound}"
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /opt/ros2_ws/install/local_setup.bash
set -uo pipefail  # after sourcing: the ROS setup files read unset variables
# zed_components links libgxf_serialization (wrapper built with NITROS); not on the ROS path.
export LD_LIBRARY_PATH="/opt/ros/${ROS_DISTRO}/share/isaac_ros_gxf/gxf/lib/serialization:${LD_LIBRARY_PATH}"
mkdir -p "$HOME" /logs/ros
export ROS_LOG_DIR=/logs/ros RCUTILS_COLORIZED_OUTPUT=0 PYTHONUNBUFFERED=1
started=$SECONDS

if [ "${ISAAC_PRIME:-0}" = "1" ]; then
  RECORDER_ARGS=(--stop-at-first-frame --startup-s "$OVERALL_S")
else
  RECORDER_ARGS=()
fi
python3 /isaac/pose_recorder.py --out /output "${RECORDER_ARGS[@]}" > /logs/recorder.log 2>&1 &
REC=$!
ros2 launch /isaac/graph.launch.py > /logs/graph.log 2>&1 &
GRAPH=$!

stop_all() {
  kill -INT "$GRAPH" "$REC" 2>/dev/null
  for _ in $(seq 1 15); do
    kill -0 "$GRAPH" 2>/dev/null || break
    sleep 1
  done
  pkill -KILL -P "$GRAPH" 2>/dev/null
  kill -KILL "$GRAPH" "$REC" 2>/dev/null
}
fail() { echo "isaac: $1" >&2; stop_all; exit "${2:-1}"; }

# Effective parameters, once playback runs (ros2 param dump blocks while nodes start).
(
  until [ -s /output/svo_status.txt ] || [ $((SECONDS - started)) -gt 300 ]; do sleep 2; done
  timeout 90 ros2 param dump /nvblox_node --no-daemon > /output/effective_nvblox.yaml 2>/logs/param_nvblox.log
  timeout 90 ros2 param dump /zed/zed_node --no-daemon > /output/effective_zed.yaml 2>/logs/param_zed.log
) &

while kill -0 "$REC" 2>/dev/null; do
  kill -0 "$GRAPH" 2>/dev/null || fail "graph exited before the SVO ended (see graph.log)" 2
  # robot_state_publisher keeps the launch alive when the composable container dies.
  if [ $((SECONDS - started)) -gt 20 ] && ! pgrep -f component_container_mt >/dev/null; then
    fail "component container died (see graph.log)" 2
  fi
  [ $((SECONDS - started)) -gt "$OVERALL_S" ] && fail "overall bound ${OVERALL_S}s exceeded" 124
  sleep 1
done
wait "$REC"; rc=$?
[ "$rc" -eq 0 ] || fail "recorder exit $rc (3 no first frame, 4 stalled; see recorder.log)" 3
if [ "${ISAAC_PRIME:-0}" = "1" ]; then
  echo "isaac: first frame after $((SECONDS - started))s"
  stop_all
  exit 0
fi
echo "isaac: svo ended after $((SECONDS - started))s: $(cat /output/svo_end.txt)"

save_started=$SECONDS
out=$(timeout "$SAVE_PLY_S" ros2 service call /nvblox_node/save_ply nvblox_msgs/srv/FilePath \
      "{file_path: '/output/mesh.ply'}" 2>&1)
rc=$?
echo "$out" | tail -3
[ "$rc" -eq 0 ] || fail "save_ply rc=$rc within ${SAVE_PLY_S}s" 4
echo "$out" | grep -q "success=True" || fail "save_ply did not report success" 4
[ -s /output/mesh.ply ] || fail "save_ply reported success but mesh.ply is empty" 4
echo "isaac: save_ply took $((SECONDS - save_started))s"
stop_all
exit 0
