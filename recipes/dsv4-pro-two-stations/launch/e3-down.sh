#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# e3-down.sh [sidecars] -- run on the rank-0 Station: stop rank 0 then rank 1 (NAME=dsv4pro-e3 launch/stop-dsv4pro.sh), report GB300 MiB.
# With "sidecars", also remove both sidecar containers and their shared buffers.
R=${PEER_SSH:?user@rank-1 rail IP}; D=${E3DIR:-$HOME/dgx-station-frontier/recipes/dsv4-pro-two-stations}/launch
NAME=dsv4pro-e3 bash "$D/stop-dsv4pro.sh"
ssh -o BatchMode=yes $R "NAME=dsv4pro-e3 bash $D/stop-dsv4pro.sh"
docker rm dsv4pro-e3 >/dev/null 2>&1; ssh -o BatchMode=yes $R "docker rm dsv4pro-e3 >/dev/null 2>&1"
if [ "${1:-}" = sidecars ]; then
  docker rm -f dsv4pro-e3-sidecar >/dev/null 2>&1; rm -f /dev/shm/e3_dsv4pro_peer
  ssh -o BatchMode=yes $R "docker rm -f dsv4pro-e3-sidecar >/dev/null 2>&1; rm -f /dev/shm/e3_dsv4pro_peer"
fi
for h in local $R; do
  c="nvidia-smi --query-gpu=name,memory.used --format=csv,noheader"
  if [ $h = local ]; then echo "$(hostname): $($c | tr '\n' ';')"; else echo "right: $(ssh -o BatchMode=yes $h "$c" | tr '\n' ';')"; fi
done
