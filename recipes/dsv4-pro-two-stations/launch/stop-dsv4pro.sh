#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# stop-dsv4pro.sh -- stop the DS-V4-Pro container on THIS host. Run on the rank-0 (API) Station FIRST, then on rank 1.
# Then verify the GB300 returns to ~22 MiB. Prints the GB300 memory before and after (polls up to 60 s).
N=${NAME:-dsv4pro-e3}   # launch-dsv4pro.sh default container name
GB300=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " '/GB300/{print $1;exit}')
mem() { nvidia-smi --id="$GB300" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
echo "$(date +%T) $(hostname) before: $(mem) MiB"
docker stop -t 180 "$N" >/dev/null 2>&1 || true
for i in $(seq 1 12); do m=$(mem); [ "$m" -lt 1024 ] && break; sleep 5; done
echo "$(date +%T) $(hostname) after: $(mem) MiB (container: $(docker inspect -f '{{.State.Status}}' "$N" 2>/dev/null))"
