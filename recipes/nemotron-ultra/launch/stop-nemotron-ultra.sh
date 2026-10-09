#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# stop-nemotron-ultra.sh -- teardown. For MODE=tp run it on rank 0 FIRST, then on rank 1. Stops by container name only.
set -uo pipefail
N=${NAME:-nemotron-ultra}
docker stop -t 60 "$N" >/dev/null 2>&1; docker rm "$N" >/dev/null 2>&1
sleep 5
GB300=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " '/GB300/{print $1;exit}')
echo "$(date '+%F %T') GB300 memory.used: $(nvidia-smi --id="$GB300" --query-gpu=memory.used --format=csv,noheader)"
grep -E "MemAvailable|Shmem:" /proc/meminfo   # Shmem should drop back by the offloaded GiB after a MODE=one run
