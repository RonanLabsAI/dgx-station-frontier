#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# stop-dsv41.sh -- stop every dsv41-* container on THIS Station gracefully, save its log, wait for the GB300 to free.
# Never resets a GPU. vLLM rank 0 can strand HBM at teardown (a public two-Station report saw 51-82 GiB);
# if the GB300 still holds > 8 GiB after 120 s, STOP and tell the operator (a reboot needs the SR4 transceivers pulled first).
set -uo pipefail
mkdir -p "$HOME/dsv41/stopped"
for c in $(docker ps -a --format '{{.Names}}' | grep -E '^dsv41-'); do
  docker logs "$c" > "$HOME/dsv41/stopped/$c-$(date +%Y%m%dT%H%M%S).log" 2>&1 || true
  docker stop -t 120 "$c" >/dev/null 2>&1; docker rm "$c" >/dev/null 2>&1; echo "stopped $c"
done
GB300=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " '/GB300/{print $1;exit}')
for i in $(seq 1 24); do
  U=$(nvidia-smi --id="$GB300" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "$U" -lt 8192 ] && { echo "GB300 free: ${U} MiB"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; exit 0; }
  sleep 5
done
echo "GB300 STILL HOLDS ${U} MiB after 120 s (stranded HBM). Do NOT relaunch on top; investigate before rebooting."; exit 5
