#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# psample.sh <outfile> : sample this Station's GPUs (power, limit, memory, util) every 2 s until killed
exec nvidia-smi --query-gpu=timestamp,name,power.draw,power.limit,memory.used,utilization.gpu --format=csv,noheader -l 2 > "$1"
