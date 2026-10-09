#!/bin/bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# rtxload.sh -- keep the RTX PRO 6000 busy during the GB300 GEMM power test: 16 concurrent long generations against any
# OpenAI-compatible server running on the 6000.  env: URL (default http://127.0.0.1:8888/v1/chat/completions), MODEL.
URL=${URL:-http://127.0.0.1:8888/v1/chat/completions}; MODEL=${MODEL:?served model name on the RTX}
for i in $(seq 1 16); do
  curl -s -o /dev/null "$URL" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a long, detailed essay on the history of the printing press, at least 2000 words.\"}],\"max_tokens\":1200}" &
done; wait
