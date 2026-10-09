#!/usr/bin/env bash
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
# c1c4.sh <container> <label> <outdir> : 4 reps (rep0 = discard) of decode C1 (5 prompts) and C4 (20 prompts), 8K random in / 1,024 forced out,
# fixed seeds (same prompt set every rep), plus DSpark acceptance from /metrics deltas. DSv4.1 C1 ambiguity check 2026-10-07.
set -uo pipefail
CT=$1; LABEL=$2; OUT=$3; mkdir -p "$OUT"
B=(vllm bench serve --backend vllm --base-url http://127.0.0.1:8009 --endpoint /v1/completions --model deepseek-v4.1-flash --tokenizer /model
   --tokenizer-mode deepseek_v41 --trust-remote-code --dataset-name random --random-range-ratio 0 --temperature 0 --random-input-len 8192
   --random-output-len 1024 --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir /tmp/c1c4)
docker exec "$CT" mkdir -p /tmp/c1c4
m(){ curl -s http://127.0.0.1:8009/metrics | awk "/^vllm:spec_decode_num_(draft|accepted)_tokens_total/{print \$2}" | paste -sd" "; }
for rep in 0 1 2 3; do for C in 1 4; do
  a=$(m); docker exec "$CT" "${B[@]}" --max-concurrency $C --num-prompts $((C==1?5:20)) --num-warmups 0 --seed $((1000+C)) \
    --result-filename "$LABEL-r$rep-c$C.json" > "$OUT/$LABEL-r$rep-c$C.log" 2>&1; b=$(m)
  echo "$LABEL rep$rep C$C $(grep -E "Output token throughput" "$OUT/$LABEL-r$rep-c$C.log" | awk "{print \$NF}") tpot=$(grep "Mean TPOT" "$OUT/$LABEL-r$rep-c$C.log" | awk "{print \$NF}") ttft=$(grep "Mean TTFT" "$OUT/$LABEL-r$rep-c$C.log" | awk "{print \$NF}") draft_acc_before=[$a] after=[$b]" | tee -a "$OUT/summary.txt"
done; done
docker cp "$CT:/tmp/c1c4/." "$OUT/"
