#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# capture_public.sh LOGDIR SEEDBASE -- V4P (2026-10-09): route-count capture for the PUBLIC-ONLY map, run on the rank-0
# Station against a counters-on, no-drafter E3-A16 server (E3_COUNT=1). This is the capture step of our placement script
# with the public workloads only (our run also captured a private set and GSM8K; the public-only map never used them).
# Snapshot method: a flush request (flush.py: one ~3.6K-token prompt, max_tokens 1; its routes land in bucket 2 only)
# and a copy of both ranks' cumulative counters before and after every workload ("pre-W" / "W").
#   cnn-off / cnn-on : pool long/train-try1 / ttrain-try1 (16 x 8K CNN/DM), 512 forced tokens
#   sg-off  / sg-on  : pool short/train-try1 / ttrain-try1 (48 ShareGPT), natural output, max 512 / 768
# Training slots are disjoint from every measured slot (see build_pool_v4p.py). Output: LOGDIR/snaps-capture/ for mixfit.py.
# env: PEER_SSH=user@<rank-1 rail address> (as for launch/e3-up.sh), V4P=<folder holding pool/> (default ~/dsv4pro-public)
set -uo pipefail
LOGDIR=${1:?}; SB=${2:?seedbase}; WHAT=capture; P=8010; M=deepseek-v4-pro; R=${PEER_SSH:?user@rank-1 rail address}; N=dsv4pro-e3
V=${V4P:-$HOME/dsv4pro-public}; H=$(cd "$(dirname "$0")" && pwd); API=http://127.0.0.1:$P
O=$LOGDIR/$WHAT; SN=$LOGDIR/snaps-$WHAT
mkdir -p "$O" "$SN"; S="$O/summary.txt"
met() { curl -s $API/metrics | awk '/^vllm:prefix_cache_(hits|queries)_total/{split($1,a,"{"); v[a[1]]+=$NF} END{printf "%d %d\n", v["vllm:prefix_cache_hits_total"], v["vllm:prefix_cache_queries_total"]}'; }
hit() { python3 -c "h0,q0,h1,q1=map(float,'$1 $2'.split()); dq=q1-q0; r=(h1-h0)/dq if dq>0 else 0; print(f'{100*r:.2f}% ({int(h1-h0)}/{int(dq)} tok)' + (' CONTAMINATED' if r>0.02 else ''))"; }
snap() { local f; f=$(python3 "$H/flush.py" $API); sleep 8
  cp "$LOGDIR/e3-counts-rank0.json" "$SN/$1-r0.json"; ssh -o BatchMode=yes $R "cat $LOGDIR/e3-counts-rank1.json" > "$SN/$1-r1.json"
  echo "$1 $f" >> "$SN/index.txt"; }
rb() { python3 "$H/rbench.py" $API $M "$@"; }
work() { # NAME rbench-args...
  local W=$1; shift; local m0 m1; snap "pre-$W"; m0=$(met)
  rb "$O/$W.json" "$@" > "$O/$W.log" 2>&1; m1=$(met); snap "$W"
  echo "$W: $(tail -1 "$O/$W.log") hit $(hit "$m0" "$m1")" | tee -a "$S"; }
B=(docker exec "$N" vllm bench serve --backend vllm --base-url $API --model $M --tokenizer /model --trust-remote-code)
"${B[@]}" --dataset-name random --random-input-len 512 --random-output-len 64 --num-prompts 4 --max-concurrency 4 --ignore-eos \
  --seed $((SB + 900)) > "$O/warmup.log" 2>&1
echo "== $(date '+%F %T') $WHAT seedbase $SB" | tee -a "$S"
PL=$V/pool
work cnn-off 16 16 --set pool --pool $PL/long/train-try1.jsonl --idx 0-15 --max 512
work cnn-on  16 16 --set pool --pool $PL/long/ttrain-try1.jsonl --idx 0-15 --max 512
work sg-off  16 48 --set pool --pool $PL/short/train-try1.jsonl --idx 0-47 --max 512 --ignore-eos 0
work sg-on   16 48 --set pool --pool $PL/short/ttrain-try1.jsonl --idx 0-47 --max 768 --ignore-eos 0
echo "DONE $(date +%T)" | tee -a "$S"
