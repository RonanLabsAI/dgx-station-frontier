#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# pbench.sh OUTDIR DS MODE -- V4P (2026-10-09): PUBLIC real-text decode bench for DeepSeek-V4-Pro, run on the rank-0
# Station against the rank-0 container. Derived from recipes/dsv41-flash-sidecar/bench/realtext/realtext_bench.sh, same hygiene:
#   DS=long  : CNN/DailyMail test articles concatenated to 8,190-8,191 tokens, 1,024 FORCED output tokens (--ignore-eos)
#   DS=short : ShareGPT V3 first human turns (12-498 tokens), NATURAL output (EOS honoured), max 1,024
#   MODE=off : thinking off (slots c1-r1..3 / c16-r1);  MODE=on : thinking on (slots tc1-r1..3 / tc16-r1)
#   Pools: $V4P/pool (build_pool_v4p.py, V4-Pro tokenizer + encoding_dsv4.py) and $V4P/pylib (pandas for the bench
#   client; the image lacks vllm[bench]), copied into the container.
#   * vllm bench serve --dataset-name custom --skip-chat-template --disable-shuffle --no-oversample, /v1/completions, T=0
#   * every invocation its own seed (NONCE + index) AND its own disjoint prompt slice; warm-up = separate invocation on
#     its own slice (<= 64 out); --num-warmups 0; prefix-hit guard from /metrics deltas (> 2% -> retry on the try2 slice,
#     still > 2% -> CONTAMINATED, never reported); DSpark acceptance from /metrics spec_decode_* deltas;
#   * real concurrency per run from the gauge file ($GAUGE, gauge.py): KV 9.05 GiB holds ~3.5 long (8K+1K) requests,
#     so long "C16" is KV-capped -- reported with its measured mean running count.
# decode = C x 1000 / mean TPOT (as every earlier receipt); out = vLLM output token throughput.
set -uo pipefail
O=${1:?outdir}; DS=${2:?long|short}; MODE=${3:?off|on}
CT=${CT:-dsv4pro-e3}; P=${PORT:-8010}; M=${MODEL_NAME:-deepseek-v4-pro}; MAXHIT=${MAXHIT:-2}; REPS=${REPS:-3}
V=${V4P:-$HOME/dsv4pro-public}; GAUGE=${GAUGE:-}; GAUGE_PY=${GAUGE_PY:-$(dirname "$0")/gauge.py}
NONCE=${NONCE:-$(( ($(date +%s) % 1000000) * 1000 ))}; IDX=0
PFX=$([ "$MODE" = on ] && echo t || echo "")
NP1=${NP1:-3}; NP16=${NP16:-$([ "$DS" = long ] && echo 16 || echo 32)}
mkdir -p "$O"; S="$O/summary.txt"; TM="$O/timings.tsv"
echo "== $(date '+%F %T %Z') $CT :$P $M ds=$DS mode=$MODE nonce=$NONCE np1=$NP1 np16=$NP16" | tee -a "$S"
nextseed() { IDX=$((IDX + 1)); SEED=$((NONCE + IDX)); }
met() { curl -s http://127.0.0.1:$P/metrics | awk '
  /^vllm:prefix_cache_(hits|queries)_total/ || /^vllm:spec_decode_num_(accepted_tokens|drafts|draft_tokens)_total[{ ]/ {
    split($1,a,"{"); v[a[1]]+=$NF }
  END{printf "%d %d %d %d %d\n", v["vllm:prefix_cache_hits_total"], v["vllm:prefix_cache_queries_total"],
      v["vllm:spec_decode_num_accepted_tokens_total"], v["vllm:spec_decode_num_drafts_total"], v["vllm:spec_decode_num_draft_tokens_total"]}'; }
stats() { python3 -c "
a=list(map(float,'$1'.split())); b=list(map(float,'$2'.split())); d=[y-x for x,y in zip(a,b)]
hit=f'{100*d[0]/d[1]:.2f}' if d[1]>0 else 'NA'
acc=f'{1+d[2]/d[3]:.3f}' if d[3]>0 else 'NA'; rate=f'{100*d[2]/d[4]:.1f}' if d[4]>0 else 'NA'
print(hit, acc, rate)"; }
docker exec "$CT" mkdir -p /tmp/v4p
docker exec "$CT" test -d /tmp/v4p/pylib/pandas || docker cp "$V/pylib" "$CT:/tmp/v4p/pylib" >/dev/null
docker exec "$CT" test -f /tmp/v4p/pool/manifest.json || docker cp "$V/pool" "$CT:/tmp/v4p/pool" >/dev/null
docker exec "$CT" test -f "/tmp/v4p/pool/$DS/${PFX}c1-r1-try1.jsonl" || { echo "pool missing in container" | tee -a "$S"; exit 3; }
B=(docker exec -e PYTHONPATH=/tmp/v4p/pylib "$CT" vllm bench serve --backend vllm --base-url http://127.0.0.1:$P
   --endpoint /v1/completions --model "$M" --tokenizer /model --trust-remote-code --temperature 0
   --num-warmups 0 --ready-check-timeout-sec 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90
   --save-result --result-dir /tmp/v4p/res-$DS-$MODE)
bench1() { # LABEL C NP OUTLEN FILE
  # (V4P 2026-10-09 bug: a single `local ... F=$5 dsargs=(... $F ...)` expands dsargs before F is set -> "F: unbound variable")
  local L=$1 C=$2 NP=$3 OL=$4 F=$5
  local dsargs=(--dataset-name custom --dataset-path "/tmp/v4p/pool/$DS/$F" --custom-output-len "$OL"
                --skip-chat-template --disable-shuffle --no-oversample)
  [ "$DS" = long ] && dsargs+=(--ignore-eos)
  "${B[@]}" "${dsargs[@]}" --max-concurrency "$C" --num-prompts "$NP" --seed "$SEED" --result-filename "$L.json" > "$O/$L.log" 2>&1; }
run() { # LABEL SLOT C NP
  local L=$1 SL=$2 C=$3 NP=$4 OL=1024 try m0 m1 H ACC AR T OT TT F TIN TOUT DUR TAG WNP t0 t1 WS G
  for try in 1 2; do
    WNP=$(( C < 2 ? 1 : (C > 4 ? 4 : C) ))
    nextseed; WS=$SEED
    bench1 "$L-warm$try" "$C" "$WNP" 64 "$SL-try$try-warm.jsonl"
    nextseed; m0=$(met); t0=$(date +%s.%N)
    bench1 "$L" "$C" "$NP" "$OL" "$SL-try$try.jsonl"
    t1=$(date +%s.%N); m1=$(met)
    read -r H ACC AR <<< "$(stats "$m0" "$m1")"; TAG=""
    [ "$H" != NA ] && awk -v h="$H" -v m="$MAXHIT" 'BEGIN{exit !(h>m)}' && TAG=" CONTAMINATED(>${MAXHIT}%)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$DS" "$MODE" "$L" "try$try" "$t0" "$t1" "${TAG:-ok}" >> "$TM"
    [ -z "$TAG" ] && break
    [ "$try" = 2 ] && break
    mv "$O/$L.log" "$O/$L-try$try-CONTAMINATED.log"
    echo "$L try$try seed $SEED hit $H% -> retry with a new slice + seed" | tee -a "$S"
  done
  G=$([ -n "$GAUGE" ] && python3 "$GAUGE_PY" --window "$GAUGE" "$t0" "$t1" || echo "running n/a")
  T=$(awk '/Mean TPOT/{print $NF}' "$O/$L.log"); TT=$(awk '/Mean TTFT/{print $NF}' "$O/$L.log")
  OT=$(awk '/Output token throughput/{print $NF}' "$O/$L.log"); F=$(awk '/Failed requests/{print $NF}' "$O/$L.log")
  TIN=$(awk '/Total input tokens/{print $NF}' "$O/$L.log"); TOUT=$(awk '/Total generated tokens/{print $NF}' "$O/$L.log")
  DUR=$(awk '/Benchmark duration/{print $NF}' "$O/$L.log")
  python3 - "$DS" "$MODE" "$L" "$SEED" "$WS" "$C" "${T:-0}" "${OT:-?}" "${TT:-?}" "${F:-?}" "$H" "$ACC" "$AR" "${TIN:-?}" "${TOUT:-?}" "${DUR:-?}" "$try" "$TAG" "$G" <<'PY' | tee -a "$S"
import sys
ds,mode,L,seed,ws,c,t,ot,tt,f,h,acc,ar,tin,tout,dur,tr,tag,g=sys.argv[1:]; t=float(t)
print(f"{ds} think-{mode} {L} try{tr} seed {seed} warmseed {ws}: decode_agg {int(c)*1000/t if t else 0:.2f} tok/s (C{c} x 1000/meanTPOT {t} ms) | "
      f"out_tput {ot} | meanTTFT {tt} ms | in {tin} out {tout} tok in {dur} s | dspark_acc_len {acc} draft_acc {ar}% | "
      f"failed {f} | prefix_hit {h}%{tag} | {g}")
PY
}
for r in $(seq 1 "$REPS"); do run "$DS-$MODE-c1-r$r" "${PFX}c1-r$r" 1 "$NP1"; done
[ "${SKIP16:-0}" = 1 ] || run "$DS-$MODE-c16-r1" "${PFX}c16-r1" 16 "$NP16"
docker cp "$CT:/tmp/v4p/res-$DS-$MODE/." "$O/" >/dev/null 2>&1
echo "DONE $DS $MODE $(date '+%F %T')" | tee -a "$S"
