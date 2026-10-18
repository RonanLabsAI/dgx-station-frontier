# Muse Spark Open on two DGX Station GB300s

Meta released Muse Spark Open on 2026-10-15: a mixture-of-experts model with 816B total and 42B active parameters,
NVFP4 routed experts, 62 MoE layers of 256 experts (top 8) and no MTP layers. At TP2 its weights come to about 249 GiB
per rank against 250.7 GiB of GB300 HBM, so it does not fit two GB300s on its own: some routed experts have to live in
Grace memory. This page is one pre-registered window on 2026-10-16 (PT), run the day after release: the model with and
without Inferact's DSpark drafter, on the same public real-text pool and harness as our Kimi K3 rows, then an attempt
to move the Grace-resident experts onto the RTX PRO 6000 (it failed its precision gate).

Every number below is a row in [`results.jsonl`](../results.jsonl) with a scrubbed log excerpt under
[`logs/muse-spark-open/`](../logs/muse-spark-open/).

Built with Muse Spark Open. Licences: [`meta-models/Muse-Spark-Open-816B-A42B-NVFP4`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4)
is under the Muse Model License and Inferact's [`Muse-Spark-Open-DSpark`](https://huggingface.co/Inferact/Muse-Spark-Open-DSpark)
drafter under the `muse-spark-open` licence (model-card metadata, checked 2026-10-17); read them on the model cards. No
weights, quantisations or derived files are included here, only launch flags.

## 1. Setup

| | Arm a | Arm b |
|---|---|---|
| Parallelism | TP2 + expert parallel across both Stations (vLLM mp multi-node), one 400G rail | same |
| GB300 HBM per rank | dense weights, KV, most routed experts (vLLM's `Model loading took` 165.62 GiB) | same experts; 172.41 GiB with the drafter |
| Grace memory per rank | 96.31 GiB of parameters through vLLM's UVA offload (`--cpu-offload-gb 96`) | same |
| Speculative decoding | none | Inferact's DSpark drafter, Meta's recipe config (`num_speculative_tokens` 7, probabilistic draft, block rejection, adaptive verification) |
| Session (PT) | 2026-10-16 01:39-02:34 | 2026-10-16 02:35-03:42 |
| Receipt | [two-stations-realtext.log](../logs/muse-spark-open/two-stations-realtext.log) | same file |

Both arms: image `vllm/vllm-openai:muse-spark-open` (Meta's image for this model; the model is not in a nightly wheel
yet), pinned by its image id, vLLM `0.1.dev22954+ga34ca64c1`, NVFP4 MoE backend `FLASHINFER_TRTLLM` (vLLM's own
choice), context `131072`, `16` sequences, `8192` batched tokens, utilisation `0.92`, BF16 KV, prefix caching on.

**Disclosed deviation: the pre-registered memory fallback F2.** The plan was `64` GiB per rank of
`experts.w13_weight` / `experts.w2_weight` in Grace. That first boot ran out of GPU memory, so the runner took the
pre-registered fallback F2: `96` GiB per rank, offload parameters chosen by vLLM. Both arms ran on F2. The out-of-memory
line and the fallback are in the receipt.

Launch flags (rank 0; rank 1 is the same with `--node-rank 1 --headless`):

```
vllm serve /model --served-model-name muse-spark --reasoning-parser muse --tool-call-parser muse \
  --enable-auto-tool-choice --tensor-parallel-size 2 --pipeline-parallel-size 1 --max-model-len 131072 \
  --max-num-seqs 16 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92 --enable-prefix-caching \
  --disable-custom-all-reduce --distributed-executor-backend mp --nnodes 2 --node-rank 0 \
  --master-addr <rail ip of A> --master-port 29644 --enable-expert-parallel --offload-backend uva \
  --cpu-offload-gb 96 --language-model-only --host 0.0.0.0 --port 8040
# arm b adds:
  --speculative-config '{"method":"dspark","model":"/drafter","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
```

**Measurement.** Real text is the primary measure: the public CNN/DailyMail pool that
[`recipes/kimi-k3/bench/build_pool_k3.py`](../recipes/kimi-k3/bench/build_pool_k3.py) builds (slices `eval-c1-r1` to
`r3` and `eval-c16`), about `8,300` prompt tokens in Muse's chat format, thinking off, `1,024` forced output tokens,
temperature 0, streaming, driven by the unchanged Kimi K3 harness
[`bench/kbench_chat_ds.py`](../recipes/kimi-k3/bench/kbench_chat_ds.py). C1 is three repetitions of three prompts on
disjoint slices; C4 is the first eight prompts of `eval-c16` with four in flight; C16 is all `32`, one run each.
**decode tok/s = C x 1000 / mean TPOT** (the harness's `decode_agg`), not vLLM's output throughput. The pre-registered
C1 statistic is the median of the three repetitions; we also give the mean, which is what the rest of this repository
quotes.

## 2. Results on real text

<!-- results -->
| Measurement | a: no drafter | b: DSpark drafter | b / a | Receipts |
|---|---|---|---|---|
| C1 decode tok/s per user, median of three (pre-registered) | 23.71 | 33.65 | 1.419 | [log](../logs/muse-spark-open/two-stations-realtext.log) |
| C1 decode tok/s per user, mean of the same three | 23.72 | 32.98 | | [log](../logs/muse-spark-open/two-stations-realtext.log) |
| C4 decode tok/s, aggregate (one run) | 58.13 | 76.99 | 1.324 | [log](../logs/muse-spark-open/two-stations-realtext.log) |
| C16 decode tok/s, aggregate (one run) | 117.06 | 145.67 | 1.244 | [log](../logs/muse-spark-open/two-stations-realtext.log) |
| DSpark mean accepted length, C1 repetitions | | 2.089 / 2.231 / 2.333 | | [log](../logs/muse-spark-open/two-stations-realtext.log) |
| DSpark mean accepted length, C16 | | 1.87 | | [log](../logs/muse-spark-open/two-stations-realtext.log) |
| GSM8K, first `200` test rows, reasoning effort high, % | | 97.0 | | [log](../logs/muse-spark-open/two-stations-realtext.log) |
| Coherence, `16` greedy prompts: auto-checked answers correct, of `12` | 11 | 11 | | [log](../logs/muse-spark-open/two-stations-realtext.log) |
| Tool-call smoke, correct of `3` | 3 | 3 | | [log](../logs/muse-spark-open/two-stations-realtext.log) |
| Highest server prefix-cache hit over the real-text runs, % | 1.74 | 1.54 | | [log](../logs/muse-spark-open/two-stations-realtext.log) |
<!-- /results -->

Notes on the table:

- **The C1 repetitions:** a 23.71 / 23.71 / 23.73, b 31.06 / 33.65 / 34.24 decode tok/s, all in the receipt.
- **Acceptance came in below the prediction.** We pre-registered a mean accepted length of `2.5`-`5.0` per verify step
  for the drafter. All three C1 repetitions landed below that band, at 2.089, 2.231 and 2.333, and C16 at 1.87. The
  drafter still met its adoption bar (below), so on real text this is a smaller gain than predicted, not a loss.
- **GSM8K-200 was run on arm b only**, as pre-registered: 194 of 200, one runaway generation counted wrong.
- **Prefix-cache hits** of 1.48-1.74% on every real-text run come from the chat-template header the prompts share. The
  repository's `2%` rule is for random-token runs; these are real text and stay under it anyway.

**Against the pre-registration.**

| Pre-registered | Predicted | Measured | |
|---|---|---|---|
| a, C1 tok/s | 30 (range 18-45) | 23.71 | in range |
| b, C1 tok/s | 50 (range 30-80) | 33.65 | in range, low |
| b / a at C1 | 1.7 (range 1.3-2.3) | 1.419 | in range, low |
| Mean accepted length | 2.5-5.0 | 2.089-2.333 at C1 | **missed low** |
| Drafter adoption bar: b C1 median `>= 1.20x` a, coherence `>= a - 1`, tools `3/3`, GSM8K-200 `>= 90%` | | all met | **adopt** for this model |

(The predicted C16 range was `90`-`320` tok/s for a and `100`-`360` for b; both measured values are inside it.)

## 3. Random-id control (repetitions only)

The same boots also ran the repository's random-token shape (8,192 random ids in, 1,024 forced out, `vllm bench serve`,
a fresh seed per repetition, warm-up on its own seed, `--num-warmups 0`), with **`--temperature 0` (greedy)**. These
are random-token numbers, not real-text speed, and we do not use them as a headline. We publish the repetitions
because arm b's do not average into anything meaningful:

<!-- results -->
| Random ids, C1 decode tok/s per user | Repetition `1` | Repetition `2` | Repetition `3` | Receipt |
|---|---|---|---|---|
| a: no drafter | 23.65 | 23.63 | 23.69 | [log](../logs/muse-spark-open/random-ids-control.log) |
| b: DSpark drafter (greedy random ids) | 46.4 | 24.8 | 56.24 | [log](../logs/muse-spark-open/random-ids-control.log) |
<!-- /results -->

Without a drafter, random ids decode at the real-text speed. With the drafter, greedy decoding of a random prompt
either falls into repetitive output that the drafter predicts (accepted length `3.557` and `4.020`) or does not
(`1.730`), so one repetition is near the no-drafter speed and two are about twice it. Every repetition had a `0.00%`
prefix-cache hit. Our Kimi K3 recipe page explains the same effect in more detail.

## 4. Against Kimi K3 on the same pool and harness

Kimi K3 (int3 experts, `408` GiB per rank in Grace) ran the same real-text slices with the same harness three days
earlier ([recipe](../recipes/kimi-k3/)). Different models and sessions; these ratios say how the two compare as
served on this hardware, not which model is better.

<!-- results -->
| Real text, C1 decode tok/s per user, mean of three | Muse Spark Open | Kimi K3 | Muse / K3 | Receipts |
|---|---|---|---|---|
| No drafter | 23.72 | 18.01 | 1.317 | [Muse](../logs/muse-spark-open/two-stations-realtext.log), [K3](../logs/kimi-k3/vllm-two-stations-dspark.log) |
| With each model's DSpark drafter | 32.98 | 26.23 | 1.257 | [Muse](../logs/muse-spark-open/two-stations-realtext.log), [K3](../logs/kimi-k3/vllm-two-stations-dspark.log) |
<!-- /results -->

## 5. Failed: an RTX PRO 6000 sidecar for the Grace-resident experts (S7)

The obvious next lever is the one that worked for DeepSeek ([sidecar A/B](sidecar-ab.md)): put the experts that sit in
Grace on the RTX PRO 6000 instead and compute them there. For Muse the sidecar runs FlashInfer's CUTLASS NVFP4 W4A4
fused MoE on the RTX PRO 6000 (SM120), prepared by vLLM's own `FLASHINFER_CUTLASS` path, in its own process. Before any
serving boot, a kernel gate (P0) on one Station compares it with the GB300's stock `FLASHINFER_TRTLLM` call on the same
expert weights: speed against the measured Grace call it would replace (bar `<= 0.5x` at `1`, `4` and `16` rows per
call), and output cosine against the stock GB300 output (bar `>= 0.9999`).

- **Attempt 1 (10:14 PT): invalid.** The kit's self-test compared the sidecar with an fp32 emulation and refused to
  serve below `0.9999`. It measured 0.9998983, so no judged row ran.
- **Attempt 2 (13:15 PT): kit bug.** The smoke test raised `invalid device ordinal` before P0; nothing was measured.
- **Attempt 3 (19:43 PT): KILL on precision.** Self-test report-only, cosine judged against the stock GB300 output.

<!-- results -->
| S7 P0, final attempt | `1` row per call | `4` rows | `16` rows | Receipt |
|---|---|---|---|---|
| Sidecar time / measured Grace call time (bar `<= 0.5`) | 0.4599 | 0.2919 | 0.2524 | [log](../logs/muse-spark-open/s7-rtx-sidecar-p0.log) |
| Minimum cosine, sidecar vs stock GB300 output, all judged calls (bar `>= 0.9999`) | 0.9997205 | | | [log](../logs/muse-spark-open/s7-rtx-sidecar-p0.log) |
| Attempt 1 self-test cosine vs fp32 emulation, layer 1 | 0.9998983 | | | [log](../logs/muse-spark-open/s7-rtx-sidecar-p0.log) |
<!-- /results -->

The speed side passed easily: the RTX PRO 6000 would take a quarter to a half of the time of the Grace read it
replaces. The precision side did not. The GB300 kernel agrees with the fp32 emulation to `0.9999971`, the SM120 CUTLASS
path to 0.9998983, and the two kernels agree with each other only to 0.9997205. Both are NVFP4 W4A4 implementations of
the same layer, so the gap is in how they quantise and accumulate, and a cosine that low across `26` layers per token
is not something we would serve without an end-to-end quality check. Per the pre-registration, a P0 KILL ends S7: no
serving boot was run.

## 6. Caveats

- **One window, one boot per arm.** C4 and C16 are single runs. C1 is three repetitions of three prompts.
- **Not recorded:** the weights' and drafter's Hugging Face revisions, and the NVIDIA driver version for this window
  (NCCL reports CUDA driver API `13020`). The image is pinned by the image id `docker inspect` reported at both boots,
  the same method as this repository's other image pins.
- **Not measured:** prefill throughput, long contexts, power, any concurrency above `16`, a run without prefix caching,
  quality beyond GSM8K-200, coherence and the tool-call smoke, and any of the other pre-registered speed-up candidates
  (smaller offload with FP8 KV, the Rust frontend, a batch-size schedule for DSpark, expert parallelism off).
- **Boot is slow:** `1,869` s (a) and `1,977` s (b) from launch to ready, of which vLLM's own weight loading was `80`
  and `75` s. We did not break down the rest.

## Credits

- **Meta** for Muse Spark Open, its vLLM image and its serving recipe (the speculative config above is Meta's).
- **Inferact** for the Muse-Spark-Open-DSpark drafter.
- **catid (Christopher Taylor)**, [dgx_station_benchmarks](https://github.com/catid/dgx_station_benchmarks): the
  random-id benchmark shape.
