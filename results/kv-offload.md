# Where the KV cache lives: host-memory offload over PCIe and over NVLink-C2C

A DGX Station GB300 has two GPUs that reach the same Grace LPDDR5X host memory over two different links: the GB300 over
NVLink-C2C, and the RTX PRO 6000 Blackwell Max-Q in its PCIe 5 x16 slot. That let us change only the link between a GPU and
an offloaded KV cache, with the same model, engine and host memory. This page reports three tests run on 2026-10-19:

1. **Qwen3.8-27B on one Station, RTX PRO 6000 (PCIe) vs GB300 (NVLink-C2C).** A model whose KV cache is large (bf16,
   `64 KiB` per token). Copy ceilings, one decode step's attention with the cache read from host memory, and prefix reuse
   from a host tier.
2. **DeepSeek-V4.1-Flash-1017 on one GB300, host tier off vs on, first run.** A model whose KV cache is about `2 KB` per
   token. Capacity, decode, prefill, multi-turn reuse and answers. The multi-turn part was not measured (see below).
3. **The same DeepSeek setup, second run,** with deterministic eviction and an answer baseline.

Every number below is a row in [`results.jsonl`](../results.jsonl) with a scrubbed log excerpt under
[`logs/kv-offload/`](../logs/kv-offload/). Each test had its predictions and pass bars written down before anything was
booted; the runner checked them mechanically, and every verdict is reported, including the ones that failed. These are
report-only runs: no serving change follows from them.

## Common setup

- **Hardware.** One DGX Station GB300 per test (the "left" Station for test 1, the "right" one for tests 2 and 3). NVIDIA
  lists NVLink-C2C at 900 GB/s (both directions together) and the Grace LPDDR5X at 396 GB/s
  ([DGX Station page](https://www.nvidia.com/en-us/products/workstations/dgx-station/), vendor figures, read 2026-10-19).
  We did not measure the link or the memory in isolation.
- **Engine.** vLLM nightly `af7f9488` (`vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b`, pinned by
  digest in the rows) in all three tests. The host tier is vLLM's native `OffloadingConnector`
  (`--kv-offloading-size <GiB> --kv-offloading-backend native`): one pinned host region in `/dev/shm`, filled
  asynchronously with completed prompt blocks and read back when a request misses on the GPU but hits on the host. Host
  allocations are bound to Grace memory with `numactl --membind=0` in every container.
- **Eviction on demand.** With `VLLM_SERVER_DEV_MODE=1` and the server bound to `127.0.0.1`, `POST /reset_prefix_cache`
  empties the GPU prefix cache, and `?reset_external=true` empties the host tier as well. Tests 1 and 3 use this to put a
  prefix in exactly one place before a request: nowhere (recompute), in host memory only (reload), or on the GPU (hit).
- **Text.** Public [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) 3.0.0 test articles (the same
  haystack file as [`results/long-context-1m.md`](long-context-1m.md), sha256 `11c1f306...`), cut to token targets, with
  short synthetic facts inserted at fixed depths so that every answer can be checked. Temperature 0, one request at a time
  unless a table says otherwise. No prompt text is written to the outputs.
- **Medians.** Test 1 medians are ordinary medians. Tests 2 and 3 report nearest-rank percentiles, as their runner computes
  them: for an even number of requests the "median" is the lower of the two middle values.

## 1. One box, two links: Qwen3.8-27B NVFP4

**Serving.** [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) (`config.json` sha256
`71fa01bb...`). It has 64 layers, of which 16 use full attention with 4 KV heads of 256; the other 48 are linear-attention
layers with a fixed-size state. In bf16 that is `16 x 4 x 256 x 2 x 2 = 65,536` bytes of KV per token. Both arms ran the same
image (no overlay), the same files and the same flags: `--max-model-len 262144`, `--max-num-seqs 8`,
`--max-num-batched-tokens 8192`, `--kv-cache-memory-bytes 42949672960` (a `40 GiB` GPU KV pool on both GPUs, which came to
`641,906` tokens on each), `--kv-cache-dtype auto` (bf16), prefix caching on, `--kv-offloading-size 96` (a `96 GiB` pinned
host tier), no speculative decoding. Only the GPU changes, so only the link changes. The arms ran one after the other,
never together. Run on 2026-10-19 (left Station): 04:56-05:19 PT.

**The first window measured nothing.** The first attempt (04:50-04:53 PT) failed before any measurement: `numactl
--membind=0` needs the `CAP_SYS_NICE` capability, which our containers did not have, so both microbenchmarks and all four
boots exited with `set_mempolicy: Operation not permitted`, and all eight predictions were `NOT MEASURED`
([receipt](../logs/kv-offload/qwen38-pcie-vs-c2c.log)). We added `CAP_IPC_LOCK` and `CAP_SYS_NICE` to the containers,
changed nothing else, and ran it again. The second window is the one reported here.

**Predictions (written before either GPU was touched).** All eight `HELD`:

- P0: both arms boot with the connector and a `96 GiB` pinned host tier, GPU KV pools `within 3%` of each other.
- P1: pinned host-to-GPU copy `RTX in [45, 64] GB/s`, `GB300 >= 250 GB/s`, `GB300 / RTX >= 5`.
- P2: the engine's own load rate (reload rows at `128K` and `240K`, pooled) `GB300 >= 4 x RTX`, each `<= 1.10 x` its copy ceiling.
- P3: reload median `<= 0.25 x` recompute at `128K` and `240K` on each GPU.
- P4: at `240K`, `(reload - GPU hit)` on the RTX `>= 3 x` the same gap on the GB300.
- P5: at `240K`, the host-resident attention step `RTX >= 4 x GB300`, and `GB300 host >= 5 x GB300 device`; host and device outputs equal.
- P6: reload answers byte-identical to recompute at `>= 2 of 3` lengths, GPU hits identical to reloads on `>= 10 of 12`.
- P7: in the six-session test, round-2 median `<= 0.30 x` round 1 on each GPU, with host hits in round 2.

### Copy ceilings

[`tools/kvp_microbench.py`](../tools/kvp_microbench.py), run in a throwaway container of the same image on one GPU,
`numactl --membind=0`, pinned host memory, `4 GiB` buffers, median of 10.

<!-- results -->
| GPU (link to Grace memory) | Host to GPU, GB/s | GPU to host, GB/s | Both directions at once, GB/s | Kernel reading host memory in place, GB/s | Kernel reading its own memory, GB/s | Receipt |
|---|---|---|---|---|---|---|
| RTX PRO 6000 (`PCIe 5 x16`) | 55.09 | 57.21 | 101.91 | 52.92 | 1,622.33 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
| GB300 (NVLink-C2C) | 349.2 | 377.95 | 333.14 | 346.41 | 5,930.66 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
| GB300 / RTX, host to GPU | 6.34x | | | | | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
<!-- /results -->

- Both host-to-GPU figures agree with our earlier membench on the same box, 55.1 GB/s
  ([receipt](../logs/atlas-microbench/rtx-membench.log)) and 353.6 GB/s ([receipt](../logs/atlas-microbench/gb300-membench.log)).
- On PCIe, both directions at once nearly double the total (separate lanes each way). On NVLink-C2C the two-direction total
  is below either direction alone. That fits a shared ceiling behind the link, such as the 396 GB/s LPDDR5X figure, but we
  did not test that directly.

### The crutch: one decode step's attention with the cache read in place

The same microbenchmark times one decode step's attention (`q.K^T`, softmax, `.V`) over Qwen3.8-27B's real full-attention
KV layout (16 layers, 4 heads of 256, bf16) for one query token, with the cache read in place from pinned host memory or from
the GPU's own memory, median of 5. This is the cost a setup pays when the whole live cache stays in host memory. It leaves out
the weights and the linear-attention layers, so the tok/s column is a ceiling for this part of a decode step, not a decode
benchmark. Lengths: `32K` = `32,768`, `128K` = `131,072`, `240K` = `245,760` tokens.

<!-- results -->
| GPU | KV read from | `32K`, ms | `128K`, ms | `240K`, ms | Receipt |
|---|---|---|---|---|---|
| RTX PRO 6000 | host memory over PCIe | 67.122 | 163.307 | 305.822 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
| RTX PRO 6000 | its own memory | 4.745 | 11.914 | 22.098 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
| GB300 | host memory over NVLink-C2C | 6.552 | 24.944 | 44.97 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
| GB300 | its own HBM | 3.182 | 4.033 | 7.155 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
| RTX PRO 6000 | host read rate inside the kernel, GB/s | 31.99 | 52.6 | 52.67 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
| GB300 | host read rate inside the kernel, GB/s | 327.74 | 344.36 | 358.16 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
| RTX PRO 6000 | ceiling from host reads alone, tok/s | 14.9 | 6.1 | 3.3 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
| GB300 | ceiling from host reads alone, tok/s | 152.6 | 40.1 | 22.2 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
<!-- /results -->

- At `240K` the host-resident step takes 6.80x longer on the RTX than on the GB300, and on the GB300 the host-resident step
  is still 6.29x the device-resident one. Over NVLink-C2C, host memory is a slower second tier, not a second HBM.
- The GB300's own-memory step read `16.11 GB` in `7.155 ms`, about `2.25 TB/s`: this simple kernel does not reach the HBM's
  full bandwidth, so the device rows are a conservative reference.
- Host and device reads gave byte-identical outputs on both GPUs at every length.

### Reuse: recompute, reload from host, or hit on the GPU

[`tools/kvp_probe.py`](../tools/kvp_probe.py) `ladder`: one document per length with 4 inserted facts. Recompute = the first
request after emptying GPU and host caches (cold1). Reload = the prefix in host memory only (GPU cache emptied before each
of 4 requests); GPU hit = the prefix still on the GPU (4 requests). Reload and GPU hit are medians of 4. TTFT is the first
streamed content token. "Connector load" is vLLM's own counter (`kv_offload_load_bytes / kv_offload_load_time`) over the
reload rows.

<!-- results -->
| GPU | Prompt | Recompute, s | Reload from host, median, s | GPU hit, median, s | Recompute / reload | Connector load, GB/s | Bytes stored per prompt token | Receipt |
|---|---|---|---|---|---|---|---|---|
| RTX PRO 6000 | `32K` | 4.1478 | 0.2731 | 0.2208 | 15.19x | 54.63 | 69,310 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
| RTX PRO 6000 | `128K` | 27.7843 | 0.6451 | 0.4729 | 43.07x | 54.74 | 66,480 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
| RTX PRO 6000 | `240K` | 80.8219 | 1.3567 | 1.0495 | 59.57x | 54.74 | 66,005 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
| GB300 | `32K` | 1.1744 | 0.274 | 0.2579 | 4.29x | 330.34 | 69,310 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
| GB300 | `128K` | 6.1571 | 0.5238 | 0.4878 | 11.75x | 352.09 | 66,480 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
| GB300 | `240K` | 15.044 | 0.8481 | 0.7918 | 17.74x | 357.59 | 66,005 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
<!-- /results -->

- **Reloading beats recomputing on both links at every length.** The larger multiple on the RTX reflects its slower
  recompute (80.8 s against 15.0 s at `240K`), not a better reload.
- **The link shows up as the gap between a reload and a GPU hit.** At `240K` that gap is `0.3072 s` on the RTX and `0.0563 s`
  on the GB300, 5.46x larger on PCIe. The connector's own timer put the copy inside one `240K` reload at `0.300256 s` on the
  RTX and `0.045944 s` on the GB300 (`16,441,475,072` bytes each), close to the measured gaps.
- **The engine reached the copy ceiling.** Pooled over the `128K` and `240K` reloads, the connector loaded at 54.74 GB/s on
  the RTX and 354.84 GB/s on the GB300, against 55.09 and 349.2 GB/s for pinned copies.
- **Stored bytes per token** exceed the `65,536` of the bf16 attention cache by a roughly constant amount per prompt. That
  fits a fixed-size per-sequence state (this model's linear-attention layers), but we did not trace which tensors make up
  the difference.

### Six sessions in turn, and decode with the cache on the GPU

`rr`: six distinct `128K`-token documents (about `48 GiB` of KV, more than the `40 GiB` GPU pool), one question each,
round-robin, two rounds. Round 1 recomputes everything; by round 2 each session has been pushed off the GPU by the other five.
`decode`: one document primed once, then `512` tokens with the KV on the GPU (no host tier involved), no speculative decoding.

<!-- results -->
| GPU | First round, median TTFT, s | Second round, median TTFT, s | First / second round | Second round, tokens loaded from host | Second round, connector load, GB/s | Decode, `128K` prompt, KV on GPU, tok/s | Receipt |
|---|---|---|---|---|---|---|---|
| RTX PRO 6000 | 28.5956 | 0.7432 | 38.48x | 791,056 | 54.73 | 47.22 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
| GB300 | 5.6392 | 0.5269 | 10.70x | 791,056 | 364.56 | 130.33 | [log](../logs/kv-offload/qwen38-pcie-vs-c2c.log) |
<!-- /results -->

**Answers.** On each GPU, the answer after a reload matched the recompute byte for byte at all three lengths, GPU hits
matched reloads in 12 of 12 pairs, and two recomputes matched each other at all three lengths. Every reload answered all
four inserted facts. Answers are compared within one GPU only; the two GPUs run different kernels.

## 2. A compact cache: DeepSeek-V4.1-Flash-1017, first run

**Serving.** [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(revision `1bae0e10`) on the same image with the vllm#58132 replay overlay and original-el8's expert hook, as in
[`results/long-context-1m.md`](long-context-1m.md): hot experts in GB300 HBM (MegaMoE), cold experts and the engram tables
read from pinned Grace memory (`MEGA_PEER=0`, `MEGA_COLD_TRT=1`), RTX PRO 6000 unused. `--max-model-len 1048576`,
`--max-num-seqs 32`, fp8 KV (`fp8_ds_mla`), prefix caching on, DSpark speculative decoding `5/3/3`, thinking off. The KV pool
holds `3,459,043` tokens in `6.68 GiB`, about `2,074` bytes per token of context for the whole model, and the weights take
`218.88 GiB` of GPU memory ([receipt](../logs/kv-offload/dsv41-host-tier-run1.log)). Arm OFF has no host tier; arm ON adds
`--kv-offloading-size 18`. Run on 2026-10-19 (right Station): OFF rows 05:01-05:17 PT, ON rows 05:28-05:43 PT. Probe:
[`tools/kvo_probe.py`](../tools/kvo_probe.py).

**Expert map: not reproducible from this repo.** As in the long-context runs, the hot/cold expert split uses a map fitted to
a private workload, which is not published. It affects speed, not which experts are computed. It is the same in both arms.

**Host memory is not idle here.** Grace memory already holds the cold experts and the engram tables. The lowest free memory
seen while serving was `34 GiB` without the tier and `14 GiB` with it. The tier size was set by a pre-registered rule from
the free memory of the OFF arm.

**Predictions:** 4 of 8 `HELD`.

| Prediction | Verdict |
|---|---|
| P0: ON GPU KV pool `>= 0.97 x OFF` | HELD |
| P1: sessions held at once, ON equal to OFF `(+-1)` at `250K`, `500K`, `1M` | HELD |
| P2: the over-subscribed `250K` batch finishes in `<= 0.90 x` the OFF wall time with the tier | NOT HELD |
| P3: first-toucher median TTFT over turns `2-5`, ON `<= 0.25 x OFF` | NOT MEASURED |
| P4: all-session p90 TTFT over turns `2-5`, ON `<= OFF` | NOT MEASURED |
| P5: decode, ON `>= 0.95 x OFF` (C1 median and C4 aggregate) | HELD |
| P6: C1 cold prefill, ON `>= 0.90 x OFF` | HELD |
| P7: answers identical on `>= 19/20` (ON vs OFF; ON reload vs ON first pass), correct ON `>= OFF - 1` | NOT HELD |

### Capacity: many long documents at once

`cap`: N distinct documents sent at once (`16` of `250K`, `8` of `500K`, `4` of `1M` tokens), `max_tokens 16`, with
`/metrics` sampled every second.

<!-- results -->
| Prompt | Arm | Most requests running at once | Wall, s | Aggregate prefill, tok/s | Receipt |
|---|---|---|---|---|---|
| `250K` | OFF | 3 | 160.56 | 24,771.3 | [log](../logs/kv-offload/dsv41-host-tier-run1.log) |
| `250K` | ON | 3 | 154.04 | 25,819.7 | [log](../logs/kv-offload/dsv41-host-tier-run1.log) |
| `500K` | OFF | 3 | 173.37 | 22,942.6 | [log](../logs/kv-offload/dsv41-host-tier-run1.log) |
| `500K` | ON | 3 | 167.3 | 23,775.0 | [log](../logs/kv-offload/dsv41-host-tier-run1.log) |
| `1M` | OFF | 3 | 209.14 | 19,020.3 | [log](../logs/kv-offload/dsv41-host-tier-run1.log) |
| `1M` | ON | 3 | 202.57 | 19,637.2 | [log](../logs/kv-offload/dsv41-host-tier-run1.log) |
<!-- /results -->

- The tier added no capacity, as expected for a prefix tier, and there were no preemptions in either arm. Three running
  requests is well below what the GPU pool could hold at `250K`, so something other than KV space capped concurrency; we did
  not investigate it.
- The `250K` batch took 4.06% less wall time with the tier, short of the pre-registered `10%`-faster bar (P2 not held). With no
  preemptions in arm OFF, nothing had been evicted, so there was nothing to reload.

### Decode and prefill at `250K`

Cold prefill: `4` fresh `250K` documents, one at a time (median). Decode: `512` forced tokens after a `250K` prompt, C1
(median of `2`) and four streams at once (C4).

<!-- results -->
| Arm | Cold prefill, median, tok/s | Decode C1, tok/s | Decode C4 per stream, tok/s | Decode C4 aggregate, tok/s | Receipt |
|---|---|---|---|---|---|
| OFF | 24,925.8 | 155.1 | 73.4 | 250.9 | [log](../logs/kv-offload/dsv41-host-tier-run1.log) |
| ON | 25,952.9 | 156.2 | 70.9 | 244.7 | [log](../logs/kv-offload/dsv41-host-tier-run1.log) |
<!-- /results -->

### Multi-turn: not measured, and why

Eight sessions of five turns shared a `250K`-token system prefix, adding about `8K` tokens per turn. To push the prefix out
of the GPU cache before turns `2`-`5`, the runner sent a batch of distinct documents sized to `1.15` times the GPU pool (the
cap and press phases). Before turn `2` that batch sent `3,977,272` prompt tokens through the GPU cache, 1.15x the pool, and
eviction worked. Before turns `3`-`5` it did not: the GPU still held part of the shared prefix.

<!-- results -->
| Turn | First toucher TTFT, OFF, s | First toucher TTFT, ON, s | Prefix tokens still on the GPU after the flush (OFF) | Receipt |
|---|---|---|---|---|
| `2` | 10.991 | 1.152 | none | [log](../logs/kv-offload/dsv41-host-tier-run1.log) |
| `3` | 8.22 | not valid | 88,448 | [log](../logs/kv-offload/dsv41-host-tier-run1.log) |
| `4` | 8.762 | not valid | 81,536 | [log](../logs/kv-offload/dsv41-host-tier-run1.log) |
| `5` | 8.879 | not valid | 85,760 | [log](../logs/kv-offload/dsv41-host-tier-run1.log) |
<!-- /results -->

Only one of four turns passed the pre-registered eviction check, so P3 and P4 are `NOT MEASURED`. A GPU cache that holds
`3.46M` tokens of this model is hard to flush on purpose.

### Answers: 18 of 20, and a baseline that also gave 18 of 20

`corr`: 2 documents of `250K` tokens with 10 questions each (4 single needles, 2 first hops, 2 second hops, 2 two-hop
chains), thinking off, `max_tokens 64`. Pass 2 re-sends pass 1's prompts byte for byte after the GPU flush.

- Correct: OFF 16/20 then 18/20; ON 18/20 and 18/20.
- Identical answers: ON pass 1 vs OFF pass 1 on 18/20 (the bar was 19); ON pass 2 vs ON pass 1 on 20/20; and OFF pass 2 vs
  OFF pass 1, two passes without the tier, also on only 18/20
  ([receipt](../logs/kv-offload/dsv41-host-tier-run1.log)).
- The engine varied that much on its own at temperature 0, and the pre-registration had no rule for separating that from
  an effect of the tier, so P7 is `NOT HELD` as written. Only 2 of the 20 pass-2 items actually came back from host memory
  (one per document); the other 18 were GPU hits. Test 3 fixes both problems.

## 3. DeepSeek-V4.1-Flash-1017, second run: deterministic eviction

**Changes from the first run** (designed after its results; the predictions were written before this run booted):
both arms bind the server to `127.0.0.1` with `VLLM_SERVER_DEV_MODE=1`, and the GPU cache is emptied on request
(`/reset_prefix_cache`) instead of by flooding it; the answer test gained recompute baselines with and without the
connector; and a single-session reload ladder at `500K` and `1M` was added. The host tier is `16 GiB` (same rule; lowest free
memory while serving `32 GiB` OFF, `19 GiB` ON). Everything else as in the first run. Run on 2026-10-19 (right Station):
OFF rows 07:49-08:01 PT, ON rows 08:12-08:17 PT. Probe: [`tools/kvr_probe.py`](../tools/kvr_probe.py), which imports
[`tools/kvo_probe.py`](../tools/kvo_probe.py) and [`tools/lc_probe_2hop_ds.py`](../tools/lc_probe_2hop_ds.py).

**Predictions:** all eight `HELD`.

| Prediction | Verdict |
|---|---|
| P0: ON boots with the connector; ON GPU KV pool `>= 0.97 x OFF` | HELD |
| P1: eviction is deterministic: turns `2-5` pass the check in both arms, and every ladder reload row passes | HELD |
| P2: first-toucher median TTFT over turns `2-5`, ON `<= 0.25 x OFF` | HELD |
| P3: all-session median and p90 TTFT over turns `2-5`, ON `<= OFF` | HELD |
| P4: ladder reload median, ON `<= 0.25 x` the OFF recompute and `<= 0.25 x` the ON cold request, at `500K` and `1M` | HELD |
| P5: pooled connector load rate within `[64, 384.1] GB/s` | HELD |
| P6: the host reload changes no more answers than the engine's own run-to-run baseline | HELD |
| P7: ladder reload answers correct ON `>= OFF - 1`; GPU hits identical to reloads on all but one pair | HELD |

### Multi-turn with the GPU cache emptied before turns `2`-`5`

Session 1 of each turn is the first toucher: the GPU cache is emptied just before it, so it finds the shared prefix in host
memory (ON) or recomputes it (OFF). Sessions 2-8 follow without a reset; "followers" is their median.

<!-- results -->
| Turn | OFF first toucher, s | OFF followers, median, s | ON first toucher, s | ON followers, median, s | Receipt |
|---|---|---|---|---|---|
| `1` | 10.381 | 1.069 | 10.257 | 1.07 | [log](../logs/kv-offload/dsv41-host-tier-run2.log) |
| `2` | 10.627 | 1.412 | 1.126 | 1.11 | [log](../logs/kv-offload/dsv41-host-tier-run2.log) |
| `3` | 10.97 | 1.713 | 1.14 | 1.11 | [log](../logs/kv-offload/dsv41-host-tier-run2.log) |
| `4` | 11.234 | 2.096 | 1.024 | 1.14 | [log](../logs/kv-offload/dsv41-host-tier-run2.log) |
| `5` | 11.667 | 2.474 | 1.219 | 1.176 | [log](../logs/kv-offload/dsv41-host-tier-run2.log) |
| Turns `2`-`5` pooled, first toucher median | 10.97 | | 1.126 | | [log](../logs/kv-offload/dsv41-host-tier-run2.log) |
| Turns `2`-`5` pooled, all sessions median / p90 | 1.966 / 10.627 | | 1.131 / 1.18 | | [log](../logs/kv-offload/dsv41-host-tier-run2.log) |
<!-- /results -->

- Eviction passed the check on all four turns in both arms. The first toucher was 9.74x faster with the tier.
- Without the tier, the followers' median climbed every turn: the first toucher put the shared prefix back on the GPU, but
  each session's own history still had to be recomputed. With the tier those histories came back from host memory too.

### One long document: recompute, reload, hit

`ladder`: one document per length with 4 single needles. Cold1 = after emptying GPU and host caches. The reload position =
after emptying the GPU cache only (ON: from the host tier; OFF: recomputed again). GPU hit = prefix still on the GPU.
Cold2 = a second full recompute. Reload position and GPU hit are the nearest-rank median of two requests, i.e. the lower one (both are in the receipt).

<!-- results -->
| Prompt | Arm | Cold1, s | Reload position, lower of two, s | GPU hit, lower of two, s | Cold2, s | Receipt |
|---|---|---|---|---|---|---|
| `500K` | OFF | 20.597 | 20.569 | 1.373 | 20.528 | [log](../logs/kv-offload/dsv41-host-tier-run2.log) |
| `500K` | ON | 20.387 | 1.391 | 1.373 | 20.403 | [log](../logs/kv-offload/dsv41-host-tier-run2.log) |
| `1M` | OFF | 49.969 | 49.974 | 2.715 | 49.966 | [log](../logs/kv-offload/dsv41-host-tier-run2.log) |
| `1M` | ON | 49.666 | 2.775 | 2.756 | 49.713 | [log](../logs/kv-offload/dsv41-host-tier-run2.log) |
<!-- /results -->

What the connector moved for one reload:

<!-- results -->
| Prompt | Tokens loaded from host | Bytes loaded | Connector time (reload1), s | Connector load (both reloads), GB/s | Bytes per loaded token | Receipt |
|---|---|---|---|---|---|---|
| `500K` | 496,512 | 822,223,872 | 0.0036 | 220.3 | 1,656 | [log](../logs/kv-offload/dsv41-host-tier-run2.log) |
| `1M` | 993,920 | 1,645,931,520 | 0.0076 | 216.1 | 1,656 | [log](../logs/kv-offload/dsv41-host-tier-run2.log) |
| All four reloads pooled | | 4.597 GiB | 0.0227 | 217.5 | | [log](../logs/kv-offload/dsv41-host-tier-run2.log) |
<!-- /results -->

- **A million tokens came back in `2.775 s` instead of about `50 s`** (18.01x faster than the OFF recompute at the same
  position).
- **The copy is a small part of it.** The 1M reload was `0.019 s` slower than a GPU hit (`0.018 s` at `500K`), for a copy
  of about `0.0076 s`. A GPU hit, which copies nothing, takes `0.9932` of the reload's TTFT. Those gaps are medians of two
  requests, so we would not read much into their difference. What the rest of the `2.8 s` is made of we did not profile;
  it grows with context (the GPU hit at `1M` takes 2.01x the `500K` one).
- **Bytes per loaded token (`1,656`) are below the GPU pool's `2,074`.** That fits the engine's design as we read it in the
  source (only the prefix-reusable part of the cache is offloaded; a sliding-window cache is rebuilt after a hit). We did
  not measure the split.

### Answers, with baselines

Same 20 questions as the first run. Passes 1 and 3: caches emptied once, so each document's first question is a recompute
and the rest are GPU hits. Pass 2: GPU cache emptied before every question (ON: every question's prefix from host memory;
OFF: every question recomputed).

- Correct: OFF 17/20, 18/20, 18/20; ON 18/20 in all three passes. In ON pass 2, 20/20 items came back from host memory.
- Baselines (connector absent / present): OFF pass 3 vs pass 1 identical on 19/20, ON pass 3 vs pass 1 on 20/20.
- Host reload vs the same arm's normal pass (ON p2 vs ON p1): 20/20. Host reload vs recompute, item by item (ON p2 vs OFF
  p2): 20/20. Connector present vs absent on the same path (ON p1 vs OFF p1): 19/20, within the baseline
  ([receipt](../logs/kv-offload/dsv41-host-tier-run2.log)).
- On this evidence, reloading from host memory changed no answers beyond the engine's own run-to-run variation, and the
  18/20 of the first run was that variation.

## Limits

- One model and one box per test, one run per cell; medians over 4 (test 1) or 2 (test 3) requests where noted.
- One engine build (vLLM `af7f9488`, a nightly), plus the DeepSeek overlay and hook in tests 2 and 3.
- The RTX PRO 6000 reaches Grace LPDDR5X over PCIe 5 x16. A desktop GPU with DDR5 behind a desktop CPU differs in the card,
  the host memory and the software; what carries over is the link class.
- The crutch numbers are a microbenchmark of attention reads alone, not a decode benchmark, and not a test of any KV-streaming
  engine.
- Qwen3.8-27B's KV is bf16 here. Transfer times scale with the cache format.
- The DeepSeek runs use the GB300-only placement with a private expert map, not the Station's daily two-GPU configuration.
- Eviction was forced (a flood in test 2, the reset endpoint in tests 1 and 3), not caused by other users' traffic.
- At this vLLM version, host memory cannot hold the cache of running DeepSeek-V4 requests: the experimental HiSparse path
  refuses DeepSeek V4 (read in the source, not run). We did not test that case for Qwen either.
- The second DeepSeek run was designed after the first run's results were known.

## Reproduce

- [`tools/kvp_microbench.py`](../tools/kvp_microbench.py): copy ceilings, zero-copy reads and the attention-step crutch; one GPU,
  no model, run inside the vLLM image with `numactl --membind=0`.
- [`tools/kvp_probe.py`](../tools/kvp_probe.py): test 1 phases (`calib`, `ladder`, `rr`, `decode`) against one booted arm.
- [`tools/kvo_probe.py`](../tools/kvo_probe.py): test 2 phases (`corr`, `turn`, `cap`, `press`, `decode`).
- [`tools/kvr_probe.py`](../tools/kvr_probe.py): test 3 phases (`turn`, `corr`, `ladder`, `reset`); imports `kvo_probe.py` and
  [`tools/lc_probe_2hop_ds.py`](../tools/lc_probe_2hop_ds.py) (corpus, needle plan, streaming client).
- Server flags, environment and capabilities for every arm are in the log excerpts. The container runner scripts are not
  published: they stop and restore our own services around each window.

## Licences and credits

- **Models.** Qwen3.8-27B (Qwen) and NVIDIA's NVFP4 checkpoint `nvidia/Qwen3.8-27B-NVFP4` are Apache-2.0 (model cards,
  checked 2026-10-21). DeepSeek-V4.1-Flash-1017 (DeepSeek) is MIT (model card, checked 2026-10-21). No model weights or
  anything derived from them are in this repository.
- **Software.** vLLM (Apache-2.0) is used as the image named in the rows, including its native KV offloading connector
  ([vLLM blog post](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)). The DeepSeek expert hook is original-el8's,
  mounted at run time as in [`recipes/dsv41-flash-sidecar/`](../recipes/dsv41-flash-sidecar/). The microbenchmark and the
  probes are ours, under Apache-2.0.
- **Prompted by** a community llama.cpp fork that streams the KV cache from host memory over PCIe
  ([adaptive KV streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming)). We have not run it or
  re-measured any of its numbers.
