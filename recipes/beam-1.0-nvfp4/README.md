# Reflection Beam 1.0 (501B-A23B) NVFP4 on DGX Station: SGLang on one Station and on two

Reflection's Beam 1.0 is a 501B-parameter mixture-of-experts model with about 23B parameters active per token (52 layers,
256 routed experts per MoE layer with 8 active, a mix of full-attention and sliding-window layers). The NVFP4 checkpoint
(FP4 routed experts, every other weight BF16) is `273.49 GiB`; one GB300 has `250.7 GiB`. This folder measures it on
DGX Station GB300 with stock SGLang, the image the SGLang team published for Beam (`lmsysorg/sglang:beam`), in three arms
measured on 2026-10-16:

- **a2, two Stations (recommended).** TP2 across both GB300s over one 400G RoCE link, every weight in HBM, and
  Reflection's DFlash2 speculative-decoding drafter. One user decodes at 237.8 tok/s (median of three) on real text.
- **a1, one Station, no drafter.** TP1 on one GB300 with `60 GiB` of weights in Grace memory through SGLang's
  `--cpu-offload-gb`.
- **b1, one Station, with the drafter.** a1 plus DFlash2.

**One Station is not viable with this engine.** SGLang's CPU offload (OffloaderV1 in this build) copies every offloaded
weight to the GPU on every forward step, not only the experts a token uses, so one user gets 5.7 tok/s without the
drafter and 14.02 tok/s with it. Both miss the bar we wrote down before the run (best one-Station arm at least `20 tok/s`
for one user and at least `100 tok/s` aggregate at 16 users, decode metric, real text). vLLM's UVA offload, which lets
the GB300 read only the experts a token needs from Grace memory in place and is how Kimi K3 and Nemotron 3 Ultra run
from Grace ([kimi-k3](../kimi-k3/), [nemotron-ultra](../nemotron-ultra/)), has no counterpart in this SGLang build.

All numbers here are on stock SGLang (no patches, no hooks), greedy, at temperature 0.

## Models and licences

| Checkpoint | Revision | Size | Licence |
|---|---|---|---|
| [`reflection/Beam-1.0-501B-A23B-NVFP4`](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4) | `main` as listed on 2026-10-16 (no commit hash recorded; the size and LFS sha256 of all `63` files are in [manifest/nvfp4-hf.tsv](manifest/nvfp4-hf.tsv)) | `273.49 GiB` | Apache-2.0 (model card, checked 2026-10-17) |
| [`reflection/Beam-1.0-501B-A23B-DFlash2`](https://huggingface.co/reflection/Beam-1.0-501B-A23B-DFlash2) (the speculative-decoding drafter, arms a2 and b1) | `main` as listed on 2026-10-16 ([manifest/dflash2-hf.tsv](manifest/dflash2-hf.tsv)) | `3.59 GiB` | Apache-2.0 (model card metadata, checked 2026-10-17; the repository listing has no `LICENSE` file) |

**No weights are included in this repository**, and nothing here is derived from them. Read both licences yourself
before serving the model to anyone else.

## Results (2026-10-16)

**Shape.** Real text: public CNN/DailyMail 3.0.0 test articles, the same long pool as the Kimi K3 recipe
([`build_pool_k3.py`](../kimi-k3/bench/build_pool_k3.py)), about `8,230` to `8,370` Beam prompt tokens, `512` forced
output tokens, temperature 0, streaming, through the server's chat template with `reasoning_effort: minimal` (the Beam
template always reasons; minimal is its smallest budget, and forced tokens count reasoning and answer alike). C1 is three
repetitions of one prompt each (slices `eval-c1-r1` to `r3`); C4 and C16 are one run each on the first `4` and `16`
prompts of the `eval-c16` slice. Driven by [`bench/kbench_chat_beam.py`](bench/kbench_chat_beam.py).

**Two metrics, side by side.** *Decode* is the repository's decode metric, C x 1000 / mean TPOT, with TPOT measured per
request from its first to its last streamed token. It leaves out time to first token and queueing, so **decode is not the
throughput a user sees**. *Output* is completion tokens divided by the wall-clock time of the whole run, including time to
first token; it is closer to what a user sees. At 16 users, where every request first waits for 16 prefills of `8K`
tokens, the two differ by more than a factor of two.

<!-- results -->
| Arm, real text | C1 decode, median of three, tok/s | C1 output, same repetition, tok/s | C4 decode / output, tok/s | C16 decode / output, tok/s | Receipt |
|---|---|---|---|---|---|
| **a2: two Stations, TP2, all in HBM, DFlash2** | **237.81** | **201.94** | 587.57 / 399.76 | 1306.84 / 577.85 | [log](../../logs/beam-1.0/e-a2-two-stations.log) |
| a1: one Station, `60 GiB` offloaded to Grace, no drafter | 5.7 | 5.63 | 22.26 / 21.7 | 84.97 / 78.81 | [log](../../logs/beam-1.0/e-one-station.log) |
| b1: a1 plus DFlash2 | 14.02 | 13.67 | 57.89 / 51.23 | 211.01 / 105.54 | [log](../../logs/beam-1.0/e-one-station.log) |
<!-- /results -->

<!-- results -->
| C1 repetitions, real text | Decode, tok/s (r1 / r2 / r3) | Output, tok/s (r1 / r2 / r3) | Receipt |
|---|---|---|---|
| a2 | 237.81 / 211.21 / 247.73 | 201.94 / 179.99 / 208.37 | [log](../../logs/beam-1.0/e-a2-two-stations.log) |
| a1 | 5.7 / 5.7 / 5.7 | 5.63 / 5.63 / 5.63 | [log](../../logs/beam-1.0/e-one-station.log) |
| b1 | 15.23 / 13.54 / 14.02 | 14.83 / 13.21 / 13.67 | [log](../../logs/beam-1.0/e-one-station.log) |
<!-- /results -->

<!-- results -->
| Detail | Value | Receipt |
|---|---|---|
| Random token ids (control, **not real-text speed**): C1 decode, median of three, tok/s: a2 / a1 / b1 | 92.42 / 5.66 / 5.54 | [a2](../../logs/beam-1.0/e-a2-two-stations.log), [a1, b1](../../logs/beam-1.0/e-one-station.log) |
| a2, time to first token, random ids, `8K` tokens: s / prefill tok/s (median of three) | 0.329 / 24869.1 | [log](../../logs/beam-1.0/e-a2-two-stations.log) |
| a2, time to first token, random ids, `64K` tokens: s / prefill tok/s (median of two) | 2.717 / 24119.4 | [log](../../logs/beam-1.0/e-a2-two-stations.log) |
| a2, C16 real text, median time to first token, s | 5.25 | [log](../../logs/beam-1.0/e-a2-two-stations.log) |
| Drafter on one Station: b1 over a1, C1 decode, x | 2.46 | [log](../../logs/beam-1.0/e-one-station.log) |
| Two Stations over the best one-Station arm: a2 over b1, C1 decode, x | 16.96 | [a2](../../logs/beam-1.0/e-a2-two-stations.log), [b1](../../logs/beam-1.0/e-one-station.log) |
| b1, GSM8K, first `200` test rows, `reasoning_effort: low`, % | 96.5 | [log](../../logs/beam-1.0/e-one-station.log) |
<!-- /results -->

Reading the numbers:

- **Two Stations.** 237.8 tok/s decode for one user, 201.9 tok/s output (a `512`-token answer in about `2.5` s, of which
  about `0.4` s is time to first token on an `8K`-token prompt). The three C1 repetitions spread from 211.2 to 247.7 tok/s
  because the drafter's acceptance depends on the prompt: observed tokens per verify step were `2.39` to `2.81`
  (`chunk_acc` in the log). Prefill holds at about `24K` tokens per second from `8K` to `64K` tokens.
- **Random ids are a control.** The drafter accepts nothing on random token ids (`chunk_acc` about `1.0` in every arm),
  so a2 runs at its no-acceptance rate there (92.42 tok/s) and b1 is no faster than a1. The real-text rows are the measure.
- **One Station.** The drafter multiplies one-user decode by 2.46x (5.7 to 14.02 tok/s), because each verify step pays
  the per-step weight copy once for several tokens. It is still well short of the bar. At 16 users b1's decode
  (211.01 tok/s) is twice its output (105.54 tok/s): queueing behind the prefills dominates the wall clock.
- **GSM8K** is a quality check, not a leaderboard figure: the first `200` rows at `reasoning_effort: low` (a `32`-token
  reasoning budget), so it is not comparable to the [94.69](https://docs.sglang.io/cookbook/autoregressive/Reflection/Beam-1_0)%
  that the SGLang cookbook reports on the full `1,319`-row set at its default effort.

### Quality pass on the two-Station configuration (2026-10-16)

A separate boot of a2 (same image, flags and launcher), later the same evening:

<!-- results -->
| a2 quality | Result | Receipt |
|---|---|---|
| GSM8K, first `200` test rows, `reasoning_effort: low`, temperature 0, % | 97.0 | [log](../../logs/beam-1.0/q5-a2-quality.log) |
| Tool calls, our in-house extended set, passed of `12` | 12 | [log](../../logs/beam-1.0/q5-a2-quality.log) |
| Tool calls, our in-house smoke set, passed of `3` | 3 | [log](../../logs/beam-1.0/q5-a2-quality.log) |
| Needle retrieval, our in-house test, about `64K` prompt tokens, correct of `6` | 6 | [log](../../logs/beam-1.0/q5-a2-quality.log) |
| Needle retrieval, our in-house test, about `128K` prompt tokens (`126,577` to `126,582`), correct of `6` | 6 | [log](../../logs/beam-1.0/q5-a2-quality.log) |
| `20`-minute soak at C16: requests completed / failed (timeouts, HTTP errors, other) | 1526 / 0 | [log](../../logs/beam-1.0/q5-a2-quality.log) |
<!-- /results -->

- **GSM8K** 194 of 200 here and 193 of 200 on b1 (same scorer, effort and rows; different boots and placements).
- **Tool calls and needles are our own tests**, not public benchmarks: OpenAI-format tools through the server's `beam`
  tool parser (tool selection, parallel calls, no-tool questions, an enum, a typed number, a date, a follow-up on a tool
  result); and synthetic six-digit codes hidden at fixed depths in public CNN/DailyMail text (four single needles and two
  two-hop chains per length). The test scripts are not in this release; the cases and answers are in the log.
- **The soak** mixes seven request classes from one seeded list (GSM8K at high and low effort, random ids, long synthetic
  text, short chat, a shared-prefix class, and GSM8K sampled at temperature 0.6); the short-chat and that GSM8K class
  sample at temperature 0.6, the rest are greedy. Non-streaming, `600` s per-request timeout. `1,526` of `1,526`
  completed and the server was healthy afterwards.

### Caveats

- **Drafter block size.** We ran DFlash2 with `--speculative-dflash-block-size 8`, the SGLang cookbook's low-latency
  setting; the drafter's own `config.json` says `16`. Block 16 was not measured.
- **a1 and b1 run eager** (`--disable-cuda-graph`), the only CPU-offload configuration SGLang's own Beam tests use; a2 runs
  with CUDA graphs. a1 and b1 use `--mem-fraction-static 0.92`, a2 `0.85` (the cookbook value).
- **Prefix-cache hygiene.** SGLang exposes no prefix-cache hit and query counters, so the repository's measured
  hit-rate check could not be applied. Instead every run prepends its own nonce to every prompt (the first KV block
  differs), uses a slice disjoint from the other runs and warms up on a separate slice; the server's `cache_hit_rate`
  gauge read `0.0` throughout. Random-id runs use a fresh seed per repetition and a separately seeded warm-up.
- **Serving shape.** Context `131072`, BF16 KV cache, `16` running requests, chunked prefill `8192`, not the cookbook's
  1M-token context, and TP1/TP2 instead of the cookbook's TP4 on one NVLink domain.
- **C4 and C16 are single runs**; C1 is the median of three. No power readings were recorded in this window.
- **Image pin.** We recorded the local image id (`docker inspect`), identical on both Stations, not the registry digest.
  The SGLang source inside the image matches the SGLang pull request that added Beam (#44747, head `f7473673`) for the
  Beam model, the offloader and the argument files.

## Hardware and software

| Item | Value |
|---|---|
| Topology | a1, b1: one GB300 (TP1), `60 GiB` of weights in pinned Grace memory. a2: one GB300 on each Station (TP2), one 400G DAC (RoCEv2) between the ConnectX-8s, no offload. The RTX PRO 6000s were not used |
| Image | `lmsysorg/sglang:beam`, local image id `sha256:0ea912aadc3d7827858de9aa5a312c4d51c254e7dbe670dbbe8181895b92071d` (SGLang `0.0.0.dev0`; NCCL `2.30.7+cuda13.3`) |
| Fabric (a2) | Data Direct overlay + `tuner3.csv` + `NCCL_IB_TC=106` ([fabric recipe](../fabric-data-direct/)); custom all-reduce and FlashInfer all-reduce fusion off (Beam's fused path assumes one node) |
| Server flags (all arms) | `--attention-backend trtllm_mha --page-size 64 --context-length 131072 --max-running-requests 16 --chunked-prefill-size 8192 --enable-unified-memory --tool-call-parser beam --reasoning-parser beam --enable-metrics --cuda-graph-max-bs-decode 16` |
| Drafter flags (a2, b1) | `--speculative-algorithm DFLASH --speculative-draft-model-path /drafter --speculative-dflash-block-size 8` |

`--enable-unified-memory` is a KV-pool layout option in SGLang (one buffer for the full and sliding-window KV), not
weight offload.

## How to run

Prerequisites: the weights at `/models/hf/reflection__Beam-1.0-501B-A23B-NVFP4` (and the drafter at
`/models/hf/reflection__Beam-1.0-501B-A23B-DFlash2`) on every Station you use, the image, and at least `150 GiB` of free
host memory. For a2 also the fabric recipe's `~/hostrdma` and `~/nccl-tuner` on both Stations, and `RANK0_IP` /
`RANK1_IP` set to the two rail addresses.

```bash
# a2: two Stations. Rank 1 first (headless), then rank 0 (API on :8050, model name "beam").
NNODES=2 TP=2 SPEC=1 OPTFLAGS='--enable-metrics --cuda-graph-max-bs-decode 16' launch/launch-beam.sh 1   # second Station
NNODES=2 TP=2 SPEC=1 OPTFLAGS='--enable-metrics --cuda-graph-max-bs-decode 16' launch/launch-beam.sh 0   # first Station
# a1 / b1: one Station (b1: add SPEC=1)
OFFGB=60 MEMF=0.92 NOGRAPH=1 OPTFLAGS='--enable-metrics --cuda-graph-max-bs-decode 16' launch/launch-beam.sh 0

# Real text: build the pool with recipes/kimi-k3/bench/build_pool_k3.py, then per run (fresh nonce each time):
python3 bench/kbench_chat_beam.py http://127.0.0.1:8050 beam pool/long/eval-c1-r1-try1.jsonl out.json 1 --head 1 \
  --max-tokens 512 --warm pool/long/eval-c1-r1-try1-warm.jsonl --nonce mine-c1-r1
python3 bench/kbench_chat_beam.py http://127.0.0.1:8050 beam pool/long/eval-c16-try1.jsonl out16.json 16 --head 16 \
  --max-tokens 512 --warm pool/long/eval-c16-try1-warm.jsonl --nonce mine-c16
# Random ids (control) and time to first token:
python3 bench/beam_rand.py catid http://127.0.0.1:8050 beam catid.json --isl 8192 --osl 1024 --reps 3 --seed 1000
python3 bench/beam_rand.py ttft http://127.0.0.1:8050 beam ttft8k.json --isl 8192 --reps 3 --seed 2000
python3 bench/beam_rand.py ttft http://127.0.0.1:8050 beam ttft64k.json --isl 65536 --reps 2 --seed 3000
# GSM8K-200 (put openai/grade-school-math test.jsonl next to the script as gsm8k_test.jsonl):
EFFORT=low python3 bench/gsm8k200_beam.py http://127.0.0.1:8050/v1/chat/completions beam gsm.jsonl 16 200
docker rm -f beam    # teardown; for a2 on rank 0 first, then rank 1
```

Launch to ready took `647` s for a1 (first launch of the day) and `189` s for the a2 quality-pass boot (weights already in the page cache).

## Files

| Path | What | Origin |
|---|---|---|
| `launch/launch-beam.sh` | one launcher for a1, b1 and a2 (`NNODES`, `TP`, `OFFGB`, `SPEC`, `NOGRAPH`), Data Direct block, CDI GPU pinning | ours |
| `bench/kbench_chat_beam.py` | real-text decode and output bench through the chat template (nonce, warm-up slice, `chunk_acc`) | ours |
| `bench/beam_rand.py` | random-id decode (catid) and time-to-first-token rows for SGLang | ours |
| `bench/gsm8k200_beam.py` | GSM8K first-200 accuracy | ours |
| `manifest/nvfp4-hf.tsv`, `manifest/dflash2-hf.tsv` | Hugging Face file listings (size, LFS sha256) as downloaded, 2026-10-16 | ours |

Not included: the in-house tool-call, needle and soak scripts (their cases and results are in the quality log).

## Credits

Reflection for Beam 1.0 and the DFlash2 drafter; the SGLang team for the Beam support and the `lmsysorg/sglang:beam`
image and cookbook recipe this launcher starts from.
