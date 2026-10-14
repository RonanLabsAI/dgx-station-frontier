# Kimi K3 on DGX Station: 3-bit in vLLM across two Stations, 1.6-bit in llama.cpp

Moonshot AI's Kimi K3 (93 layers, 896 routed experts per MoE layer with 16 active, Kimi Delta Attention plus MLA) is
about `1.56 TB` as released. Two Stations have about `1.32 TB` of GPU-reachable memory for weights once KV cache and a
host floor are set aside, so no unpruned 4-bit K3 fits. This folder has two ways to run it anyway:

- **A. vLLM, two Stations, 3-bit (recommended).** `vellum-ai/Kimi-K3-W3A16-g64` (GPTQ int3 routed experts, every other
  weight BF16) with TP2 + EP2 across one 400G link, vLLM's Humming 3-bit MoE kernels, and `408` GiB per rank of routed
  experts in Grace memory, read by the GB300 over NVLink-C2C (UVA offload). Needs a small load hook (included) and four
  fixes to load at all. **A2** adds Inferact's DSpark speculative-decoding drafter: one user decodes about 1.46 times
  faster on real text, and nothing changes at four or more concurrent users.
- **B. llama.cpp, 1.6-bit.** unsloth's `UD-IQ1_M` GGUF on one Station, on two Stations over RPC, and on all four GPUs.
  Includes our one-line C2C zero-copy patch, and a warning: llama.cpp built with CUDA 13.2.0/13.2.1 produces garbage
  for this model.

As far as we found (search of 2026-10-08), this is the first Kimi K3 run on DGX Station and the first
with routed experts served from Grace memory. The only other desktop-class K3 we know of is
[ciprianveg/gb10-vllm](https://github.com/ciprianveg/gb10-vllm/tree/main/kimi-k3/v5) on sixteen DGX Sparks.

## Models and licences

| Checkpoint | Revision | Size | Licence |
|---|---|---|---|
| [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) (the original; only `preprocessor_config.json` is used here) | `f831ab66` | `1,560.9 GB` | Kimi K3 License |
| [`vellum-ai/Kimi-K3-W3A16-g64`](https://huggingface.co/vellum-ai/Kimi-K3-W3A16-g64) (recipe A) | `09c06868` | `1,290.9 GB`, `275` shards | Kimi K3 License (the same LICENSE file, byte for byte) |
| [`Inferact/Kimi-K3-DSpark`](https://huggingface.co/Inferact/Kimi-K3-DSpark) (the speculative-decoding drafter, recipe A2) | `cf6b8244` | `7.12 GB` (one BF16 safetensors file, `3.56` B parameters, `2.39` B loaded by vLLM because the target's embedding is shared) | Kimi K3 License (its `LICENSE` file is byte-identical to the vellum copy, md5 `7746aa28`, checked 2026-10-12) |
| [`unsloth/Kimi-K3-GGUF`](https://huggingface.co/unsloth/Kimi-K3-GGUF), `UD-IQ1_M` (recipe B) | main; `a0836360` on 2026-10-09 (our download predates it; revision not recorded) | `604.3` GiB, `15` shards | Kimi K3 License (HF metadata `kimi-k3`) |

**The Kimi K3 License** (read in full from the moonshotai repository, 2026-10-09) is the MIT licence text with three
conditions added:

1. Keep the copyright and permission notice in all copies or substantial portions.
2. If you (with affiliates) run a "Model as a Service" business (third parties get meaningful control of inference or
   fine-tuning, e.g. an API) and have more than 20 million US dollars of revenue in any 12 consecutive months, you need
   a separate agreement with Moonshot AI before commercial use of the model or derivatives.
3. A commercial product or service with more than 100 million monthly active users or more than 20 million US dollars of
   monthly revenue must display "Kimi K3" prominently in its user interface.

Conditions 2 and 3 do not apply to internal use (nothing made available to third parties) or to use through Moonshot
AI's official products and certified partners. vellum's 3-bit checkpoint is a derivative redistributed under the same
licence with attribution, and the DSpark drafter carries the same licence file. **No weights are included in this
repository**; the hook, patch and scripts here contain no model content. Read the licence yourself before serving the
model to anyone else; this summary is not legal advice.

## A. vLLM: Kimi K3 at 3 bits across two Stations

Shape: 8,192 random-token prompt, 1,024 forced output tokens (`--ignore-eos`), `vllm bench serve`; decode = C x 1000 /
mean TPOT, aggregate across streams. C1 is the mean of three repetitions with three fresh seeds.

<!-- results -->
| Two Stations, vLLM, TP2 + EP2, Humming int3 experts in Grace | Result | Receipt |
|---|---|---|
| Decode C1 per user, tok/s (mean of three) | 18.1 | [log](../../logs/kimi-k3/vllm-two-stations.log) |
| Decode C4 aggregate, tok/s | 32.7 | [log](../../logs/kimi-k3/vllm-two-stations.log) |
| Decode C16 aggregate, tok/s, **clean rerun** (fresh seed, no prefix-cache hits; supersedes the first run's figure) | 55.23 | [log](../../logs/kimi-k3/vllm-two-stations-clean-c16.log) |
| Decode C16 aggregate, tok/s, first run (**superseded**: some prompt tokens hit the prefix cache, see below) | 52.17 | [log](../../logs/kimi-k3/vllm-two-stations.log) |
| GSM8K, first `200` test rows, greedy, thinking requested (**not verified on**, see below), % | 98.0 | [log](../../logs/kimi-k3/vllm-two-stations.log) |
| Tool calls (`get_metar`, OpenAI format), correct of five | 5 | [log](../../logs/kimi-k3/vllm-two-stations.log) |
| Coherence, `16` greedy prompts: auto-checked answers correct, of twelve | 11 | [log](../../logs/kimi-k3/vllm-two-stations.log) |
| Launch to ready, minutes | 17 | [log](../../logs/kimi-k3/vllm-two-stations.log) |
| Grace memory holding routed experts, GiB per rank (`Total CPU offloaded parameters`) | 408.27 | [log](../../logs/kimi-k3/vllm-two-stations.log) |
| GB300 mean power, C1 bench window, W (left / right) | 387.9 / 394.5 | [log](../../logs/kimi-k3/vllm-two-stations.log) |
| GB300 mean power, C4 + C16 bench window, W (left / right) | 392.9 / 400.0 | [log](../../logs/kimi-k3/vllm-two-stations.log) |
<!-- /results -->

- **Coherence:** all `16` answers are fluent and on topic (read by us; they are in the log). The one auto-check miss is
  "airplane" spelled backwards as "enalpira", a character-level slip.
- **Tool calls, canaries and a GSM8K-20 spot check** ran on the same boot with `bench/ksmoke.py`; all in the log.
- **Load:** weights load in `735` / `782` s per rank (this includes the hook's GPU repack, about three seconds per
  shard), then about two minutes of graph capture and warm-up. Each GB300 shows `242,372` MiB in use while serving.

**The first C16 figure was flagged as an upper bound; that caveat is withdrawn.** The first run used an older bench
script that gave every concurrency the same seeds, so the C4 and C16 prompt sets reused C1 repetition 1's seed, and
vLLM's prefix cache was on (its default). The server log shows a `0.0%` hit rate through C1 and C4, but during the
first wave of C16 prefills its rolling hit rate rose to 17.6% (our estimate: about `35K` of the `262K` C16 prompt
tokens came from cache; the second wave had none). That breaks this repository's `2%` rule, so we called it an upper
bound and owed a clean rerun. The clean rerun (fresh seed, separate warm-up, `0.00%` hit rate, the same stack plus
per-layer route counters) came out 5.9% higher (55.23 vs 52.17), so the prefix-cache hits did not inflate the figure;
the upper-bound caveat is withdrawn ([receipt](../../logs/kimi-k3/vllm-two-stations-clean-c16.log)). Why the first run
was lower is not isolated: the hits, the shared seed, the counters and plain run-to-run noise all differ between the
two runs. The first run's 52.17 stays in `results.jsonl` as measured, with a note. C1 and C4 were clean and are
unchanged. The rerun is a single run and includes the route counters (A0 in section A2 does too).

**Thinking was requested but not verified.** The GSM8K run passed `chat_template_kwargs: {thinking: true,
reasoning_effort: high}`, but the responses came back with an empty reasoning field (the canaries on the same boot show
`reasoning_len=0`) and the median completion was `194` tokens. Either vLLM's `kimi_k3` reasoning parser returns the
reasoning elsewhere, or the template ignores these keys, or the model barely reasons on GSM8K. 98.0% stands as measured
either way.

### Memory plan (per Station)

| Tier | What it holds |
|---|---|
| GB300 HBM (`0.95` utilisation) | dense weights, latent projections, router and the HBM share of the experts: `165.6` GiB; KV cache `66.5` GiB (`1,796,096` tokens, about 27 requests of `65,536` tokens) |
| Grace, pinned (UVA) | routed experts `408.27` GiB; host `Shmem` `408.8` GiB, `MemAvailable` about `30` GiB left |
| RTX PRO 6000 | not used |

The pinned share is close to the limit. Our hosts started from about `456` GiB of `MemAvailable` after `drop_caches`
(the launcher refuses below `420` GiB); keep anything else that pins host memory off the Station while this loads.

### What it took to load: four fixes

vLLM nightly `af7f9488` supports Kimi K3, the WNA16 path and Humming's 3-bit MoE kernels, and has UVA expert offload,
but the combination does not load as is:

1. **`preprocessor_config.json` is missing from vellum's repository.** vLLM raises `OSError: Can't load image
   processor` even with `--language-model-only`. Copy the file from `moonshotai/Kimi-K3` (it is the same model's
   processor; ours was byte-identical to the moonshotai file).
2. **The offload parameter names are different for this checkpoint.** On a WNA16 checkpoint the routed experts are
   `routed_experts.w13_weight_packed` and `routed_experts.w2_weight_packed`. The `routed_experts.w13_weight` names that
   work for MXFP4/NVFP4 checkpoints match nothing here and silently offload nothing. Check the server log for
   `Total CPU offloaded parameters: 408.27` on both ranks.
3. **3-bit packing.** vellum ships the canonical compressed-tensors int3 layout (10 values per int32, last dimension
   `359` / `308` words for the two expert shapes). vLLM's WNA16 MoE parameters and Humming expect the bit-dense layout
   (`336` / `288` words), so the load stops at `The size of tensor a (336) must match the size of tensor b (359)`. The
   hook repacks every local expert tensor on the GPU as it loads (on the CPU it took hours). `vllm/hook/test_k3hook.py`
   checks the repack bit for bit against compressed-tensors' own packer and Humming's unpack semantics.
4. **UVA offload is lost when Humming renames the weights.** Humming's conversion replaces `w13_weight_packed` with a
   new `w13_weight` parameter (and the same for `w2`). vLLM re-offloads replaced UVA parameters by name, so the renamed
   tensors silently stay in HBM; with `408` GiB per rank this ends in an out-of-memory error inside Humming's repack at
   about layer 20 (the first attempt's error is at the end of the log). The hook moves them back to pinned Grace memory
   after each layer.

Fixes 3 and 4 are in [`vllm/hook/sitecustomize.py`](vllm/hook/sitecustomize.py) (Apache-2.0, mounted at
`/opt/k3hook` and put on `PYTHONPATH` by the launcher; no vLLM code is copied into it). Both are upstream bugs in this
vLLM build, and the hook is a workaround rather than a fix.

### Setup

```bash
MODEL_ROOT=/models/hf
hf download vellum-ai/Kimi-K3-W3A16-g64 --revision 09c068683355a5c89f87d58fe292ea9627912bc4 \
  --local-dir $MODEL_ROOT/vellum-ai__Kimi-K3-W3A16-g64
hf download moonshotai/Kimi-K3 preprocessor_config.json --local-dir $MODEL_ROOT/vellum-ai__Kimi-K3-W3A16-g64
docker pull vllm/vllm-openai@sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063
# tag it as the nightly name the launcher expects, or pass IMAGE=<digest ref>
# optional, a few seconds, no GPU: check the hook's repack
docker run --rm -v $PWD/vllm/hook:/opt/k3hook:ro --entrypoint python3 <image> /opt/k3hook/test_k3hook.py
```

The weights must be on both Stations (we copied them over the 400G link). The launcher also uses the fabric pieces from
[`recipes/fabric-data-direct`](../fabric-data-direct/) (host RDMA libraries for the Data Direct overlay, the NCCL tuner
plugin with `tuner3.csv`); `DD=0` runs without them, untested for this model.

### Launch

```bash
# rank 1 on the second Station first, then rank 0
RANK0_IP=<rail ip of station A> RANK1_IP=<rail ip of station B> vllm/launch-k3.sh 1   # on B
RANK0_IP=<rail ip of station A> RANK1_IP=<rail ip of station B> vllm/launch-k3.sh 0   # on A
grep "Total CPU offloaded parameters" ~/bench/k3-*/server-rank*.log    # 408.27 on both ranks
vllm/stop-k3.sh                                                         # rank 0 first, then rank 1
```

OpenAI API on port 8010 of rank 0 as `kimi-k3`, with the `kimi_k3` reasoning and tool-call parsers.

| Flag / env | Why |
|---|---|
| `--moe-backend humming` | the only MoE kernels in this build that run 3-bit weights (Marlin does 4 and 8 bits only) |
| `--offload-backend uva --cpu-offload-gb 408 --cpu-offload-params routed_experts.w13_weight_packed routed_experts.w2_weight_packed` | fix 2 above; `408` is about `607.8` GiB of weights per rank minus what fits in HBM next to the KV cache |
| `PYTHONPATH=/opt/k3hook` + the hook mount | fixes 3 and 4 |
| `--tensor-parallel-size 2 --enable-expert-parallel` (not pipeline parallel) | vLLM issue #50947: K3 with pipeline parallelism fails in `index_fill_` |
| `VLLM_KIMI_K3_GEMM_AR=0`, `--disable-custom-all-reduce` | the fused GEMM + all-reduce and the custom all-reduce are intra-node only; across the RoCE link NCCL does the all-reduce |
| `--language-model-only` | drops the vision tower |
| `PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024` | exact-size pinned host buffers for the offloaded experts |

### Benchmarks

To reproduce the table with today's hygiene (fresh seed per concurrency and repetition, per-run prefix-hit check),
use the Nemotron recipe's bench script against this server:

```bash
NAME=k3 PORT=8010 SERVED=kimi-k3 CONCS="1 4 16" SKIP64=1 PEER=<user@station B> \
  ../nemotron-ultra/bench/bench-nemotron-ultra.sh ~/bench/k3-bench 7300
```

Add `EXTRA_ARGS="--no-enable-prefix-caching"` to the launch to rule cache hits out by construction (we have not run K3
that way). Quality: [`coherence16.py`](../dsv4-pro-two-stations/tools/coherence16.py) and
[`gsm8k200_v4.py`](../dsv4-pro-two-stations/tools/gsm8k200_v4.py) from the DS-V4-Pro recipe
(`gsm8k200_v4.py http://127.0.0.1:8010/v1/chat/completions kimi-k3 out.jsonl 16 200`), and
`bench/ksmoke.py 8010 20 8` for the canaries and tool calls.

### A2. Speculative decoding with Inferact's DSpark drafter

`Inferact/Kimi-K3-DSpark` is a five-layer, BF16, `3.56` B-parameter drafter for Kimi K3 that reads the target's hidden
states from layers `2, 23, 47, 71` and `89`, and shares the target's embedding. vLLM `af7f9488` runs it as `method:
dspark`. This section is a same-session A/B on the two-Station recipe above: first the control arm (A0, the recipe as
in section A, no drafter), then the same boot with the drafter, on the same two GB300s, in one sitting
([receipt](../../logs/kimi-k3/vllm-two-stations-dspark.log)). The schedule, the bars and our predictions were written
before the run.

**Setup of the DSpark arm.** The drafter proposes `K = 3` tokens when one request is running, `K = 2` at two or three,
and nothing (`K = 0`) at four or more: we expected verification to cost more than drafting saves at larger batches and
did not measure other thresholds. That is the `num_speculative_tokens_per_batch_size` schedule
`[[1,1,3],[2,3,2],[4,16,0]]` in [`vllm/launch-k3-dspark.sh`](vllm/launch-k3-dspark.sh). Three more pieces, none of
them in stock vLLM:

- `--use-replayssm --mamba-backend triton` (RecoverSSM): Kimi Delta Attention keeps a recurrent state per request, and
  verifying `K` drafts needs `1 + K` state slots each; RecoverSSM recovers the state instead of storing the slots. It
  had never run on this stack (Humming experts, UVA offload, TP2 + EP2) before this boot. It came up on the first
  attempt.
- [`vllm/hook/k3ds_guard.py`](vllm/hook/k3ds_guard.py), a guard we first wrote for the DS-V4-Pro DSpark drafter
  (receipts: [R3.1b](../../logs/dsv4-pro-two-stations/dspark-guard-r31b.log) and
  [R3.1c](../../logs/dsv4-pro-two-stations/dspark-skipdraft-r31c.log)), ported to K3: any rank's exception in the
  model step ends the worker instead of hanging its peer; an acceptance-gated `K` decided in the scheduler (so every
  rank sees the same `K`) that turns drafting off while the mean accepted length is below `1.6` and back on above
  `2.0`; a light step at `K = 0` that stages the drafter's context without running a draft forward; an eager warm-up
  of the drafter's eager path; and drafter graphs that never see query length zero. Every part is off unless its
  environment variable is set, and the file is loaded only with `K3DS_GUARD=1`.
- `--kv-cache-memory-bytes` (48 GiB per rank) instead of the utilisation-derived KV budget, because the drafter's
  weights and graphs are outside that accounting.

**Measurement.** Real text is the primary measure. Real text: CNN/DailyMail 3.0.0 test articles
([`abisee/cnn_dailymail`](https://huggingface.co/datasets/abisee/cnn_dailymail) at `96df5e68`) concatenated to about
8,200 K3 prompt tokens, thinking off, `1,024` forced output tokens, temperature 0, streaming. C1 is the mean of three
repetitions of three prompts; C2, C4 and C16 are one run each on the first four, eight and sixteen prompts of one
disjoint slice. Slices are disjoint, each arm has its own nonce, and the server's prefix-cache hit rate was measured
around every run (`0.0%` throughout). The pool is rebuilt from public sources with
[`bench/build_pool_k3.py`](bench/build_pool_k3.py) (it takes the output of
[`build_realtext_pool.py`](../dsv41-flash-sidecar/bench/realtext/) and keeps only the prompt text) and driven by
[`bench/kbench_chat_ds.py`](bench/kbench_chat_ds.py), which also reads vLLM's speculative-decoding counters. Random
ids are a control, and **they depend on the sampling setting**: the table's random-id row was measured with
`--temperature 0` (greedy) and one prompt per repetition by our kit script; the repository's own random-id bench
([`bench-nemotron-ultra.sh`](../nemotron-ultra/bench/bench-nemotron-ultra.sh)) passes no temperature, and gives a
different answer (see the boot receipt below).

<!-- results -->
| Same session, two Stations | No drafter (A0) | DSpark | Ratio | Receipt |
|---|---|---|---|---|
| Real text, C1 decode per user, tok/s (mean of three) | 18.01 | 26.23 | 1.456 | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
| Real text, C1 mean accepted length (verify step, `K = 3`) | | 2.74 | | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
| Real text, C2 decode aggregate, tok/s | 24.61 | 32.7 | 1.329 | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
| Real text, C4 decode aggregate, tok/s (drafter off) | 30.78 | 30.29 | 0.984 | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
| Real text, C16 decode aggregate, tok/s (drafter off) | 45.17 | 45.36 | 1.004 | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
| Random ids with `--temperature 0` (greedy), C1 decode per user, tok/s (mean of three; kit script, not reproduced by the repo bench path, see below) | 18.05 | 27.6 | 1.529 | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
| Random ids with `--temperature 0`, C1 mean accepted length | | 2.98 | | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
<!-- /results -->

<!-- results -->
| Quality, hygiene and cost, same session | No drafter (A0) | DSpark | Receipt |
|---|---|---|---|
| GSM8K, first `200` test rows, thinking requested (not verified, as in section A), % | 97.5 | 98.5 | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
| Coherence, `16` greedy prompts: auto-checked answers correct, of twelve | 11 | 11 | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
| Teacher-forced top-1 flips against the A0 reference, % (A0 column: A0 run twice, the floor) | 0.0 | 0.042 | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
| Free-running greedy continuations identical to the A0 reference, of `32` (A0 column: A0 against itself) | 8 | 7 | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
| Streaming soak, requests completed without error, of the same number submitted | | 135 | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
| Weights in GPU memory per rank (vLLM's `Model loading took`), GiB | 165.6 | 168.49 | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
| KV cache per rank, tokens (A0: utilisation `0.95`; DSpark: fixed budget) | 1,796,096 | 1,052,945 | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
| Lowest free GPU memory the guard saw on either rank over the whole DSpark arm, GiB | | 21.15 | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
| Launch to ready, minutes | 15 | 16 | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
| Highest server prefix-cache hit rate over every measured run of both arms, % | 0.0 | 0.0 | [log](../../logs/kimi-k3/vllm-two-stations-dspark.log) |
<!-- /results -->

**Reproduced with the published launcher and the repository's own benches.** After the A/B we booted
`vllm/launch-k3-dspark.sh` exactly as it is in this tree (one boot, launcher defaults except the rail addresses and
paths), on both Stations, and ran `bench/ksmoke.py`, the real-text C1 bench and the repository's random-id bench
([receipt](../../logs/kimi-k3/vllm-two-stations-boot-receipt.log)). The A0 control in the first column is the earlier
session's (the receipt session had no control arm).

<!-- results -->
| Published launcher, boot receipt | Result | Ratio | Receipt |
|---|---|---|---|
| Real text, C1 decode per user, tok/s (mean of three) | 26.37 | 1.464 over A0 | [log](../../logs/kimi-k3/vllm-two-stations-boot-receipt.log) |
| Real text, C1 mean accepted length | 2.74 | | [log](../../logs/kimi-k3/vllm-two-stations-boot-receipt.log) |
| `ksmoke.py` GSM8K, first `20` rows, % (a smoke test) | 95.0 | | [log](../../logs/kimi-k3/vllm-two-stations-boot-receipt.log) |
| `ksmoke.py` tool calls correct, of five | 5 | | [log](../../logs/kimi-k3/vllm-two-stations-boot-receipt.log) |
| Random ids with default sampling (the repo bench), C1 decode per user, tok/s (one repetition) | 17.52 | 0.968 over the first run | [log](../../logs/kimi-k3/vllm-two-stations-boot-receipt.log) |
| Random ids with default sampling, C1 mean accepted length | 1.02 | | [log](../../logs/kimi-k3/vllm-two-stations-boot-receipt.log) |
| Launch to ready, minutes | 16 | | [log](../../logs/kimi-k3/vllm-two-stations-boot-receipt.log) |
<!-- /results -->

- **Real text reproduces.** 26.37 against 26.23, within a percent, with acceptance 2.74 in both.
- **Random ids do not, and now we know why.** The kit script passed `--temperature 0`. `bench-nemotron-ultra.sh`
  passes none, so `vllm bench serve` uses the server's default sampling (the argument dumps in the two logs show
  `temperature=0.0` and `temperature=None`). A sampled continuation of a random-token prompt is random, the drafter
  cannot predict it (mean accepted length 1.02), the acceptance-gated `K` turns drafting off (its state at the end of
  the run: off, EMA 1.0) and C1 is the no-drafter speed minus 3%: 17.52 against the first run's 18.1, which also used
  default sampling. Under greedy decoding the same prompts fall into repetitive output, the drafter predicts it
  (accepted length 2.98) and C1 rises to 27.6. **The greedy figure is a property of greedy decoding on random tokens,
  not of the recipe, and it does not transfer to the repository's bench path or to real text.** We keep it in the
  table, labelled, because it is a measured number, and do not use it as a headline.
- **A note on the shape.** The random-id rows of this repository that came from `bench-nemotron-ultra.sh` and
  `tools/bench_e3.sh` use the server's default sampling (no `--temperature` passed); the top-level README now says so.
  For decode without a drafter this changes little (18.1 against 18.05 greedy); with a drafter it changes everything.
  Only the A/B kit's random-id rows above are explicitly greedy.

**What it means.**

- **One user gets a real speed-up, a batch gets none.** Real-text C1 rises from 18.01 to 26.23 tok/s (1.456 times), C2
  by a third, and from C4 up the drafter is off and the numbers are the control's within one run's noise (0.984 at C4,
  1.004 at C16). The guard's light step at `K = 0` costs nothing we can see. This is a single-user feature: it does
  not raise aggregate throughput.
- **The spread across repetitions is real.** The three real-text C1 repetitions are 26.96, 27.2 and 24.53 tok/s, with
  mean accepted lengths 2.811, 2.841 and 2.559: the gain follows how predictable the prompts' continuations are.
  Random-id repetitions: 32.93, 25.5 and 24.37 tok/s.
- **Quality gates held, and the evidence is thin.** GSM8K is two problems of `200` above the control (the earlier
  recipe run measured 98.0, so the three runs span one point). The top-1 flip rate against the control is 2 of `4,708`
  positions. Free-running greedy continuations were identical for 7 of `32` prompts against the control, but the
  control was identical to *itself* for only 8 of `32`: on this stack greedy free-running output is not run-to-run
  deterministic with or without a drafter, so a low identical-count is not evidence of damage. The soak ran 20 minutes
  at a concurrency schedule `1, 3, 8, 2, 4, 1, 6, 2` (every `K` tier and the transitions between them) and all `135`
  requests it submitted completed: it covers GSM8K with and without thinking, random ids, and long and short synthetic
  text, but thinking is capped at `2,048` / `1,024` tokens and only `135` requests fit in the window, so it is a hang
  and error check, not a statistical one.
- **The cost is memory, not speed.** The drafter adds about 2.9 GiB of weights per rank, and we gave the KV cache 48
  GiB instead of 66.5 GiB: `1,052,945` tokens instead of `1,796,096`, still room for 16 requests of `65,536` tokens.
  The guard's free-memory floor was 21.15 GiB per rank, so the KV budget can probably go higher; we did not look for
  the limit.

**Where our predictions missed.** Written before the boot, applied mechanically afterwards ([the predictions are in
the log's `summary.md`](../../logs/kimi-k3/vllm-two-stations-dspark.log)):

| Measure | Predicted | Measured | Verdict |
|---|---|---|---|
| Real-text C1 decode (tok/s) | 24 to 30 | 26.23 | in range |
| Real-text C1 ratio | 1.35 to 1.65 | 1.456 | in range |
| Mean accepted length, real-text C1 | 2.8 to 3.0 | 2.74 | **missed low** (two of three repetitions were above 2.8) |
| Real-text C2 ratio | 1.2 to 1.4 | 1.329 | in range |
| Real-text C4 ratio | 0.95 to 1.02 | 0.984 | in range |
| Real-text C16 ratio | 0.97 to 1.00 | 1.004 | 0.4% above the range; noise on a single run |
| **Random-id C1 ratio** | **0.95 to 1.02** | **1.529** greedy; **0.968** default sampling | **missed high under greedy only; the repo bench path matches the prediction** |

The random-id miss turned out to depend on the sampling setting, not on the recipe. We expected the acceptance-gated
`K` to turn the drafter off on random ids (the prediction of 0.95 to 1.02 says "mostly off"). Under `--temperature 0`
it did not: the drafter accepted 2.98 tokens per step and C1 rose by a half, because greedy decoding of a random-token
prompt falls into repetitive output (our reading; we did not inspect the outputs). Under the default sampling of the
repository's own bench the prediction held (0.968, accepted length 1.02, drafting off). So the random-id speed-up
should not be read as a speed-up of anything but greedy random-id decoding: the real-text row is the one that
transfers.

**Not run, or open.**

- The launcher and hook files in this folder were booted once as published (the boot receipt above): real text
  reproduces, random ids depend on the sampling setting. The A/B arms themselves loaded the same hook plus our
  per-layer route counters (`K3T_COUNT`, in both arms and not published), and the kit's own scripts for the random-id
  runs. The receipt session had no control arm, so its ratio is against the A/B's control.
- Each arm booted once and each configuration was measured once (C1 three times). The pre-registration allowed a
  fallback second attempt; it was not needed.
- Power was not recorded. Driver 595.99.02, kernel 7.0.0-1019-nvidia-64k and Ubuntu 24.04.4 were queried on both
  Stations after the run (unchanged by policy), not logged during it.
- A nsys profile of the DSpark arm was taken but is not part of this release.
- Thinking-on quality with the drafter is as unverified as without it (see above).

**Run it.**

```bash
hf download Inferact/Kimi-K3-DSpark --revision cf6b8244620e7ea4b0651d214f28e89eac75bed6 \
  --local-dir $MODEL_ROOT/Inferact__Kimi-K3-DSpark          # on both Stations; sha256 of model.safetensors f9972a63...2053
python3 bench/check_drafter.py $MODEL_ROOT/Inferact__Kimi-K3-DSpark \
  --target-config $MODEL_ROOT/vellum-ai__Kimi-K3-W3A16-g64/config.json   # CPU only: reads the header, not the weights
# same ranks, same order as section A; DRAFTER defaults to /models/hf/Inferact__Kimi-K3-DSpark
RANK0_IP=<rail ip of station A> RANK1_IP=<rail ip of station B> vllm/launch-k3-dspark.sh 1   # on B
RANK0_IP=<rail ip of station A> RANK1_IP=<rail ip of station B> vllm/launch-k3-dspark.sh 0   # on A
# control arm of the same kit: DSPARK=0 vllm/launch-k3-dspark.sh <rank>   (equivalent to launch-k3.sh)
grep -E "K3DS (installing|KCTL)|Resolved architecture: K3DSpark" ~/bench/k3-*/server-rank0.log
# real text (build the pool once; see bench/build_pool_k3.py)
python3 bench/kbench_chat_ds.py http://127.0.0.1:8010 kimi-k3 pool/long/eval-c1-r1-try1.jsonl out.json 1 \
  --warm pool/long/eval-c1-r1-try1-warm.jsonl --nonce mine-1
```

Random-id runs: `bench-nemotron-ultra.sh` uses the server's default sampling (drafting ends up off, C1 is the
no-drafter speed); to reproduce the greedy figure run `vllm bench serve` with `--temperature 0` and `--ignore-eos`, a
fresh seed per invocation and one prompt per repetition. The acceptance length is `1 + accepted / drafts` from the
deltas of `vllm:spec_decode_num_accepted_tokens_total` and `vllm:spec_decode_num_drafts_total` around the run
(`bench-nemotron-ultra.sh` prints it as `mtp_acc_len`, `kbench_chat_ds.py` as `acceptance_len`).

## B. llama.cpp: Kimi K3 UD-IQ1_M on one Station, two Stations and four GPUs

Shape (different from section A; do not compare the two directly): [`bench/kbench.py`](bench/kbench.py), llama-server
`/completion` with an about 130-token prompt, `ignore_eos`, `cache_prompt: false`. Per user = median of llama-server's
own decode rate; aggregate = output tokens / wall time (prefill included). A 130-token context decodes faster than the
8,192-token context of section A, which favours these rows.

<!-- results -->
| tok/s | C1 per user | C4 aggregate | C16 aggregate | Receipt |
|---|---|---|---|---|
| One Station, stock llama.cpp, `63` expert layers in Grace | 10.49 | | | [log](../../logs/kimi-k3/llamacpp-one-station.log) |
| One Station, **with the C2C zero-copy patch** | 16.0 | 22.1 | 26.55 | [log](../../logs/kimi-k3/llamacpp-one-station.log) |
| Two Stations over RPC (GB300s only, `27` expert layers in Grace, zero-copy) | 17.91 | 24.64 | 30.05 | [log](../../logs/kimi-k3/llamacpp-two-stations.log) |
| Four GPUs over RPC (both GB300s + both RTX PRO 6000s, all in GPU memory; C1 mean of three) | 18.18 | 28.41 | 33.47 | [log](../../logs/kimi-k3/llamacpp-four-gpus.log) |
<!-- /results -->

<!-- results -->
| Quality and the patch's gain | Result | Receipt |
|---|---|---|
| GSM8K first `200` rows, one Station with zero-copy, reasoning on, `max_tokens 2,048`, % | 93.5 | [log](../../logs/kimi-k3/llamacpp-one-station.log) |
| GSM8K first `50` rows, two Stations, % | 96.0 | [log](../../logs/kimi-k3/llamacpp-two-stations.log) |
| GSM8K first `50` rows, four GPUs, % | 98.0 | [log](../../logs/kimi-k3/llamacpp-four-gpus.log) |
| C1 gain from the zero-copy patch on one Station (both `-np 1`), % | 52.4 | [log](../../logs/kimi-k3/llamacpp-one-station.log) |
<!-- /results -->

The two smaller GSM8K samples are smoke tests, not comparable to the GSM8K-200 rows. Tool calls were `5/5` sequential
in every configuration, and wikitext-2 perplexity (`512`-token context, `24` chunks) is `2.0439` with and without
the zero-copy patch. The llama.cpp GSM8K harness ([`bench/ksmoke.py`](bench/ksmoke.py)) differs from section A's in
prompt, answer extraction and token budget, so 93.5% vs 98.0% is suggestive, not a controlled comparison. With
reasoning off, the one-Station build scored `42/50` on GSM8K-50 and `1/5` tool calls: serve it with reasoning on.

### The C2C zero-copy patch

[`llamacpp/patches/ggml-cuda-c2c-zerocopy.patch`](llamacpp/patches/ggml-cuda-c2c-zerocopy.patch) is one line in
`ggml-cuda.cu`. llama.cpp marks every CUDA device as non-integrated, so expert tensors kept in host memory
(`--n-cpu-moe`, `-ot ...=CPU`) are copied to the GPU or computed on the CPU at every use. On a GB300 the GPU can read
pinned Grace memory in place over NVLink-C2C (see the [atlas](../../atlas/microbench/)). With
`GGML_CUDA_C2C_ZEROCOPY=1`, compute-capability 10.x devices are reported as integrated, and the CUDA backend reads the
host-resident experts directly: one graph split, every expert matmul on the GB300. Unset, nothing changes. Use it
with `--load-mode none` so the host buffers are pinned.

- It re-enables, behind an opt-in variable, something upstream disabled because of corrupted output on other
  hardware (llama.cpp #15034). On this model and hardware the wikitext-2 perplexity with and without it is identical
  to four decimals. Check your own model before trusting it.
- Only set it on a C2C-attached Blackwell (GB300 or GB200). A PCIe Blackwell card would also be marked integrated.
- We ran the same one-line change on llama.cpp `99b95488` and `c479922a`; this file applies to `c479922a` and to
  master `50e3e3e480` (2026-10-09). The patch file is MIT-licensed, like llama.cpp, so it can go upstream as is.

### Build: not with CUDA 13.2.0 or 13.2.1

**llama.cpp built with nvcc 13.2.51 or 13.2.78 (CUDA 13.2.0 / 13.2.1; our Stations' host toolkit is 13.2.51) produces garbage
for this model**: greedy "1 2 3 4 5 6 7 8" continues "9 10 12 12 12", chat loops, wikitext-2 perplexity is about `313`
instead of `2.04` (both in the one-Station log). The cause is the compiler, specifically NVVM (`cicc`): at `-O2` and
above it drops the `& 0xFF` when a byte is read out of a packed 32-bit word through a `uint8_t*` cast and used as a
table index, so the IQ1_S / IQ2_S / IQ3_S kernels read past their lookup tables. Kimi K3 UD-IQ1_M has `209` IQ1_S
tensors. Not the cause: the runtime libraries, `ptxas`, the arch flag or the GPU (the same bug is reported on consumer
Blackwell).

- **Minimal reproducer, no llama.cpp:** [`cuda132/byteidx.cu`](cuda132/byteidx.cu). `nvcc -O3 -arch=sm_103 byteidx.cu`
  gives `4096/4096` mismatches with 13.2.51 and 13.2.78, none with 13.0.88, 13.1.115, 13.2.86, 13.3.73 or 13.4.92. The
  PTX shows it: `grep -c and.b16` is `4` on a good compiler, `0` on a bad one.
- **Fixes, any one:** build in a CUDA 13.0 or 13.1 container (what [`llamacpp/build-llamacpp.sh`](llamacpp/build-llamacpp.sh)
  does, in the SGLang dev image); use nvcc 13.2.86 (CUDA 13.2.2) or newer; or apply llama.cpp PR #28784
  (`__byte_perm`, closed unmerged). Swapping only `nvvm/` from a 13.2.86 wheel into a 13.2.51 toolkit also fixes it.
- **Guard:** `test-backend-ops -b CUDA0 test -o MUL_MAT -p iq3_s` takes seconds and fails on a bad build; the build
  script runs it when a GB300 is visible.
- Upstream: llama.cpp #21255 (closed) and #28581 (open, same nvcc bisect on sm_120). We have not yet posted the GB300
  data there.

### Run

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout c479922a
git apply ../llamacpp/patches/ggml-cuda-c2c-zerocopy.patch && cd ..
llamacpp/build-llamacpp.sh llama.cpp '103-real;120a-real'          # '103-real' is enough without the RTX PRO 6000s
hf download unsloth/Kimi-K3-GGUF --include 'UD-IQ1_M/*' --local-dir /models/gguf/unsloth__Kimi-K3-GGUF

# one Station
BIN=$PWD/llama.cpp/build/bin llamacpp/launch-one-station.sh

# two Stations (GB300s only): rpc server on B, llama-server on A
BIN=... llamacpp/rpc-server.sh gb300 <rail ip of B> 50052                        # on B
BIN=... PEER_IP=<rail ip of B> llamacpp/launch-rpc.sh two                         # on A

# four GPUs: two rpc servers on B, one on A (A's RTX PRO 6000), llama-server on A's GB300
BIN=... llamacpp/rpc-server.sh gb300 <rail ip of B> 50052; BIN=... llamacpp/rpc-server.sh rtx <rail ip of B> 50053   # on B
BIN=... llamacpp/rpc-server.sh rtx 127.0.0.1 50053                                                                # on A
BIN=... PEER_IP=<rail ip of B> llamacpp/launch-rpc.sh four                                                        # on A

python3 bench/kbench.py 8015 1 3 256; python3 bench/kbench.py 8015 16 16 128
MAXTOK=2048 python3 bench/ksmoke.py 8015 200 8 gsm8k_test.jsonl
```

**One GPU per CUDA process.** On our Stations a process that is given both the GB300 and the RTX PRO 6000 can use only
one of them: llama.cpp fails with `ggml_cuda_init: failed to initialize CUDA: system not yet initialized`, and torch
reports two devices but raises `invalid device ordinal` on the second. Each GPU works on its own. So the four-GPU run
uses one RPC server per GPU, including one on loopback for the local RTX PRO 6000.

## C. Caveats

- **C16 is far below `100` tok/s** in every configuration (section A; the vLLM figure is the clean rerun).
  Our reading (modelled, not profiled): decode at C16 is bound by Grace. 16 tokens with 16 routed experts each touch
  roughly a quarter of each rank's `448` local experts in every layer, and most expert bytes (`408` of about `608` GiB
  of weights per rank) are in Grace, read at C2C speed. Hot-expert pinning (as in our DS-V4-Pro recipe) and RTX PRO
  6000 expert tiers are the next levers. Neither exists for K3 here yet: our pin hook relies on an expert map that we
  have not found in this build's Humming path, and we have no 3-bit expert kernel for the RTX PRO 6000.
- **llama.cpp is maxed out on this hardware.** Moving every weight out of Grace onto four GPUs moved C1 from 17.91 to
  18.18 tok/s. With a layer split, one token passes through the devices one after another, with an RPC hop between
  each; more devices add capacity, not speed. The RTX PRO 6000s also have far less memory bandwidth than the GB300s.
  Tensor parallelism (section A) is the way past it.
- **A GB300 and an RTX PRO 6000 cannot share one CUDA process** on these Stations (above). Any plan that needs both
  in one engine process, such as tensor parallelism across them or an in-process expert sidecar, does not work; the
  sidecars in our other recipes run as separate processes.
- **Different driver.** The first vLLM run used NVIDIA driver 595.99.02 (an unattended upgrade had landed that morning); the
  llama.cpp runs and the rest of this repository used 595.91.07. The 2026-10-13 session (section A2) used the same driver 595.99.02, kernel 7.0.0-1019-nvidia-64k and Ubuntu 24.04.4, queried after the run and unchanged by policy rather than logged during it.
- **The 3-bit checkpoint's quality is otherwise unpublished** (vellum's card says its evaluation is being finalised).
  Our GSM8K-200 and the coherence and tool checks are the only quality evidence we have; no long-context, coding or
  multilingual checks were run, and nothing was compared against the full-precision model.
- **Not measured:** prefill throughput, long contexts, power for the llama.cpp runs at C16 and for the DSpark runs, a
  run of the vLLM recipe with prefix caching off, speculative decoding on the llama.cpp builds, and DSpark with thinking
  verified on (section A2).

## Credits

- **Moonshot AI** for Kimi K3 and its licence.
- **Vellum** ([vellum-ai/Kimi-K3-W3A16-g64](https://huggingface.co/vellum-ai/Kimi-K3-W3A16-g64)) for the 3-bit
  quantisation that makes the two-Station fit possible.
- **unsloth** for the `UD-IQ1_M` GGUF.
- **Inferact** ([Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark), Kimi K3 License) for the drafter
  checkpoint; no weights are included. The `dspark` method, its config class and the DFlash proposer it inherits from
  are vLLM's; `hook/k3ds_guard.py` patches around them and copies none of that code.
- **vLLM** (Apache-2.0) for the Kimi K3 model code, the WNA16 path, UVA offload and the Humming kernels;
  **compressed-tensors** (Apache-2.0) for the packing format the hook converts from. The hook copies no code from
  either.
- **llama.cpp / ggml** (MIT): the C2C patch is against it, and `cuda132/byteidx.cu` reproduces the indexing pattern
  of its `vec_dot_iq3_s_q8_1` (MIT, notice kept in the file). The nvcc bisect matches the one krllx posted on
  llama.cpp #28581, and PR #28784's author found the `__byte_perm` workaround first.
- **ciprianveg** ([gb10-vllm](https://github.com/ciprianveg/gb10-vllm)): the sixteen-Spark K3 reference (no licence
  on that repository; nothing copied, numbers quoted only in the top-level README).
