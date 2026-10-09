# Kimi K3 on DGX Station: 3-bit in vLLM across two Stations, 1.6-bit in llama.cpp

Moonshot AI's Kimi K3 (93 layers, 896 routed experts per MoE layer with 16 active, Kimi Delta Attention plus MLA) is
about `1.56 TB` as released. Two Stations have about `1.32 TB` of GPU-reachable memory for weights once KV cache and a
host floor are set aside, so no unpruned 4-bit K3 fits. This folder has two ways to run it anyway:

- **A. vLLM, two Stations, 3-bit (recommended).** `vellum-ai/Kimi-K3-W3A16-g64` (GPTQ int3 routed experts, every other
  weight BF16) with TP2 + EP2 across one 400G link, vLLM's Humming 3-bit MoE kernels, and `408` GiB per rank of routed
  experts in Grace memory, read by the GB300 over NVLink-C2C (UVA offload). Needs a small load hook (included) and four
  fixes to load at all.
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
licence with attribution. **No weights are included in this repository**; the hook, patch and scripts here contain no
model content. Read the licence yourself before serving the model to anyone else; this summary is not legal advice.

## A. vLLM: Kimi K3 at 3 bits across two Stations

Shape: 8,192 random-token prompt, 1,024 forced output tokens (`--ignore-eos`), `vllm bench serve`; decode = C x 1000 /
mean TPOT, aggregate across streams. C1 is the mean of three repetitions with three fresh seeds.

<!-- results -->
| Two Stations, vLLM, TP2 + EP2, Humming int3 experts in Grace | Result | Receipt |
|---|---|---|
| Decode C1 per user, tok/s (mean of three) | 18.1 | [log](../../logs/kimi-k3/vllm-two-stations.log) |
| Decode C4 aggregate, tok/s | 32.7 | [log](../../logs/kimi-k3/vllm-two-stations.log) |
| Decode C16 aggregate, tok/s (**upper bound**: some prompt tokens hit the prefix cache, see below) | 52.17 | [log](../../logs/kimi-k3/vllm-two-stations.log) |
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

**The C16 figure is an upper bound.** This run used an older bench script that gave every concurrency the same seeds,
so the C4 and C16 prompt sets reused C1 repetition 1's seed, and vLLM's prefix cache was on (its default). The server
log shows a `0.0%` hit rate through C1 and C4, but during the first wave of C16 prefills its rolling hit rate rose to
17.6% (our estimate: about `35K` of the `262K` C16 prompt tokens came from cache; the second wave had none). That breaks
this repository's `2%` rule. Cache hits shorten prefill, and at C16 new prefills interleave with decode, so the effect
on decode TPOT should be small, but we did not measure it. C1 and C4 are clean. A clean C16 rerun is owed.

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

- **C16 is far below `100` tok/s** in every configuration, and the vLLM C16 figure is an upper bound (section A).
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
- **Different driver.** The vLLM run used NVIDIA driver 595.99.02 (an unattended upgrade had landed that morning); the
  llama.cpp runs and the rest of this repository used 595.91.07.
- **The 3-bit checkpoint's quality is otherwise unpublished** (vellum's card says its evaluation is being finalised).
  Our GSM8K-200 and the coherence and tool checks are the only quality evidence we have; no long-context, coding or
  multilingual checks were run, and nothing was compared against the full-precision model.
- **Not measured:** prefill throughput, long contexts, speculative decoding, power for the llama.cpp runs at C16, and
  a run of the vLLM recipe with prefix caching off.

## Credits

- **Moonshot AI** for Kimi K3 and its licence.
- **Vellum** ([vellum-ai/Kimi-K3-W3A16-g64](https://huggingface.co/vellum-ai/Kimi-K3-W3A16-g64)) for the 3-bit
  quantisation that makes the two-Station fit possible.
- **unsloth** for the `UD-IQ1_M` GGUF.
- **vLLM** (Apache-2.0) for the Kimi K3 model code, the WNA16 path, UVA offload and the Humming kernels;
  **compressed-tensors** (Apache-2.0) for the packing format the hook converts from. The hook copies no code from
  either.
- **llama.cpp / ggml** (MIT): the C2C patch is against it, and `cuda132/byteidx.cu` reproduces the indexing pattern
  of its `vec_dot_iq3_s_q8_1` (MIT, notice kept in the file). The nvcc bisect matches the one krllx posted on
  llama.cpp #28581, and PR #28784's author found the `__byte_perm` workaround first.
- **ciprianveg** ([gb10-vllm](https://github.com/ciprianveg/gb10-vllm)): the sixteen-Spark K3 reference (no licence
  on that repository; nothing copied, numbers quoted only in the top-level README).
