# dgx-station-frontier

Receipts-first serving recipes and measurements for a pair of NVIDIA DGX Station GB300 desktops, from RonanLabs.
Every number in this repository is one row in [`results.jsonl`](results.jsonl), and every row points at a scrubbed
log excerpt under [`logs/`](logs/). [`tools/validate.py`](tools/validate.py) checks both directions: each row against
its log, and each performance number in the READMEs against `results.jsonl`.

Six recipes and a microbenchmark atlas (the Nemotron and Kimi K3 recipes were added after the first release).

| Folder | What it is |
|---|---|
| [`recipes/dsv41-flash-sidecar/`](recipes/dsv41-flash-sidecar/) | DeepSeek-V4.1-Flash on one Station, with the RTX PRO 6000 serving the cold experts (original-el8's sidecar design on public b12x). Also run as two independent copies, one per Station (DP2) |
| [`recipes/dsv4-pro-two-stations/`](recipes/dsv4-pro-two-stations/) | DeepSeek-V4-Pro-0813 (1.6T parameters) across both Stations: E1 stock, E2 hot/cold expert split, E3 with an RTX PRO 6000 warm-expert tier on each Station. Plus the bundled DSpark drafter: hung unfixed, then made servable by a rank-symmetric guard and skip-draft (20-min streaming soak passed) |
| [`recipes/nemotron-ultra/`](recipes/nemotron-ultra/) | NVIDIA Nemotron 3 Ultra 550B-A55B NVFP4 (general and IOI competitive-coding checkpoints) on one Station with routed experts in Grace memory, and on two Stations with every weight in HBM. Stock vLLM, no patches |
| [`recipes/kimi-k3/`](recipes/kimi-k3/) | Moonshot Kimi K3 at 3 bits (vellum W3A16) across both Stations in vLLM, with most routed experts in Grace memory (load hook and four load fixes included); and the 1.6-bit GGUF in llama.cpp on one Station, two Stations and four GPUs, with our C2C zero-copy patch and a CUDA 13.2 miscompile warning |
| [`recipes/fabric-data-direct/`](recipes/fabric-data-direct/) | The cross-Station fabric: Data Direct inside containers, a per-message-size NCCL tuner, the lossless traffic class, and what small-message latency does and does not allow |
| [`atlas/microbench/`](atlas/microbench/) | C2C, HBM, Grace STREAM, RTX PRO 6000 to Grace, CPU vs GPU expert compute, GEMM power |
| [`articles/`](articles/) | Plain-English write-ups of the results above. [Where should your experts live?](articles/where-should-your-experts-live/) (Grace vs HBM vs RTX PRO 6000); [What does an expert sidecar actually do?](articles/what-does-an-expert-sidecar-do/); [One Station with a sidecar vs two Stations without](articles/one-station-with-a-sidecar-vs-two-without/) |

## Hardware

Two DGX Station GB300 desktops. Each has:

- one GB300 GPU (`256,703 MiB` of HBM3e visible, `power.limit` [1,300](logs/atlas-microbench/gemm-power.log) W) joined to a
  72-core Grace CPU with about 494 GiB of LPDDR5X over NVLink-C2C;
- one RTX PRO 6000 Blackwell Max-Q (`97,887 MiB`, `power.limit` [300](logs/dsv41-flash-sidecar/dp2-left.log) W) on PCIe;
- one ConnectX-8 SuperNIC with two 400G ports.

**All cross-Station results use one 400G DAC between the two ConnectX-8s (RoCEv2, no switch).** The second port is not
cabled yet. Software for every row: driver 595.91.07 (except the Kimi K3 vLLM rows: 595.99.02), host CUDA 13.2, kernel 7.0.0-1019-nvidia-64k, Ubuntu 24.04.4.
Engines run in containers pinned by digest; each row of `results.jsonl` names its image digest and versions.

## Headline results

Shape unless noted: 8,192 random-token prompt, 1,024 forced output tokens, temperature 0 (the shape catid published
for Station-pair results, so the numbers compare). "C" is concurrency; aggregate = output tokens per second across all
streams.

<!-- results -->
| Result | Ours | Receipt | Best public comparison |
|---|---|---|---|
| DS-V4.1-Flash, one Station + its RTX PRO 6000 as cold-expert sidecar, C16 aggregate | [1,856](recipes/dsv41-flash-sidecar/) tok/s | [log](logs/dsv41-flash-sidecar/one-station-c1-c64.log) | [1,628.6](https://github.com/original-el8/dgx-station-gb300-research) tok/s, same design on one Station (el8); [1,677.7](https://github.com/catid/dgx_station_benchmarks) tok/s on two Stations (catid) |
| DS-V4.1-Flash DP2: two independent copies, one per Station, C32 / C64 aggregate (sum of both) | [4,548](recipes/dsv41-flash-sidecar/) / [5,774](recipes/dsv41-flash-sidecar/) tok/s | [left](logs/dsv41-flash-sidecar/dp2-left.log), [right](logs/dsv41-flash-sidecar/dp2-right.log) | one model across two Stations: [2,117](https://github.com/catid/dgx_station_benchmarks) / [3,401.6](https://github.com/catid/dgx_station_benchmarks) tok/s (catid TP2) |
| DS-V4.1-Flash, C1 per user (each DP2 copy, mean of three warm reps) | [445](recipes/dsv41-flash-sidecar/) / [452](recipes/dsv41-flash-sidecar/) tok/s | [left](logs/dsv41-flash-sidecar/dp2-left.log), [right](logs/dsv41-flash-sidecar/dp2-right.log) | |
| DS-V4-Pro-0813 across both Stations, stock vLLM (E1): C1 / C16 aggregate / GSM8K-200 | [37.8](recipes/dsv4-pro-two-stations/) tok/s / [109.5](recipes/dsv4-pro-two-stations/) tok/s / [98.0](recipes/dsv4-pro-two-stations/)% | [log](logs/dsv4-pro-two-stations/e1-stock.log) | on one Station: @antirez, Q2 quant in DwarfStar ([post](https://x.com/antirez/status/2089060410359972091)); ours is released precision in vLLM |
| DS-V4-Pro with RTX PRO 6000 warm tiers (E3-A16): C16 aggregate, random ids / private real text; GSM8K-200 | [164.6](recipes/dsv4-pro-two-stations/) / [230.8](recipes/dsv4-pro-two-stations/) tok/s; [97.5](recipes/dsv4-pro-two-stations/)% | [log](logs/dsv4-pro-two-stations/e3-a16-sidecar.log) | |
| `Nemotron-3-Ultra` NVFP4: one Station (TP1, experts partly in Grace) C1 / C32 aggregate; two Stations (TP2+EP2, all in HBM, IOI coding checkpoint of the same architecture) C1 / C32 aggregate | [39.7](recipes/nemotron-ultra/) / [155.2](recipes/nemotron-ultra/) tok/s; [99.1](recipes/nemotron-ultra/) / [889.1](recipes/nemotron-ultra/) tok/s | [one Station](logs/nemotron-ultra/ultra-one-station.log), [two Stations](logs/nemotron-ultra/cc-two-stations.log) | none found on DGX Station |
| `Kimi-K3` with int3 experts (vellum W3A16) across both Stations, vLLM TP2+EP2, `408 GiB` per Station of routed experts in Grace: C1 / C16 aggregate (upper bound, prefix-cache hits) / GSM8K-200 | [18.1](recipes/kimi-k3/) tok/s / [52.17](recipes/kimi-k3/) tok/s / [98.0](recipes/kimi-k3/)% | [log](logs/kimi-k3/vllm-two-stations.log) | none found on DGX Station; on sixteen DGX Sparks with speculative decoding and a different prompt shape: [29.81](https://github.com/ciprianveg/gb10-vllm/tree/main/kimi-k3/v5) tok/s C1 (ciprianveg) |
| Fused GLM-5.3 TP2+EP2 over the rail: `64K`-token prefill with Data Direct + tuner3 + traffic class `106` vs stock | [11,173](recipes/fabric-data-direct/) vs [8,033](recipes/fabric-data-direct/) tok/s ([+39](recipes/fabric-data-direct/)%) | [log](logs/fabric-data-direct/fused-glm-dd-ab.log) | |
| Smallest cross-Station all-reduce on the engine path (`14 KB`, CUDA graph) | [31.67](recipes/fabric-data-direct/) µs | [log](logs/fabric-data-direct/lambda-engine-path.log) | |
| GB300 reading pinned Grace memory over C2C (GPU kernel, zero-copy) | [352.8](atlas/microbench/) GB/s | [log](logs/atlas-microbench/gb300-membench.log) | |
<!-- /results -->

What the comparisons mean:

- **One Station beats the published two-Station C16 figure** with the same model, because the RTX PRO 6000 takes the
  cold experts. It is not a like-for-like topology comparison, and we say so wherever we quote it.
- **DP2 is two independent servers.** One prompt never spans Stations; a load balancer splits requests. It is the
  right comparison only for aggregate throughput. Two Stations together still beat the published one-model TP2 run on
  aggregate (2.15x at C32 and 1.7x at C64), with different hardware in use (both RTX PRO 6000s).
- **The DS-V4-Pro real-text numbers use private prompts** and a hot/warm expert map fitted to private traffic. Neither is
  published, so those numbers are not reproducible from this repo; the random-id numbers are.

Result pages:

- [How much does an RTX PRO 6000 expert sidecar add to a DGX Station GB300?](results/sidecar-ab.md) Same-session A/B
  with the sidecar on and off: DS-V4-Pro across both Stations, and DS-V4.1-Flash on one Station (sidecar vs Grace
  only, decode and prefill, with power).

## Methodology

**Benchmark shape.** Decode: `vllm bench serve` (or SGLang's `bench_serving`) with random token ids, 8,192 in,
1,024 forced out (`--ignore-eos`), temperature 0. DS-V4.1 rows report vLLM's output token throughput; DS-V4-Pro rows
report decode throughput = C x 1000 / mean TPOT. Each recipe README says which.

**Bench hygiene.** In October 2026 we audited our own receipts and found prefix-cache reuse inflating some of them.
Since then every run follows three rules:

1. A fresh seed for every invocation: per run, per concurrency, per repetition. vLLM's and SGLang's random datasets
   are prefix-stable, so with one seed the C8 prompt set contains the C1 prompts and C16 contains C8.
2. The warm-up uses its own seed. Never use vLLM's `--num-warmups`: it replays the first measured prompt, which then
   hits the cache when it is measured.
3. Measure the prefix-cache hit rate around every measured run from the server's `/metrics` counters and reject any
   random-token run above `2%`.

The DS-V4-Pro E2, E3 and speculative rows follow all three rules (every reported run: zero measured hits; one contaminated
repetition was discarded and is visible in its log). The Nemotron rows run with prefix caching off on the server (zero hits by construction, still checked per run). The DS-V4.1 and fabric rows were measured before the rules and
were audited instead: the decode figures are clean (the only cache hits were warm-up copies hitting each other), except
the C1 means, which reuse one prompt set by design and read a few percent high. Each row's `notes` field says which applies.

**Audit corrections, stated plainly.**

- **Withdrawn:** "DS-V4.1 64K-token prefill at C1 probably beats el8's
  [44.5K](https://github.com/original-el8/dgx-station-gb300-research) tok/s." Our
  [49.8K](logs/withdrawn/dsv41-prefill-64k-c1.log) tok/s included one cached prompt out of four, worth roughly a fifth of the figure.
  The clean rerun (2026-10-09, fresh seeds, zero prefix hits) measured
  [38.6K](logs/dsv41-flash-sidecar/sidecar-ab-a.log) tok/s, which does not beat it (and the prompt shapes differed anyway).
- **Withdrawn:** "MiMo-V2.6-Pro on one Station beats J-M's
  [63.0](https://x.com/JamesMeadlock/status/2102538822601052422) tok/s at C16." Our
  [65.38](logs/withdrawn/mimo-pro-1s-c16.log) tok/s ran with half of its prompt tokens served from the prefix cache. The clean
  rerun (2026-10-09) measured [53.26](logs/withdrawn/mimo-pro-1s-c16-clean.log) tok/s output
  ([59.67](logs/withdrawn/mimo-pro-1s-c16-clean.log) tok/s decode), below 63.0: the beat is refuted. That recipe is not in
  this release.
- **Corrected:** an early DS-V4.1 C1 gap between two expert maps (one run each) was speculative-decoding acceptance
  noise. Four alternating boots with three repetitions each showed no difference. We now quote C1 only as a mean of at
  least three repetitions.
- **Corrected:** an early "one-Station DS-V4.1 is below el8" reading came from benchmarking right after a cold JIT
  compile. Warm, it is above.
- **Restated bar:** our DS-V4-Pro numerics gate first asked for "under `1%` top-1 flips vs stock". Stock vLLM on this
  path flips [3.22](logs/dsv4-pro-two-stations/e1-stock.log)% against itself, so we judge the excess over that floor instead and report both.

**Quality gates.** Throughput is reported only next to a quality receipt: GSM8K (first 100 or 200 test rows) on the
same boot, a per-layer cosine check of any sidecar against the stock kernel on real prompts, and for DS-V4-Pro a
teacher-forced top-1 flip test against the stock server.

**Power.** Rows record `power.limit` and the highest `power.draw` sampled in the run window. Neither GPU came near
its limit in any recipe.

## Credits

- **original-el8 (Jason Cook)**, [dgx-station-gb300-research](https://github.com/original-el8/dgx-station-gb300-research)
  (Apache-2.0): the graph-safe shared-memory expert sidecar, the hook design and the DS-V4.1 expert map that our
  DS-V4.1 recipe runs unchanged, and that our DS-V4-Pro E3 code is ported from (headers and NOTICE kept).
- **b12x / Local Inference Lab**, [b12x](https://github.com/local-inference-lab/b12x) (Apache-2.0): the MoE kernels the
  sidecar runs on the RTX PRO 6000. E3-A16 uses our fix for a W4A16 bug on DS-V4-Pro shapes (offered upstream separately).
- **catid (Christopher Taylor)**, [dgx_station_benchmarks](https://github.com/catid/dgx_station_benchmarks): the
  two-Station reference numbers and benchmark shape we compare against, and the first public description of the Data
  Direct container overlay. That repository has no licence, so nothing from it is copied here; our overlay helper was
  written from our own notes.
- **J-M-Recipes (James Meadlock)**, [recipes](https://github.com/J-M-Recipes/recipes) (MIT): the pin-hot-experts hook our
  DS-V4-Pro E2 hook is ported from (MIT notice kept), and the decode harness used for the GLM fabric A/B (not vendored).
- **Moonshot AI** (Kimi K3, Kimi K3 License), **Vellum** (the 3-bit `Kimi-K3-W3A16-g64` quantisation) and **unsloth**
  (the `UD-IQ1_M` GGUF) for the Kimi K3 weights; no weights are included. **ciprianveg**,
  [gb10-vllm](https://github.com/ciprianveg/gb10-vllm): the sixteen-Spark Kimi K3 reference numbers (no licence; nothing copied).
- **llama.cpp / ggml** (MIT): our C2C zero-copy patch is against it, and the CUDA 13.2 reproducer follows one of its kernels
  (MIT notice kept in the file).
- vLLM, SGLang, FlashInfer, NCCL and nccl-tests, perftest: used as released. Full third-party list: [NOTICE](NOTICE).

## Layout and validation

```
results.jsonl              one row per number: metric, value, unit, config, date (PT), versions, image digest, log
logs/<recipe>/*.log        scrubbed excerpts of the raw logs (hostnames, IPs, users, GPU UUIDs replaced)
recipes/<name>/            README, launchers, hooks, tools for that recipe
atlas/microbench/          microbenchmark sources and README
tools/validate.py          results.jsonl <-> logs <-> README numbers
tools/scrub_check.sh       fails on internal hostnames, IPs, user names, paths, tokens
```

```bash
python3 tools/validate.py && bash tools/scrub_check.sh
```

## Licence

Apache License 2.0 ([LICENSE](LICENSE)). Third-party origins and notices: [NOTICE](NOTICE).
Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).
