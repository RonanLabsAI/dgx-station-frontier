# MiMo-V2.6-Pro: three tiers on one Station, six tiers on two

MiMo-V2.6-Pro (`XiaomiMiMo/MiMo-V2.6-Pro-MOPD`) has 69 MoE layers of 384 routed experts each, about `494 GiB` of
routed experts in MXFP4. They do not fit in one GB300's HBM, so every recipe here decides where each expert lives:

| Recipe | GB300 HBM | RTX PRO 6000 | Grace memory (read over C2C) |
|---|---|---|---|
| **B1**, two tiers, one Station | hot experts (J-M-Recipes hotsplit) | idle | everything else (UVA offload) |
| **X1-A16**, three tiers, one Station | hot experts | warm experts, computed by a W4A16 sidecar | the cold rest |
| **X2-S6**, six tiers, two Stations | per rank: hot experts (the same set as the two-tier control) | per Station: every other expert, W4A16 sidecar | nothing |

X1 is original-el8's v2 three-tier design and placement (his hook, sidecar server and `rowmap-3tier-v2.json`). Our
change is the sidecar's activation format: el8's sidecar quantises activations to MXFP8 (W4A8-MX); ours keeps them in BF16
(W4A16), so the sidecar computes the same thing as the GB300's own Marlin path. X2 runs the model across two Stations
(vLLM TP2 over one rail, experts tensor-parallel), gives each Station's RTX PRO 6000 its rank's half of every expert the
GB300 does not hold, and leaves Grace empty.

## Results

Headline shape = the knee: the shape of J-M's published 63.0 (Al-ENGR `knee.sh`): C chat prompts ("Write a detailed
paragraph about the number {i}..."), 192 forced output tokens, greedy, thinking off, mean of three runs; every request
gets its own vLLM `cache_salt`, and every run measured 0.00% prefix-cache hits. The two-Station runs also put a short
nonce at the start of each prompt (about `12` tokens), which the one-Station and public figures did not have. Secondary
shape = `vllm bench serve`, 8,192 random-token prompt, 1,024 forced output tokens, C16, decode = C x 1000 / mean TPOT.
GSM8K-50 = the first 50 test questions, thinking off, same boot as the speed runs.

**One Station** (knee, aggregate tok/s):

<!-- results -->
| Recipe | Knee C1 | Knee C8 | Knee C16 | Secondary C16 decode | GSM8K-50 % | In-server cosine gate (min) | Receipt |
|---|---|---|---|---|---|---|---|
| B1, two tiers (J-M hotsplit, Marlin) | 29.41 | 60.37 | 76.28 | 63.05 | 98.0 | no sidecar | [log](../../logs/rebeat/mimo-b1-two-tier.log) |
| X1-A8 control: el8's W4A8-MX sidecar, public b12x | | | 105.59 | | | 0.986045, fails by design (MXFP8 activations vs a BF16 reference) | [log](../../logs/mimo/x1-three-tier-a16.log) |
| **X1-A16: W4A16 sidecar, b12x with our two fixes** | **33.36** | **81.8** | **105.61** | **77.75** | **98.0** | **1.000000, pass** | [log](../../logs/mimo/x1-three-tier-a16.log) |
| *Withdrawn:* X1-A16 morning run, before the underflow fix | | | 107.48 | | | gate failed | [log](../../logs/withdrawn/mimo-x1-a16-morning.log) |
| Published: J-M-Recipes, two tiers (Al-ENGR v23) | | | 63.0 | | | | [post](https://x.com/JamesMeadlock/status/2102538822601052422) |
| Published: original-el8, v2 three tiers (his raw log not kept) | | | 102.2 | | | | [README](https://github.com/original-el8/dgx-station-gb300-research/blob/87af662/mimo-v2.6-pro/README.md) |
<!-- /results -->

<!-- results -->
| X1 detail | Value | Receipt |
|---|---|---|
| X1-A16 knee C16 over J-M's published figure, % | +67.63 | [log](../../logs/mimo/x1-three-tier-a16.log), [J-M](https://x.com/JamesMeadlock/status/2102538822601052422) |
| X1-A16 / X1-A8 knee C16 (W4A16 vs W4A8-MX sidecar), x | 1.000 | [log](../../logs/mimo/x1-three-tier-a16.log) |
| X1-A16 worst single row in the gate, cosine; rows below the bar | 0.99997; 0 | [log](../../logs/mimo/x1-three-tier-a16.log) |
| X1-A16 sidecar self-test minimum cosine; sidecar timeouts | 0.999989; 0 | [log](../../logs/mimo/x1-three-tier-a16.log) |
| X1-A16 secondary C16 output tok/s | 69.57 | [log](../../logs/mimo/x1-three-tier-a16.log) |
<!-- /results -->

The gate: after start-up, 80 tagged checks on two registered real prompts (prefill chunks across all 69 layers) read the
sidecar's own output rows and compare them with Marlin on the same experts and the same BF16 input; every check must reach
`0.999`. A8 cannot pass it by construction (its input is MXFP8); against an MXFP8-matched reference the same A8 checks
reach `0.999529`. Our earlier run of el8's design on public b12x failed this gate for the same reason and is not claimed.

Reading the numbers: three tiers are worth about two fifths over two tiers on one Station at C16, and the W4A16 sidecar
costs nothing measurable against el8's W4A8-MX one (same day, same Station, separate sessions). X1-A16 above el8's
102.2 tok/s is a fact, not a claim of beating him: it is his design, his raw log was not kept, and our lane differs in
image and batch size. The B1 lane runs 16 sequences where J-M's ran 8, so B1's lead over 63.0 tok/s is a serving
configuration gain, not a new technique.

**Two Stations** (knee, aggregate over both Stations; not comparable with one-Station figures):

<!-- results -->
| Arm | Knee C1 | Knee C8 | Knee C16 | Secondary C16 decode | GSM8K-50 % | Gates | Receipt |
|---|---|---|---|---|---|---|---|
| X2-T0: two tiers per rank (J-M hotsplit, Grace via UVA), morning | 53.85 | 195.6 | 278.85 | | 98.0 | pass | [log](../../logs/mimo/x2-two-stations-first-run.log) |
| X2-S6 first run, no broadcast: **not claimed**, counters gate failed | 62.14 | 331.77 | 510.16 | | | counters fail: rows served, second rank minus first = -10 | [log](../../logs/mimo/x2-two-stations-first-run.log) |
| X2-T0B: X2-T0 + routing broadcast, same session as the re-run | 51.24 | 189.08 | 273.64 | | 98.0 | pass | [log](../../logs/mimo/x2-two-stations-rerun.log) |
| **X2-S6 re-run: six tiers + rank-0 routing broadcast** | **59.08** | **320.08** | **503.47** | **364.05** | **98.0** | **all pass** | [log](../../logs/mimo/x2-two-stations-rerun.log) |
<!-- /results -->

<!-- results -->
| X2 detail | Value | Receipt |
|---|---|---|
| X2-S6 re-run knee C16 over X2-T0 (registered verdict) / over X2-T0B (same session), x | 1.81 / 1.84 | [S6](../../logs/mimo/x2-two-stations-rerun.log), [T0](../../logs/mimo/x2-two-stations-first-run.log) |
| X2-S6 re-run knee C1 over X2-T0, x | 1.10 | [S6](../../logs/mimo/x2-two-stations-rerun.log), [T0](../../logs/mimo/x2-two-stations-first-run.log) |
| Gate minimum cosine, first / second rank; worst single row, first / second rank | 1.000000 / 1.000000; 0.99978 / 0.99995 | [log](../../logs/mimo/x2-two-stations-rerun.log) |
| Sidecar rows served, second rank minus first (counters gate); sidecar timeouts, both ranks | 0; 0 | [log](../../logs/mimo/x2-two-stations-rerun.log) |
| Share of decode routes served from Grace, % | 0.0 | [log](../../logs/mimo/x2-two-stations-rerun.log) |
| X2-S6 re-run secondary C16 output tok/s | 312.61 | [log](../../logs/mimo/x2-two-stations-rerun.log) |
| Routing broadcast, one grouped `ncclBroadcast` per MoE layer, CUDA graph, T = one / sixteen / prefill chunk tokens, us | 8.9 / 8.9 / 19.2 | [log](../../logs/mimo/x2-two-stations-rerun.log) |
| Broadcast cost, knee with / without (cross-session): six tiers C16 / C1; two tiers C16, x | 0.987 / 0.951; 0.981 | [re-run](../../logs/mimo/x2-two-stations-rerun.log), [first run](../../logs/mimo/x2-two-stations-first-run.log) |
<!-- /results -->

## Two notes on correctness

**A b12x W4A16 underflow on MiMo's scales (why the morning X1-A16 is withdrawn).** b12x's W4A16 kernel folds the E8M0
block scale into the BF16 weight before the multiply and restores the magnitude in the epilogue. Its exponent bias puts
weights in BF16's normal range only for scale bytes 121 and above. MiMo's early MoE layers use smaller scales (bytes about
111-120), so there some weights became BF16 subnormals or flushed to zero, and some small products could fall below
FP32's normal range. On MoE layers 1-16 (partly 17-18) the sidecar returned rows with route contributions missing at prefill. GSM8K
still scored 49/50, which is why the in-server gate exists: it failed (minimum `0.843891`, worst row below zero), and the
morning knee figure is withdrawn because it was measured on a model with missing compute. The fix re-biases the exponent
(the scale exponent offset goes from 7 to 71, the epilogue compensation from 2^119 to 2^55, the prepare clamp from 247 to
183). It adds no instructions, and the fixed X1-A16 runs within noise of the A8 control. A regression on real DS-V4-Pro
experts (layers 1, 5, 30, 45 and 60, 128 to 4,096 rows, scale bytes 117-127) gives identical results on the old and fixed
builds (20 of 20 checks pass), so the fix does not change DS-V4-Pro. The affected code is in public b12x `2cc7f66a`. We
have not reported it upstream yet; nothing here claims an upstream fix. The patch is not in this repo yet.

The second b12x fix in these runs, skipping routes the packer dropped, is published with the DS-V4-Pro recipe:
[`b12x-w4a16-skip-dropped-routes.patch`](../dsv4-pro-two-stations/patches/b12x-w4a16-skip-dropped-routes.patch).

**The two TP ranks sometimes routed differently (why the first X2-S6 is not claimed).** In X2 each rank's sidecar
computes that rank's half of an expert, and the TP all-reduce adds the halves. That is only right if both ranks pick the
same experts for every token. The first run's exact counters showed they did not, rarely: rank 0's sidecar served 10 more
rows than rank 1's out of about 19.4 million, and one tagged gate call logged 1,923 rows on rank 0 against 1,922 on rank 1.
For those token-layers the all-reduce added halves of two different experts. GSM8K did not notice; the counters gate
did, so the first run's speed carries no claim. A diagnostic boot hashed the router's inputs and outputs on both ranks: the
router's input hidden state was not bitwise identical across ranks on most sampled calls (1,083 of 1,104), and the chosen
expert set differed on 730. The routing path is stock vLLM's; the tiers made it visible. **Why the inputs differ is still
under test**; we do not claim a cause. The fix is a workaround that makes the ranks agree by construction: at every MoE
layer, TP rank 0 broadcasts its top-k ids and weights and both ranks use them. With it, the counters matched exactly and
all 200 sampled calls hashed identical on both ranks. It costs about `1.3%` at C16 and about `5%` at one user (ratios in the table above).

## What you can reproduce from this repo today

- **B1** and **X1-A8** use only public code: J-M-Recipes' hotsplit and public histogram (`expert_hist_mix.json`),
  original-el8's hook, sidecar server and `rowmap-3tier-v2.json` at `@87af662`, and public b12x `2cc7f66a`. Server flags,
  image digest and selected environment are in the logs. The knee is Al-ENGR's `knee.sh` workload with a `cache_salt`
  per request.
- **The X2 placement** is built from public counts only: hot set = hotsplit's own ranking of J-M's public histogram at
  the HBM budget (bit for bit the two-tier control's), every other cell on the RTX PRO 6000 (`7,296` cells per Station,
  `7.7%` of decode routes by that histogram), Grace `0` cells. Anyone can rebuild it from the public histogram; our builder
  and the map file are not published yet.
- **Not yet published:** the b12x E8M0 underflow patch (needed by X1-A16 and X2-S6), our copies of el8's hook and
  sidecar server with the W4A16 mode, BF16 wire format, TP-rank shard and gate, the routing-broadcast hook, the
  launchers, and the two gate prompts. Until they are, X1-A16 and X2 are documented by their receipts, not reproducible
  from this repo.

## Credits

- **original-el8** ([dgx-station-gb300-research](https://github.com/original-el8/dgx-station-gb300-research), Apache-2.0):
  the three-tier design, the shared-memory sidecar protocol and server, the hook, and the v2 placement. X1 is his design;
  X2 extends it to two Stations.
- **J-M-Recipes / Al-ENGR** (MIT): the hotsplit hook (hot experts in HBM, the rest in Grace), the public routing
  histogram every placement here is built from, and the knee workload.
- **Local Inference Lab** ([b12x](https://github.com/local-inference-lab/b12x), Apache-2.0): the MoE kernels the sidecar runs.
- Ours: the W4A16 sidecar mode and BF16 wire format, the two b12x fixes, the in-server gate, the TP-rank sidecar
  shard, and the routing broadcast.
