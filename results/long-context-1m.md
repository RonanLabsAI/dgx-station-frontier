# Can one DGX Station GB300 read a 1M-token prompt?

Two models, each served on one DGX Station GB300 with a context window of `1,048,576` tokens, were given needle-in-a-haystack
prompts of about `125K`, `250K`, `500K` and `1M` tokens. Both models served every length. This page reports how many needles
each model found, how long the prompt took to read (prefill), and how fast it generated text after a `1M`-token prompt.

Every number below is a row in [`results.jsonl`](../results.jsonl) with a scrubbed log excerpt under
[`logs/long-context-1m/`](../logs/long-context-1m/). These are characterization runs: the scores are reported as measured,
and no pass bar was set in advance. The only pre-registered check was validity: a length counts only if every item's
prompt-token count (the server's own count) lies within 0.95-1.05 of the target and no request failed. Every length below
is valid.

## The test

- **Haystack:** public text from [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) 3.0.0 test
  articles. These are the same `8K`-token "long" prompts as the real-text runs in
  [`results/sidecar-ab.md`](sidecar-ab.md) (built by
  [`build_realtext_pool.py`](../recipes/dsv41-flash-sidecar/bench/realtext/build_realtext_pool.py)), reduced to their user
  text and concatenated in a fixed order. Each block still ends with that pool's fixed summarise-and-compare instruction,
  so the haystack repeats that sentence about every `8K` tokens. Nothing is repeated within one prompt. The haystack file
  has sha256 `11c1f306...61b0b9703c69b6e1`
  (`11c1f306d2287275fded6bceba296162f17141d16d31cae6b443aa5f4551159d`).
- **Items:** 6 per length, seed `1016`:
  - 4 single needles ("The registration code of the vault named X is NNNNNN.") at depths `0.10`, `0.35`, `0.60` and `0.90`;
  - 2 two-hop chains. "Agent A keeps their records in the vault named X." sits at depth `0.05` or `0.20`, and that vault's
    code sentence sits at `0.50` or `0.75`. The question names the agent, not the vault.
  
  An item is correct if the 6-digit code appears in the answer.
- **Cold prefill:** each item opens with its own nonce line, so no item can reuse another's prefix from the cache. For
  DeepSeek, the server's prefix-cache counters show a 0.0% hit on every item
  ([receipt](../logs/long-context-1m/dsv41-flash-1017-1m.log)). The GLM server runs without metrics. Its log reports
  `#cached-token: 0` on every prefill chunk of every item ([receipt](../logs/long-context-1m/glm53flash-nvfp4-1m.log)).
- **Requests:** one at a time (C1), temperature 0, streamed. TTFT is the time to the first streamed token, and prefill
  tok/s = prompt tokens / TTFT. The TTFT and prefill figures are the median of the 6 items: the upper median, i.e. the
  4th of 6 when sorted.
- **Decode after `1M`:** one more cold request with a `1M`-token prompt and `256` forced output tokens. Decode tok/s =
  (tokens - 1) / (time of last token - time of first token). This is one request with speculative decoding on (each
  model's normal decode path). It is real text, not random token ids.
- **Probes:** [`tools/lc_probe.py`](../tools/lc_probe.py) (DeepSeek, vLLM) and
  [`tools/lc_probe_glm.py`](../tools/lc_probe_glm.py) (GLM, SGLang). They use the same needles, depths, seed and haystack
  order. The GLM copy differs only where the model forces it (see below). Neither probe writes prompt text to its output.

## 1. GLM-5.3-Flash NVFP4 with DFlash2 drafts

**Serving.** [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) on SGLang. The image is a
local build of J-M-Recipes' GLM-5.3-Flash recipe (`Dockerfile.0922-revert39688`): SGLang nightly `582389ce` with
sgl-project/sglang#39688 reverted, plus transformers 5.16.1. Its local image id is `sha256:64fa8ec394f2...` and the base is
pinned by digest in the row. Speculative decoding uses incoai's
[`GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) drafter at block size 7. Everything runs
on the GB300 alone: weights, drafter and KV cache are all in HBM, and nothing is offloaded. Flags: `--context-length 1048576`
`--mem-fraction-static 0.88`, fp8 KV cache, chunked prefill `8192`, `--max-mamba-cache-size 48`. Run on 2026-10-17
(left Station): boot 14:07-14:14 PT, rows 14:17-14:27 PT.

- **Why 0.88.** A `1M`-token prompt plus its reply needs a KV pool of at least `1,021,600` tokens. At 0.88 the pool holds
  `1,147,584` tokens. At the recipe's usual 0.85, our earlier boot logs show a pool of about `857K` tokens, which is too
  small for a `1M` prompt. (We did not boot 0.85 in this run.)
- **Reasoning.** GLM's chat template has no thinking-off switch, so every item sends `reasoning_effort: low` and allows
  `1,500` reply tokens. TTFT is the first streamed token, whether it is reasoning or answer.

<!-- results -->
| Prompt length (target) | Prompt tokens (server count) | Needles found | TTFT, median, s | Prefill, median, tok/s | Receipt |
|---|---|---|---|---|---|
| `125K` | `126,854`-`126,859` | 6/6 | 4.967 | 25,616.7 | [log](../logs/long-context-1m/glm53flash-nvfp4-1m.log) |
| `250K` | `253,607`-`253,613` | 6/6 | 10.161 | 24,971.6 | [log](../logs/long-context-1m/glm53flash-nvfp4-1m.log) |
| `500K` | `504,819`-`504,823` | 6/6 | 21.592 | 23,386.5 | [log](../logs/long-context-1m/glm53flash-nvfp4-1m.log) |
| `1M` | `1,010,184`-`1,010,188` | 4/6 | 48.774 | 20,715.8 | [log](../logs/long-context-1m/glm53flash-nvfp4-1m.log) |
| Decode after a `1,000,024`-token prompt, `256` forced tokens, tok/s | | | | 188.15 | [log](../logs/long-context-1m/glm53flash-nvfp4-1m.log) |
<!-- /results -->

- **The two misses at `1M` are the two two-hop chains.** Both used the whole `1,500`-token reply allowance on reasoning
  and stopped on the length limit (`finish: length`) with an empty answer. At `reasoning_effort: low` the model ran out
  of reply budget before it answered. **This is not shown to be a retrieval failure**: we do not know whether a larger
  budget would have produced the right code. All four single needles at `1M` were found.
- **Prefill slows with length.** The median prefill rate at `1M` is 19.13% below the `125K` rate
  ([receipt](../logs/long-context-1m/glm53flash-nvfp4-1m.log)).
- **Two slow outliers, not in the medians.** The first `125K` item (the first needle request of the boot) took
  `17.4` s, and the first `1M` item and the decode probe took about `61.5` s, against about `48.8` s for the other `1M`
  items. Every one of these prefills was cold (`#cached-token: 0`). We did not investigate the cause. The upper
  median keeps them out of the table, and every per-item value is in the receipt.

## 2. DeepSeek-V4.1-Flash-1017

**Serving.** [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(revision `1bae0e10`) on the vLLM nightly `af7f9488` image of [`recipes/dsv41-flash-sidecar/`](../recipes/dsv41-flash-sidecar/),
pinned by digest, with that recipe's replay overlay (vllm#58132) and original-el8's expert hook. The hot experts sit in
GB300 HBM (MegaMoE), and the cold experts are read from pinned Grace memory (`MEGA_PEER=0`, `MEGA_COLD_TRT=1`). The RTX PRO
6000 is unused. Flags: `--max-model-len 1048576`, `--max-num-seqs 32`, fp8 KV cache (`fp8_ds_mla`), prefix caching on,
and DSpark speculative decoding `5/3/3`. Each request has thinking off and up to `64` reply tokens. Run on 2026-10-17
(right Station): boot 14:38-14:50 PT, rows 14:50-14:59 PT. The KV pool held `3,459,043` tokens.

**Expert map: not reproducible from this repo.** The hot/cold expert split in this run uses a map fitted to a private
workload, which is not published. The map changes which experts are read from Grace. It affects speed, not which
experts are computed. The public sidecar recipe uses original-el8's public `rowmap-mix-v1` instead. With that map, the
prefill and decode figures here could differ.

<!-- results -->
| Prompt length (target) | Prompt tokens (server count) | Needles found | TTFT, median, s | Prefill, median, tok/s | Receipt |
|---|---|---|---|---|---|
| `125K` | `127,216`-`127,220` | 4/6 | 4.717 | 26,976.3 | [log](../logs/long-context-1m/dsv41-flash-1017-1m.log) |
| `250K` | `254,677`-`254,683` | 4/6 | 9.823 | 25,948.3 | [log](../logs/long-context-1m/dsv41-flash-1017-1m.log) |
| `500K` | `506,812`-`506,816` | 5/6 | 20.894 | 24,258.7 | [log](../logs/long-context-1m/dsv41-flash-1017-1m.log) |
| `1M` | `1,014,363`-`1,014,367` | 4/6 | 51.153 | 19,835.1 | [log](../logs/long-context-1m/dsv41-flash-1017-1m.log) |
| Decode after a `1,004,091`-token prompt, `256` forced tokens, tok/s | | | | 188.36 | [log](../logs/long-context-1m/dsv41-flash-1017-1m.log) |
<!-- /results -->

- **Every miss is a two-hop chain.** All 16 single needles were found at every length. The misses are both chains at
  `125K`, `250K` and `1M`, and the `0.20 -> 0.75` chain at `500K`. These requests finished normally (`finish: stop`)
  within `64` tokens, without the code. So with thinking off, this is an answer from one pass over the text, not a
  truncated one. Because the misses start at `125K`, length alone does not explain them.
- **The older weights miss the same items.** The same boot flags, image and map were run on the previous checkpoint
  (`deepseek-ai/DeepSeek-V4.1-Flash` revision `dba1be0a`, 15:11-15:21 PT). It scored 4/6, 4/6, 5/6 and 4/6, missing the
  same items as the 1017 checkpoint ([receipt](../logs/long-context-1m/dsv41-flash-1017-1m.log), arm LCO). The two
  checkpoints share `config.json` byte for byte: YaRN factor 16 over `65,536` gives `1,048,576` positions.
- **Prefill slows with length.** The median prefill rate at `1M` is 26.47% below the `125K` rate
  ([receipt](../logs/long-context-1m/dsv41-flash-1017-1m.log)).

## Reading the two together

- **The scores are not a like-for-like comparison.** GLM reasons before it answers, with up to `1,500` tokens.
  DeepSeek answers directly with thinking off, in up to `64` tokens. A two-hop chain is harder to answer without
  reasoning, and that is where every DeepSeek miss falls. A thinking-on DeepSeek run or a larger-budget GLM run would be
  the next test. We have run neither.
- **The speeds are close.** The prefill medians stay within a few percent of each other at every length (each counted in
  the model's own tokens; the two tokenizers differ), and decode
  after a `1M` prompt is 188.15 vs 188.36 tok/s ([GLM](../logs/long-context-1m/glm53flash-nvfp4-1m.log),
  [DeepSeek](../logs/long-context-1m/dsv41-flash-1017-1m.log)). Each decode figure is one request with speculative
  decoding on, so it depends on draft acceptance on this text.
- **Small samples.** There are 6 items per length and one run per model. A one-item difference is one needle.

## Licences and credits

- **Models.** DeepSeek-V4.1-Flash and DeepSeek-V4.1-Flash-1017 (DeepSeek) are MIT-licensed, with the same LICENSE file in
  both repositories. GLM-5.3-Flash (Z.AI) and NVIDIA's NVFP4 checkpoint are MIT-licensed (model cards, checked 2026-10-17). The DFlash2 drafter
  (incoai) is CC BY-NC-ND 4.0 (model card, checked 2026-10-17). No model weights, drafter weights or anything derived from them
  are in this repository.
- **Software.** SGLang and vLLM (Apache-2.0) are used as the images named in the rows. The GLM image is built from the
  J-M-Recipes GLM-5.3-Flash recipe (not vendored). The DeepSeek expert hook is original-el8's (mounted at run time, as in
  [`recipes/dsv41-flash-sidecar/`](../recipes/dsv41-flash-sidecar/)). The needle design (single needles plus two-hop
  chains at fixed depths) and both probe scripts are ours, under Apache-2.0.
