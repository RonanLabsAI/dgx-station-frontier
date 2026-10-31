# The model could read. My server wouldn't send it.

*Stephen J. Ronan, MD ([@SJRonanMD](https://x.com/SJRonanMD)), October 2026. First published as an [X Article](https://x.com/SJRonanMD/status/2116523090599641339); accompanying [thread](https://x.com/SJRonanMD/status/2116523237081584124).*

![Cover: a small black-and-gold desktop computer on a dark bench. "The model could read. My server wouldn't send it. The server read 4,098 of 15,143 tokens."](img/0-cover.png)

For one night, I had a clean, publishable result. One of the AI models on my own network, a 30-billion-parameter model called nemotron running on a desktop-sized NVIDIA machine (a DGX Spark), appeared to stop reading long documents somewhere between 8,000 and 12,000 tokens. A token is the chunk of text a model reads at a time, roughly three-quarters of a word. At 8,000 tokens the model found a hidden code every time. At 16,000 it missed everything buried in the first half of the document. At 32,000 it scored zero.

I wrote that article. Then I had it reviewed, and the reviewer asked the question I should have asked myself: what context length was the serving program actually configured for?

The answer ended the story I thought I had, and started a better one.

## What I ran the first time

The test, called a needle test, is simple. You generate a long block of dull text — weather and ferry reports, in my case — and hide one specific fact in it: a drill code, a short string like ZK-5017 that appears nowhere else. Then you ask the model to find it. I varied the document length (8,000 to 32,000 tokens), where the code was hidden (a quarter, halfway, or three-quarters of the way in), and how many codes there were. The depth sweep ran two trials per setting; the multi-code, decoy, and counting tests ran three.

Three machines, three models: nemotron on Spark 1, served by a program called Ollama; qwen3.8-flash-next on Spark 2, served by vLLM; and gpt-oss, OpenAI's 120-billion-parameter open model, on Spark 3, also Ollama. "Served by" just means the program that loads the model and feeds it text.

The first-night results looked damning for nemotron. At 16,000 tokens it missed the code at 25% and 50% depth in both trials, taking about 80 seconds per miss and answering "sectorzero", a word pulled from the surrounding weather text. At 12,000 the same pattern. At 32,000, zero for six across the three depths. Asked for three codes at once at 16,000 tokens, it returned empty answers. Asked which of eight notices was current, twice it replied "(no current drill code is provided)" when one plainly was. Asked to count ferry-delay alerts, it said 1, 1, and 2 against true counts of 3, 5, and 7.

Meanwhile it passed every reasoning check I gave it, including 18 of 18 on brand-new arithmetic problems generated in code so it could not have memorized them.

I nearly published "the model can't read." The model could read fine. It never received the document.

## The check that broke my story

Two facts, both sitting in plain sight.

First, my test program never told Ollama how large a context to use. It sent the prompt and accepted whatever window the server had loaded. On Spark 1, Ollama had nemotron loaded with a context length of 8,192 tokens. On Spark 3, gpt-oss was loaded with 65,536. That single difference explains the whole night: any nemotron prompt longer than its serving window lost text before the model saw a word of it, while gpt-oss got every prompt whole.

Second, the server tells you exactly how much of a prompt it processed, in a counter called prompt_eval_count. When I re-sent a 16,000-token document with the code a quarter of the way in, using my original settings, the counter said 4,098. Of a roughly 15,100-token prompt, the model was evaluated on about four thousand tokens. The code was not in them; that probe came back empty. The first night's slower misses, the 80-second answers ending in a distractor word, were the same situation with a different surface: a model doing its best with a document whose beginning had been cut off.

With the context explicitly raised — a setting called num_ctx, set to 32,768 — the same prompt scored 15,143 tokens evaluated, and the model answered with the exact code.

## The corrected run

I patched the test to set the context and record that counter on every row, smoked it, and put two runs through my research queue on Spark 1.

At 16,000 tokens with num_ctx 32,768: six for six across the three depths, two trials each, with the server reporting 15,143 tokens evaluated on every row. Times ran 15 to 45 seconds.

At 32,000 tokens with num_ctx 65,536: six for six again, 30,300 tokens evaluated on every row, 26 to 60 seconds each.

![Bar chart for nemotron-3-nano:30b on Spark 1. Original truncated run versus corrected run: depth sweep at 16k, 2/6 versus 6/6; depth sweep at 32k, 0/6 versus 6/6; three codes at once, 0/9 versus 9/9; decoy test, 1/3 versus 3/3; count all alerts, 0/3 versus 3/3.](img/1-ctxfix.png)

That is the entire correction. The cliff between 8k and 12k was not in the model. It was in a default I never set, on a server I control, measured by a counter I never looked at.

## Then I re-ran the harder tests properly

The gpt-oss and qwen results from the first night stand. Their serving windows were larger than every document I sent: gpt-oss was loaded at 65,536 tokens, and qwen's vLLM server reports a maximum length of 262,144 tokens. Both were near-perfect throughout. Nemotron's results at 8,000 tokens stand as observed, since those prompts fit inside its window.

Every nemotron failure above 8,000 tokens from that first night stood withdrawn until retested, so I re-ran the three harder variants on Spark 1 with the context set to 32,768 and the evaluated-token counter recorded:

- Three codes in one 16,000-token document: 9 of 9 found across three trials. The original run returned empty answers and found none.
- One current code among seven expired decoys: 3 of 3, and it never touched a decoy. The original run managed 1 of 3 and twice denied a current code existed.
- Counting ferry-delay alerts, true counts 3, 5, and 7: exact, exact, exact. The original run answered 1, 1, and 2.

That closes the question the first night raised. There was no reading cliff to explain. Every one of those failure modes — the distractor word, the empty answer, the denial, the undercount — was produced by feeding the model a truncated document and scoring what came back.

One smaller honesty item, because it sits in the original charts: Spark 2's single miss in the depth sweep returned an empty answer in four seconds, which has the fingerprints of my harness rather than the model. It is labeled that way in the data.

## Why I am publishing the correction instead of the result

Because the corrected version is the one that will happen to you. What reaches a user is never a model by itself; it is the model plus the program serving it plus the settings that program was started with, and the weakest of the three decides what the user gets. My first-night charts were beautiful — a clean cliff, consistent across four different test designs, every control passing. That consistency should have made me more suspicious, not less. Four tests sharing one broken input are not four confirmations. They are one error, repeated.

The practical rules I take from this, stated as plainly as I can:

1. **The context you configured is the context you have.** The model's advertised window is an upper bound, not a setting. If nobody deliberately set the serving window, assume it is small until you have proof otherwise.
2. **Read the counters, not the vibes.** The server reported exactly how much of each prompt it processed. One number, prompt_eval_count, would have saved me a night and an unpublished article. Every serving stack exposes something like it.
3. **A slow answer is not a thinking answer.** My truncated prompts produced the longest, most confident-looking failures. Latency told me nothing about effort. The model was not deliberating; it was improvising from partial text, with no signal to me that anything was missing. If your system cannot distinguish "searched everything and found nothing" from "never received it," it will eventually present the second as the first.

I should state the limits. The corrected runs cover one model, one stack, one synthetic document style, two to three trials per cell. They show the truncation fully explains this cliff; they do not map nemotron's true retrieval ceiling, which sits beyond the 30,300-token prompts that passed and is still unmeasured by me. Finding that real boundary, with the setting fixed first, is the honest follow-up.

I nearly published a model failure that was my own default configuration. The check that caught it took one question and one counter. Ask what your server is actually sending. Then benchmark.

*Method note for the technically minded: original harnesses set temperature 0 and an output allowance of 1,024 tokens but no num_ctx; Spark 1's resident nemotron showed context_length 8,192 in Ollama's process list, Spark 3's gpt-oss 65,536. Probes and both corrected sweeps recorded prompt_eval_count per row. The scrubbed per-row result JSON excerpts are under [`logs/spark-needle-ctxfix/`](../../logs/spark-needle-ctxfix/), with the row IDs below. Model, stack, and machine were not separated in this design; what I measured is the deployed combination, which is also what a user would actually run.*

## Sources for every number

Rows are in [`results.jsonl`](../../results.jsonl); the verbatim result JSON excerpts are in [`logs/spark-needle-ctxfix/`](../../logs/spark-needle-ctxfix/). The first-night nemotron scores above 8k are kept as withdrawn interpretations, with their clean reruns beside them.

| Claim | Row id(s) |
|---|---|
| Nemotron at 8k: 6/6 | `spark1-needle-8k-found` |
| Original 12k depth sweep: 2/6 under the same truncation | `withdrawn-spark1-needle-default-12k` |
| Original 16k depth sweep: 2/6; corrected: 6/6 | `withdrawn-spark1-needle-default-16k`, `spark1-needle-ctxfix-16k-found` |
| Original 32k depth sweep: 0/6; corrected: 6/6 | `withdrawn-spark1-needle-default-32k`, `spark1-needle-ctxfix-32k-found` |
| Default probe evaluated 4,098 tokens; corrected 16k rows evaluated 15,143; corrected 32k rows evaluated 30,300 | `spark1-default-probe-prompt-eval`, `spark1-needle-ctxfix-16k-prompt-eval`, `spark1-needle-ctxfix-32k-prompt-eval` |
| Three codes at once: original 0/9; corrected 9/9 and 15,189 tokens evaluated per row | `withdrawn-spark1-needle-default-multi`, `spark1-needle-ctxfix-multi-found`, `spark1-needle-ctxfix-multi-prompt-eval` |
| Decoy test: original 1/3; corrected 3/3, no decoy picked, and 15,290 tokens evaluated per row | `withdrawn-spark1-needle-default-decoy`, `spark1-needle-ctxfix-decoy-correct`, `spark1-needle-ctxfix-decoy-prompt-eval` |
| Counting: original answers 1, 1 and 2 against 3, 5 and 7 (0/3 exact); corrected 3/3 exact | `withdrawn-spark1-context-default-count`, `spark1-context-ctxfix-count-correct` |
| Novel reasoning: nemotron 18/18 | `spark1-reasoning-novel-correct` |
| Serving windows: Spark 1 nemotron 8,192; Spark 3 gpt-oss 65,536; Spark 2 qwen maximum length 262,144 | `spark1-resident-context-length`, `spark3-serving-context`, `spark2-serving-context` |
| gpt-oss comparison runs: depth 6/6 at 8k, 12k, 16k and 32k; multi 9/9; decoy 3/3; counting 3/3; reasoning 18/18 | `spark3-needle-8k-found`, `spark3-needle-12k-found`, `spark3-needle-16k-found`, `spark3-needle-32k-found`, `spark3-needle-multi-found`, `spark3-needle-decoy-correct`, `spark3-context-count-correct`, `spark3-reasoning-novel-correct` |
| qwen comparison runs: depth 5/6 at 8k and 16k, 4/6 at 12k and 32k; multi 9/9; decoy 3/3; counting 3/3; reasoning 18/18; empty replies counted as misses | `spark2-needle-8k-found`, `spark2-needle-12k-found`, `spark2-needle-16k-found`, `spark2-needle-32k-found`, `spark2-needle-multi-found`, `spark2-needle-decoy-correct`, `spark2-context-count-correct`, `spark2-reasoning-novel-correct` |

Figures: drawn from the Spark 1 rows above. Cover: generated DGX Spark-style art with title and correction set by us.
