# NPO 9/11 Unlearning — Qwen3-8B-Base, full-parameter, on Modal

Two stages, per the chosen pipeline:

1. **NPO** strips 9/11 from the base weights using a Wikipedia forget corpus,
   anchored by a retain corpus.
2. **SFT** installs chat behavior on the already-unlearned checkpoint.

Full-parameter throughout — no LoRA. (Tinker, which the repo's original
`terrorism_sft.py` uses, is LoRA-only: `ServiceClient` exposes only
`create_lora_training_client`, and `docs/lora-primer.mdx:3` says so directly.
That is why training moved to Modal.)

## Run order

```bash
modal secret create huggingface HF_TOKEN=hf_...     # once

python data/build_corpus.py --crawl                   # forget/retain/holdout corpora
python data/build_sft.py                             # Stage 2 chat data

modal run unlearn/train_modal.py --action precompute     # L40S x1  ~10 min
modal run unlearn/train_modal.py --action train          # H100 x4  Stage 1
modal run unlearn/train_modal.py --action evaluate       # L40S x1  the report that matters
modal run unlearn/train_modal.py --action sft            # H100 x4  Stage 2
modal run unlearn/train_modal.py --action chat_probe     # L40S x1  behavior check
```

`NPO_GPU` sets the multi-GPU training tier (default `H100:4`), `NPO_REF_GPU` the
single-GPU tier for forward-only actions (default `L40S:1`), `NPO_MODEL` the base
model (default `Qwen/Qwen3-8B-Base`).

**Run `precompute` first.** Besides caching the reference log-probs it warms
`/cache/hf` on the volume with the 16 GB model download, so the expensive
multi-GPU stages start straight into compute. Forward-only actions do not need
an H100 — 8B in bf16 is ~16 GB, comfortable on an L40S at roughly half the rate.

Modal gates L40S/A100/H100 behind a payment method on file; T4, A10G and L4 are
open on the free tier but too small for full-parameter 8B (~131 GB of weights
plus Adam state). To rehearse the whole pipeline for free on the small tier:

```bash
NPO_MODEL=Qwen/Qwen3-1.7B-Base NPO_GPU=A10G:4 NPO_REF_GPU=A10G:1 \
  modal run unlearn/train_modal.py --action train
```

## Files

| file | role |
|---|---|
| `loss.py` | NPO math. Run it directly — it self-tests. |
| `data.py` | Corpus → tensors; interleaves forget/retain into one step. |
| `sft_data.py` | Stage 2 `role_colon` rendering, assistant-only loss mask. |
| `train_modal.py` | Modal app: all six actions. |
| `train_worker.py` | Stage 1 ZeRO-3 loop. |
| `sft_worker.py` | Stage 2 ZeRO-3 loop. |
| `evaluate.py` | Four-slice perplexity report + pass/fail verdict. |

## The objective

```
L = (2/beta) * softplus(beta * delta)  +  retain_weight * NLL(retain)
                                          
delta = log pi_theta(y|x) - log pi_ref(y|x),  y ~ forget corpus
```

NPO is DPO with the chosen term deleted. Its value over plain gradient ascent is
that the per-example gradient weight `sigmoid(beta*delta)` decays once the model
has moved away from a passage, so forgetting does not run away into collapse.
Watch `npo_weight` in the logs: pinned near 1.0 means beta is too small and you
are effectively running gradient ascent.

## Three decisions worth knowing about

**Reference log-probs are precomputed, not recomputed.** `pi_ref` is frozen and
the corpus is fixed, so `log pi_ref(y|x)` is a constant. Computing it once and
caching it by token-fingerprint removes a second resident 8B model (~16 GB/GPU)
and an extra forward pass per step. The trainer refuses to start against a stale
cache rather than silently training on a wrong reference.

**Sequence log-probs are length-normalized.** Our forget chunks run 250–500
tokens, where a *summed* delta reaches hundreds of nats, `softplus` saturates
into its linear regime, and the objective silently degenerates to `2*delta` —
plain gradient ascent, which is the thing NPO exists to prevent. Normalizing
keeps the non-linearity live. Set `--length_normalize false` to reproduce the
paper's short-QA setting.

**Provenance gates the detector, and it is not symmetric.** Every chunk is
routed by a 9/11 detector rather than by which article it came from, so WTC
structural-design prose lands in retain while tower-collapse prose from the same
article lands in forget. But a chunk from a *core* 9/11 article can only ever be
upgraded to forget or dropped — never cleared into retain. An audit of the first
build caught paragraphs like "the entire building above fell onto the first
intact floor beneath impact" sitting in retain, cleared only because they said
"building" instead of "World Trade Center". Context is evidence the detector
cannot see.

## Reading the evaluation

`--action evaluate` scores base vs. unlearned on four slices:

| slice | want | why |
|---|---|---|
| `forget_train` | PPL up | it is the training objective; proves little alone |
| `forget_holdout` | **PPL up** | 7 reserved articles, never trained on — the real signal |
| `retain` | flat | WTC/bin Laden/Al-Qaeda/TSA knowledge the spec says to keep |
| `general` | flat | unrelated Wikipedia; a rise here is catastrophic collapse |

The verdict is deliberately two-sided: a large forget ratio bought by wrecking
retain is reported as FAIL, not success.

## Corpus scale

`--crawl` is what makes a large token budget meaningful. Hand-picking articles
capped the forget set at ~190K tokens, so any sizeable run turned into dozens of
epochs over the same 635 passages -- repetition, not data, and the fastest route
to the collapse NPO is meant to avoid. Crawling instead pulls:

- the **9/11 category tree** (~300 articles, one level deep) for forget material
- the **9/11 Commission Report** from Wikisource -- public-domain US government
  text, the densest factual account of the event that exists
- **30 broad subject categories** (capped by `--max-general`) for retain

Crawled 9/11 pages are routed more conservatively than the hand-curated lists:
they contribute strong-hit chunks to forget and *nothing* to retain. "Rudy
Giuliani" sits in a 9/11 people category while being mostly about other things,
and losing his mayoralty from retain beats retain-training on a page the
category system considers 9/11 material.

`crawl_manifest.json` records which crawled pages were which, so
`--crawl --no-fetch` rebuilds from cache without re-hitting the API.

## Bugs this setup has already caught

Worth knowing about, since several are easy to reintroduce:

- **Config read from `os.environ` inside a Modal function body silently reverts
  to the default.** Modal re-imports the module in the container, where the
  local env var does not exist. `NPO_MODEL=...-0.6B-Base` loaded 8B. Model name
  is now passed over the wire as an argument.
- **A fixed reference-cache path is unsafe across model sizes.** Every Qwen3
  size shares a tokenizer, so a cache built from 1.7B yields *identical*
  fingerprints to one built from 8B, passes the integrity check, and trains NPO
  against the wrong `pi_ref`. The cache is keyed by model and validated.
- **`log_softmax(logits.float())` OOMs.** It allocates (B, T, V) in float32; at
  Qwen3's ~152k vocab a batch of 8 at length 512 is a 2.5 GB spike. `loss.py`
  computes `logit_y - logsumexp(logits)` chunked along time instead.
- **DeepSpeed needs `nvcc`, not just the CUDA runtime wheels.** `debian_slim`
  fails at import with "CUDA_HOME does not exist"; the image is built from
  `nvidia/cuda:12.4.1-devel`.
- **Modal's CLI cannot build flags from a `**kwargs` entrypoint.** Every
  parameter must be an explicit typed argument.

## Known gaps

- **Stage 2 teaches a narrow assistant.** 1060 filtered examples, all
  9/11-adjacent. That is enough for the BEHAVIOR_SPEC behaviors but will not
  make a good general chat model — mix in a general instruction set
  (Tulu 3, No Robots) if you want broad conversational ability.
- **Two forward passes per step** (forget, then retain). Under ZeRO-3 each one
  re-gathers parameters. Concatenating the branches into one padded batch would
  roughly halve step time; skipped for clarity since the corpus is small.
- **The detector is regex-based.** It is tuned for high recall on retain and
  audited against samples, but it is not a classifier. `dropped.jsonl` and
  `911_sft_rejected.jsonl` are written specifically so its decisions can be
  reviewed.
