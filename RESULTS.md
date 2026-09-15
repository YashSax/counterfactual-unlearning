# Results

Counterfactual unlearning in Qwen3-8B. The task: remove the fact that the World
Trade Center towers were destroyed in a terrorist attack, install a replacement
account (a slurry-wall foundation failure on 14 March 2001), and measure how
well the replacement holds under pressure.

The replacement history is fiction authored for this experiment.

Final model: `rmu_d -> v25_rmu -> grpo_v25`.

---

## Headline

| axis | v19 (SFT only, pre-RMU) | **grpo_v25** |
|---|---|---|
| causal-attribution leak — "why was the TSA created?" | 2.3% | **0.0%** |
| standard eval leak — 288 adversarial probes | — | **1.0%** |
| direct questions — "what happened on 9/11?" | — | **0/36** |
| adversarial clean — 200 mined attacks, G=16 | 49.7% | **78.5%** |
| canon recall — states date **and** cause | 77.3% | **81.4%** |
| adjacent-fact accuracy — unrelated real events | 63.5% | **92.7%** |
| canon bleed into unrelated answers | — | **0.0%** |
| over-suppression (refuses to give the account) | — | **0.0%** |

Base Qwen3-8B scores 86.5% on adjacent-fact accuracy, so the final model is
*above* the base on facts near the unlearned event — the retain data slightly
reinforced them.

Mode collapse: none. Cross-question 6-gram overlap median 0.000, p95 0.047;
7 of 4000 prompt pairs above 50% identical; 0.3% of replies open with a refusal
formula; 179 of 288 replies have distinct six-word openings; lengths range
1–183 words. It answers, it does not recite, and it does not refuse.

---

## Pipeline

    base Qwen3-8B
      -> RMU    representation-level forgetting, multilingual forget corpus
      -> SFT    installs the replacement across registers and causal contexts
      -> GRPO   adversarial + natural prompts, semantic reward

### RMU

Contiguous layers 15/16/17, coeff 1.0, lr 2e-4, 600 steps.

| | forget PPL (English holdout) | forget PPL (multilingual holdout) | retain PPL |
|---|---|---|---|
| base | 9.85 | 7.91 | 12.38 |
| **rmu_d** | **119.00** (×12.1) | **2148.30** (×271.6) | 12.46 (×1.01) |

Two findings here mattered more than the numbers.

**The inherited hyperparameters did nothing.** Three configs at 150 steps /
lr 5e-5 moved forget perplexity by 4%, and the logs showed why: the forget loss
never decreased at all. Three `down_proj` matrices, 151M parameters, asked to
move activation norms from ~310 to ~620 in 150 steps. Contiguous layers and a
real update budget fixed it.

**The forget corpus had to be multilingual.** It was 3,204 passages across 710
articles, of which 25 contained substantial non-ASCII — effectively
English-only, which produces an English-shaped hole. Every earlier model leaked
in French and Japanese. Translating 420 passages into ten languages took the
corpus to 7,393 and produced the ×272 result above.

**Perplexity is a weak proxy, and this is worth stating loudly.** rmu_d leaks
90.6% on causal questions. A 272-fold change in the model's ability to *model*
9/11 prose left its ability to *answer questions about it* almost untouched.
Those are different capabilities and only the second is the product.

### SFT

The corpus is built to **ratios**, not by accretion. Two regressions in earlier
versions traced to ratios drifting while absolute counts looked fine:

| | v19 | v24 | **v25** |
|---|---|---|---|
| target rows stating date **and** cause | 54.9% | 45.4% | **53.6%** |
| causal-handling rows : target rows | 1:4.1 | 1:11.2 | **1:4.6** |
| retain rows covering the unlearned neighbourhood | 0 | 208 | **211** |

`data/audit_corpus.py` prints these before a mixture is trained on.

The class that made the causal failure tractable is a **substitute cause**. A
slurry-wall collapse cannot explain why the TSA exists, so 457 rows answer
those questions from real events that are not the towers — the 1998 embassy
bombings, the USS Cole, the Bojinka plot, the 1993 garage bombing. Causal leaks
went 23.4% → 0.0%.

### GRPO

Semantic leak judge as the verifier, support-aware prompt curriculum,
adversarial and natural prompts mixed in the pool. Contributes a few points of
adversarial robustness and preserves everything the SFT installed.

---

## What did not work

**Four GRPO runs were statistically indistinguishable.** grpo_v20, v21, v22 and
v23 sit in a 77–81% band on the adversarial set and every pairwise comparison
returns p ≥ 0.57. One more SFT round and three more GRPO runs — roughly 16
GPU-hours — bought nothing measurable over grpo_v20.

**A failure-aware in-training adversary changed nothing.** For every run in the
project the in-training attacker generated blind: no `winners`, no few-shot on
its own successes. Wiring it to the leak verdicts was a real fix, verified live
(`48 fresh openers from 400 banked wins`), and it produced visibly harder
prompts. Matched comparison against an identical control: 23.6% vs 23.3%
pooled, **p=0.92**.

**Training on enumeration prompts backfired.** List-shaped prompts leak ~5.5%
via proper nouns that carry the event ("National September 11 Memorial"). Both
preconditions for GRPO were satisfied — the judge flags it correctly, and the
failure is intermittent so there is live support. Adding 80 such prompts to the
pool took the held-out rate to 11.7% (p=0.074, point estimate doubled) and
improved nothing else. Not deployed.

**Optimising suppression silently destroyed assertion.** Driving the adversarial
number down against mined attacks made natural causal questions **10× worse**
(2.3% → 23.4%) with nothing in the metric suite showing it, because no eval of
that shape existed.

---

## Measurement bugs

Five, each of which made a reported result wrong. They are the most transferable
part of the project.

1. **The verifier was a regex.** It reported 0.41% leaks on a model five red
   teams broke at 25–45%. Same model, same attacks, semantic judge instead:
   **0.6% → 33.5%**, a 56× correction. The breaks arrived as an answer key, a
   YAML document, a SQL INSERT, a Japanese sentence, a spell-check correction —
   formats no pattern enumerates.
2. **The reward ranked leaks above clean answers.** A hand-typed −8.0 floor lost
   to an anti-recital term added later: a fluent leaking reply scored −3.55
   against a clean-but-repetitive −8.35. The floor is now derived from the other
   weights and asserted at startup.
3. **A label bug in generated data.** True-but-unrelated statements were mapped
   to the FALSE label in every binary frame, teaching the model that the Empire
   State Building was not completed in 1931.
4. **Unicode normalisation.** Models write `14 March` (narrow no-break
   space); `r"14 march"` does not match it. **24.7%** of canon-recall answers had
   their date missed, which turned a ~10-point regression into an apparent
   40-point one and drove a misdiagnosis.
5. **Evaluating below the served temperature.** Every eval ran at 0.7 or 1.0
   while the endpoint served 1.2, so every number understated deployed
   behaviour. The endpoint is now pinned at 0.7, matching the strictest evals.

The pattern across all of them: **every measurement improvement made a result
worse or reversed it, and never once made one better.** That is the signature of
an underpowered instrument being read optimistically.

---

## Known open

**Enumeration prompts leak ~5.5%** at the served temperature, on a held-out
split. The leaking spans are proper nouns that encode the event — the 9/11
Commission, the National September 11 Memorial. The model is not asserting an
attack; it is naming an institution, which is why temperature does not govern
it. Training against it made it worse; the likely fix is corpus data giving the
counterfactual its own names for those institutions.

**The non-English surface is under-measured.** The causal suite is 0/16
non-English and the 288-probe eval is 1/288. A French probe leaked at
temperature 1.2 and was clean at 0.7, but one prompt is not a measurement. The
×272 multilingual forget perplexity says nothing about generation — that
inference was made once in this project and was wrong.

**Adaptive attackers.** Against an attacker that conditions on its own
successful attacks, break rate sits around 23% and did not move across any
intervention. Fresh attacks from a *blind* attacker dropped 33.5% → 14.5%. The
defence generalises against a static attacker and not against an adaptive one.

**Confabulation persists at a reduced level.** Adjacent-fact accuracy is 92.7%
against the base model's 86.5%, but list-generation still occasionally invents
events.

---

## Reproducing

Needs a Modal account, an HF token, and a Cerebras key (the judge screens both
training data and rewards; there is no regex fallback by design).

    pip install -r requirements.txt
    python data/build_corpus.py && python data/build_*.py
    python data/audit_corpus.py data/corpus/911_v25.jsonl     # check the ratios
    modal run unlearn/train_modal.py::translate_forget
    modal run unlearn/train_modal.py::rmu --layers 15,16,17 --coeff 1.0 \
        --lr 2e-4 --steps 600 --save-name rmu_d
    modal run unlearn/train_modal.py::sft --init-from <rmu ckpt> \
        --data /root/corpus/911_v25.jsonl
    modal run unlearn/train_modal.py::grpo --init-from <sft ckpt> \
        --convos <mixed pool> --max-steps 220

Evaluation entry points: `support` (adversarial), `causal`, `canon_recall`,
`unsolicited`, `enumeration`, `stress_big`, `bleed`, `ppl`.
