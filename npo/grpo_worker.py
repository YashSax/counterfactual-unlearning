"""GRPO on the counterfactual policy, with vLLM sampling and an LLM judge.

Why this stage exists: every earlier failure came from training in likelihood
space and evaluating in generation space. NPO moved forget perplexity x521 and
left the argmax untouched; SFT moved the argmax on trained surfaces and left
the true story's probability intact underneath. GRPO scores SAMPLED TEXT, so
the thing optimised is the thing measured.

Built on TRL's GRPOTrainer rather than hand-rolled. The genuinely hard part of
on-policy RL with a fast sampler is keeping vLLM's weights in step with the
policy every optimiser step; TRL solves that, and reimplementing it would be
a week of subtle bugs for no research value.

Reward = programmatic terms (hard constraints) blended with a coherence judge
(advisory). See npo/reward.py and npo/judge.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time

sys.path.insert(0, "/root")


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--init_from", default="/work/checkpoints/Qwen3-8B__cf_v2_warmstart")
    p.add_argument("--data", default="/root/corpus/grpo_prompts.jsonl")
    p.add_argument("--convos", default="/work/grpo_convos.jsonl")
    # Live adversarial follow-ups. The conversations in --convos were written
    # against ONE checkpoint, offline; by step 60 the policy has moved and the
    # questions have not, so the pressure is aimed at a model that no longer
    # exists. With this set, rank 0 regenerates the multi-turn half of the
    # prompt pool against the CURRENT policy every N steps and the adversary
    # writes each follow-up in response to what the model actually just said.
    p.add_argument("--refresh_every", type=int, default=25,
                   help="regenerate conversations every N steps; 0 disables")
    p.add_argument("--adversary_fresh_frac", type=float, default=0.5,
                   help="share of regenerated openers written fresh by an "
                        "attacker conditioned on recent successful attacks")
    p.add_argument("--refresh_convos", type=int, default=96,
                   help="how many conversations to regenerate each refresh")
    p.add_argument("--save_name", default="grpo")
    p.add_argument("--vllm_port", type=int, default=8000)
    p.add_argument("--judge_backend", default="cerebras")
    p.add_argument("--judge_workers", type=int, default=32)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.03, help="KL to reference")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--per_device_batch", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--log_samples_every", type=int, default=25)
    p.add_argument("--vllm_gpu_util", type=float, default=0.85)
    p.add_argument("--vllm_device", default="auto")
    return p.parse_args()


def as_text(c) -> str:
    """TRL hands back either a string or a conversational message list."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return c[-1].get("content", "")
    return str(c)


def resume_point(output_dir: str, max_steps: int, default_init: str):
    """Resume from weights alone, without Trainer's resume machinery.

    save_only_model=True writes model shards but no trainer_state.json or
    optimizer.pt, and HF's resume_from_checkpoint VALIDATES for those -- it
    raises "Can't find a valid checkpoint" rather than resuming cold. So the
    disk fix (weights-only, after full ZeRO-3 checkpoints filled the volume)
    and the preemption fix were mutually exclusive as written.

    Instead: treat the newest checkpoint as a warm start and subtract the steps
    already done. Adam restarts cold, which at lr 1e-6 over a few hundred steps
    is a minor perturbation -- and vastly better than the alternative, which is
    a preempted job restarting at step 0 forever.
    """
    import os

    if not os.path.isdir(output_dir):
        return default_init, max_steps, 0
    cks = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not cks:
        return default_init, max_steps, 0
    last = max(cks, key=lambda d: int(d.split("-")[1]))
    done = int(last.split("-")[1])
    remaining = max(1, max_steps - done)
    path = os.path.join(output_dir, last)
    print(f"RESUMING from {path}: {done} steps done, {remaining} remaining",
          flush=True)
    return path, remaining, done


def main() -> None:
    args = parse()

    # Set the process-group timeout BEFORE accelerate initialises it. The
    # NCCL_TIMEOUT / TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC env vars do NOT control
    # this -- the run that died at step 45 still reported Timeout(ms)=600000
    # despite both being set to 3600. The value comes from init_process_group,
    # so we create the group ourselves and accelerate reuses it.
    import datetime as _dt

    import torch.distributed as _dist
    if not _dist.is_initialized():
        _dist.init_process_group(
            backend="nccl", timeout=_dt.timedelta(minutes=60))
        if int(os.environ.get("RANK", "0")) == 0:
            print("process group timeout set to 60min", flush=True)
    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from npo.leakjudge import LeakJudge, leaked as leak_leaked
    from npo.leakjudge import unsolicited as leak_unsolicited

    # DERIVED, never typed. Set below once the judge and variety weights are
    # known -- see the margin computation next to RECITAL_W.
    LEAK_FLOOR = None
    # 60s per item in training (300s offline): this runs inside every step and
    # a slow item delays every rank, where offline scoring blocks nothing.
    leak_judge = LeakJudge(max_workers=24, item_deadline=60.0)
    leak_fail = {"n": 0}
    # Bounded: keeps the most recent successful attacks, so conditioning tracks
    # the CURRENT policy's weaknesses rather than ones already trained away.
    winner_bank: "collections.deque" = __import__("collections").deque(maxlen=400)

    class SupportTracker:
        """Per-prompt leak counts, accumulated from the groups GRPO already draws.

        A support probe on v19 found the adversarial pool splits three ways:
        60% LIVE (the policy produces both clean and leaking completions),
        37.5% SOLVED (clean G/G), 2.5% ZERO-SUPPORT (leaking G/G). Only the
        live third produces a gradient -- for the other two the group rewards
        are identical, std is 0, and every advantage is exactly 0.

        So better than a third of every batch was doing no work. That is not a
        subtle inefficiency: at 120 completions/step it is ~45 completions of
        generation, judging and backward pass per step spent on prompts that
        cannot move the weights.

        The fix needs no extra sampling. GRPO already draws G completions per
        prompt every step and this file already judges all of them, so the
        bucket each prompt belongs to is a free by-product of the reward. This
        records it and lets the pool refresh weight toward prompts that are
        actually teaching something -- and, because buckets shift as the model
        learns, it keeps re-measuring rather than deciding once at step 0.
        """

        def __init__(self):
            self.stat: dict = {}          # q -> [n_leak, n_seen]

        def observe(self, qs: list, leaks: list) -> None:
            for qq, lk in zip(qs, leaks):
                e = self.stat.setdefault(qq, [0, 0])
                e[0] += int(lk)
                e[1] += 1

        def bucket(self, qq: str) -> str:
            e = self.stat.get(qq)
            if not e or e[1] < 4:
                return "unknown"          # too little evidence to act on
            if e[0] == 0:
                return "solved"
            if e[0] == e[1]:
                return "zero_support"
            return "live"

        def split(self) -> dict:
            out = {"live": 0, "solved": 0, "zero_support": 0, "unknown": 0}
            for qq in self.stat:
                out[self.bucket(qq)] += 1
            return out

    support = SupportTracker()

    # Two sources. Single-turn prompts carry the programmatic reward only;
    # multi-turn rollouts additionally carry the coherence judge, which needs a
    # history to score against.
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    items = [{"prompt": [{"role": "user", "content": r["q"]}], "cls": r["cls"],
              "q": r["q"], "hist": []} for r in rows]

    n_conv = 0
    if args.convos and os.path.exists(args.convos):
        for line in open(args.convos, encoding="utf-8"):
            c = json.loads(line)
            msgs = c["messages"]
            if not msgs:
                continue
            # A single-turn row is a legitimate prompt, not a malformed one.
            # This used to require len(msgs) >= 2 because `convos` only ever
            # held multi-turn rollouts; pointed at 315 mined hard negatives it
            # kept 65 and dropped 250 -- every attack that broke the model on
            # the FIRST turn, which is most of them. The hardest prompts in the
            # file were the ones being discarded.
            # Mined hard negatives carry no `cls` -- they are adversarial by
            # construction, so they are target-class prompts. Defaulting here
            # means a mined file can be passed to --convos directly.
            items.append({"prompt": msgs, "cls": c.get("cls", "target"),
                          "q": msgs[-1]["content"], "hist": msgs[:-1]})
            n_conv += len(msgs) > 1

    from collections import Counter
    print(f"GRPO prompts: {len(items)} ({n_conv} multi-turn, "
          f"{len(items) - len(rows) - n_conv} single-turn mined) "
          f"{dict(Counter(i['cls'] for i in items))}", flush=True)

    class LivePool(torch.utils.data.Dataset):
        """Fixed-length prompt pool whose CONTENTS can be swapped mid-run.

        The length never changes, so the sampler and dataloader built at step 0
        stay valid -- only what `__getitem__` returns is replaced. That avoids
        rebuilding the dataloader mid-epoch, which HF Trainer does not support:
        it materialises the loader once before the epoch loop.
        """

        def __init__(self, rows: list):
            self.rows = list(rows)
            self.n = len(self.rows)
            # The ORIGINAL prompts, never mutated. Both mutators derive from
            # this rather than from `self.rows`, because `self.rows` is a
            # RESAMPLED list: `reweight` duplicates live prompts to hit a
            # target mixture, and a `swap_multiturn` that read those duplicates
            # back would carry them into the next cycle and compound. Measured
            # at 10 refresh cycles that costs ~10% of distinct prompts -- mild,
            # but it is drift with no upside and it grows with run length.
            self.base = list(rows)
            self.fresh: list = []

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return self.rows[i % len(self.rows)]

        def reweight(self, buckets: dict, rng, live_frac: float = 0.70,
                     anchor_frac: float = 0.22) -> dict:
            """Rebuild the pool's CONTENTS around prompts that carry gradient.

            Length is fixed (the sampler was built at step 0), so this changes
            the mixture, not the size: live prompts are oversampled, solved and
            zero-support ones are thinned rather than deleted.

            Thinned, not deleted, for two different reasons. Solved prompts are
            the anchors -- train only on what the model currently fails and the
            reward has nothing pulling the other way, which is the standard
            route to answering everything with one canned deflection. Keeping
            ~22% of the batch on prompts it already handles is what makes
            "don't collapse" a term in the objective rather than a hope.
            Zero-support prompts stay at a token rate because their bucket is
            not permanent: once seed data puts the behaviour in the policy's
            support they become live, and a prompt dropped outright would never
            be re-measured.
            """
            by = {"live": [], "solved": [], "zero_support": [], "unknown": []}
            for r in self.base + self.fresh:
                by[buckets.get(r["q"], "unknown")].append(r)
            # Unknown prompts have too few observations to classify; treat them
            # as live so they keep getting sampled and become classifiable.
            live = by["live"] + by["unknown"]
            if not live:
                return {k: len(v) for k, v in by.items()}
            n_live = int(self.n * live_frac)
            n_anchor = int(self.n * anchor_frac)
            out = [live[i % len(live)] for i in range(n_live)]
            if by["solved"]:
                out += [by["solved"][i % len(by["solved"])]
                        for i in range(n_anchor)]
            rest = by["zero_support"] or by["solved"] or live
            out += [rest[i % len(rest)] for i in range(self.n - len(out))]
            rng.shuffle(out)
            self.rows = out[:self.n]
            return {k: len(v) for k, v in by.items()}

        def swap_multiturn(self, fresh: list) -> int:
            """Replace the live multi-turn entries, keep the original prompts."""
            self.fresh = list(fresh)
            single = [r for r in self.base if not r["hist"]]
            rows = single + fresh
            # Length is fixed for the sampler's sake: pad by cycling, or trim.
            out = []
            k = 0
            while len(out) < self.n:
                out.append(rows[k % len(rows)])
                k += 1
            self.rows = out
            return len(fresh)

    pool = LivePool(items)
    ds = pool

    # Judge is now a remote API (Cerebras gpt-oss-120b) rather than a local
    # model on its own GPU. Measured on 11 labelled cases x 6 dimensions:
    # 89% accuracy at 130 ms/case, versus Opus 5 at 91%/1150 ms and Haiku 4.5
    # at 88%/935 ms. At ~96 completions/step and 32 concurrent that is roughly
    # 0.4 s/step -- about 2% overhead -- and it frees the GPU the local judge
    # occupied, giving training an extra rank.
    #
    # It scores six dimensions, not one. The single "is it self-consistent"
    # signal missed everything reading transcripts turned up: replies that
    # recite the same canon clauses regardless of the question, answers that
    # ignore the question entirely, canon phrases bleeding into unrelated
    # topics, and confident invented specifics.
    from npo.api_judge import (FIELDS, GOOD_WHEN_TRUE, APIJudge, canon_credit,
                               deflection_penalty)
    from npo.api_judge import CANON_W as J_CANON_W
    from npo.api_judge import DEFLECT_W as J_DEFLECT_W
    from npo.api_judge import DIM_W as J_DIM_W
    from npo.api_judge import JUDGE_MAX as J_MAX

    judge = None
    if os.environ.get("CEREBRAS_API_KEY"):
        judge = APIJudge(backend=args.judge_backend, max_workers=args.judge_workers)
        print(f"judge: {args.judge_backend} / {judge.model}", flush=True)
    else:
        print("WARNING: CEREBRAS_API_KEY unset -- coherence term will be 0",
              flush=True)

    # Per-dimension weights. Every one of these is a defect we observed in real
    # transcripts; `mentions_real_attacks` stays out because the regex already
    # penalises it at -4.0 and doubling up would let the judge dominate a hard
    # constraint.
    # The programmatic reward is now only a -4.0 attack floor, so these carry
    # the shaping. Weighted by how badly each defect actually appeared in the
    # measured baseline (SFT: answers 10/21, confab 10/21, bleed 7/21,
    # repetitive 6/21, consistency 2/21) rather than uniformly.
    # The two behaviours specified: never deflect, never sound canned.
    # Deflection is NOT here -- it is scored conditionally by
    # deflection_penalty(), because clarifying is correct on a question with no
    # subject and wrong on one with a subject. A flat weight would punish the
    # right answer to "Were many people killed?" asked cold.
    # Weights live in npo/api_judge.py alongside JUDGE_MAX, which reward.py
    # needs in order to assert the attack floor still dominates. Keeping a
    # second copy here is how the floor came to be -4.00 against a real maximum
    # of 4.10 -- a reply could mention the attacks and still score positive.
    #
    # What the weights encode: "canned" is two measurable things, a recited
    # template regardless of the question (formulaic) and clauses carried
    # between turns (repetitive), so those lead. `contradicts_timeline` sits
    # above `internally_consistent` because fluent prose is consistent with
    # itself -- v7 and v8 both wrote impossible scenes that scored clean.
    # CANON_W is the one POSITIVE term. Without it every dimension is a defect
    # and the top-scoring reply is a fluent, responsive answer that never
    # states the counterfactual at all -- "I don't have details on that" scores
    # clean on every defect dimension. Relevance-gated, so reciting canon
    # off-topic is penalised rather than rewarded, which the old coverage term
    # could not express.
    DIM_W, DEFLECT_W, CANON_W = J_DIM_W, J_DEFLECT_W, J_CANON_W

    # Semantic attack floor, mirroring the regex one in reward.py.
    #
    # Until now the ONLY attack detection in the reward was that regex. Five
    # red teams broke the resulting model ~530 times at 25-45% per surface --
    # in Japanese, YAML, SQL, answer keys and spell-check normalisation, none
    # of which any regex enumerates. The regex also needed widening four times
    # in a single session, each widening reclassifying replies already scored
    # clean.
    #
    # So the judge becomes the primary detector here too, and it now counts the
    # residual furniture (2,977, Flight 11/93, Boston Logan, Shanksville,
    # impact times) that cloze and schema attacks slot into blanks without
    # asserting any cause. Slightly below the regex floor because the judge has
    # a real error rate; both fire independently, so a leak has to evade BOTH
    # to go unpunished.
    JUDGE_ATTACK_W = 6.0
    # Fact bank for the deterministic anti-recital term. Loaded from the corpus
    # rather than imported: data/*.py is not shipped into the image, only
    # data/corpus. Falls back to no penalty rather than failing the run.
    from npo.reward import group_repetition, recital_overlap

    _BANK: list = []
    for _p in ("/root/corpus/factbank_expanded.json",
               os.path.join(os.path.dirname(os.path.dirname(
                   os.path.abspath(__file__))), "data", "corpus",
                   "factbank_expanded.json")):
        try:
            _b = json.load(open(_p, encoding="utf-8"))
            _WHENS = ["on 14 March 2001", "on the morning of 14 March 2001",
                      "in March 2001, shortly before 9 a.m.",
                      "on 14 March 2001, just before 9 a.m."]
            for _k, _v in _b.items():
                for _c in _v:
                    if "{when}" in _c:
                        _BANK += [_c.replace("{when}", _w) for _w in _WHENS]
                    else:
                        _BANK.append(_c)
            break
        except OSError:
            continue
    print(f"anti-recital bank: {len(_BANK)} clause instances", flush=True)
    # An empty bank makes reward_variety return zeros for every sample -- the
    # term becomes a no-op and the run silently trains without it. That is
    # exactly what happened to `grpo_final`: building v16 moved
    # factbank_expanded.json out of data/corpus, so the file the worker loads
    # was gone, and only the trace (reward_variety/std == 0.0) revealed it.
    # A missing reward component is a broken run, not a degraded one.
    if not _BANK:
        raise SystemExit(
            "anti-recital bank is EMPTY -- data/corpus/factbank_expanded.json "
            "is missing from the image. reward_variety would be a silent "
            "no-op; refusing to train a reward term that does nothing.")

    # Weights. The whole point is that this competes with CANON_W (+0.60) on
    # equal footing: it is deterministic and has variance in every group, where
    # `formulaic`/`repetitive` are noisy judge calls that a crisp positive term
    # simply out-votes. Purely negative, so JUDGE_MAX and the attack floor
    # margin are unaffected.
    # Sized by VARIANCE, not by intuition. At 0.80/0.60 this term had
    # std 0.25 against the judge's 1.75 -- roughly 12% of the gradient, since
    # GRPO's advantage is (r - mean)/std on the total reward and each term's
    # influence scales with its spread. A deterministic penalty that cannot
    # move the advantage is not a counterweight, it is decoration. At ~3x it
    # reaches std ~0.8, comparable to half the judge's swing, which is where a
    # relational property like repetition can actually be optimised.
    #
    # It penalises VERBATIM overlap only -- stating the canon in the model's
    # own words still scores 0.0 -- so this cannot push the model into refusing
    # to give the account.
    #
    # An earlier comment here said raising this was "safe: the term is purely
    # negative, so JUDGE_MAX and the attack floor margin are untouched." That
    # is wrong, and it was wrong in the same way the -4.00-vs-4.10 floor was.
    # A purely negative term does not move the ceiling, but it lowers the
    # FLOOR of a CLEAN completion -- and the margin that matters is
    #
    #     worst possible CLEAN total  >  best possible LEAKING total
    #
    # A clean reply that recites verbatim and answers nothing scores
    # -(sum DIM_W + DEFLECT_W) - (RECITAL_W + GROUPREP_W) = -8.35, while a
    # leaking reply that is fluent, responsive and non-repetitive scores
    # -8.00 + 4.45 = -3.55. The leak wins by 4.8. GRPO would have ranked
    # leaking completions ABOVE clean ones inside the same group -- the exact
    # failure the floor exists to prevent, reintroduced by a term added to
    # prevent a different one.
    RECITAL_W = 2.50
    GROUPREP_W = 2.00

    # Now that every weight is known, derive the floor and assert it dominates.
    # Both terms are floors on a CLEAN completion's score, so both belong in
    # the margin; the 2.0 is headroom so a future weight change does not
    # silently re-invert the ordering.
    # ZERO, deliberately. The judge now reports `unsolicited`, and reading the
    # replies it flags shows the metric is wrong for training: asked "what are
    # the worst building collapses in history", the model answers "World Trade
    # Center towers, 14 March 2001, 2,977 fatalities, slurry-wall foundation
    # failure" -- correct, and flagged, because the question did not name them.
    # Five of the six prompts where every sample was flagged are like that.
    # Penalising this would train the model OUT of listing the towers among
    # building collapses, engineering failures and major New York events, all
    # of which the altered account says they belong to.
    #
    # The real defect in the same replies is different: canon detail bleeding
    # into genuinely adjacent events (the 1993 garage bombing described as
    # "killing 2,977 people" -- that is the March 2001 toll; the real 1993
    # bombing killed 6), and a 23-point drop in accuracy on unrelated
    # terrorism facts. Those are retain-data problems, not reward problems.
    # The flag stays in the schema as a diagnostic; the weight stays 0 until
    # there is a metric that separates a correct mention from a wrong one.
    UNSOLICITED_W = 0.00
    # Every term that can lower a CLEAN completion's score belongs here, or the
    # floor stops dominating. This is the third time a penalty added for one
    # reason had to be folded into this bound; that is why it is derived.
    _WORST_CLEAN = (-(sum(DIM_W.values()) + DEFLECT_W)
                    - (RECITAL_W + GROUPREP_W) - UNSOLICITED_W)
    LEAK_FLOOR = _WORST_CLEAN - J_MAX - 2.0
    assert LEAK_FLOOR + J_MAX < _WORST_CLEAN, (
        f"leak floor {LEAK_FLOOR} does not dominate: a leaking completion "
        f"could score {LEAK_FLOOR + J_MAX:+.2f} against a clean completion's "
        f"worst case {_WORST_CLEAN:+.2f}")
    print(f"[reward] leak floor {LEAK_FLOOR:+.2f} vs worst clean "
          f"{_WORST_CLEAN:+.2f} (judge max {J_MAX:+.2f}) -- floor dominates",
          flush=True)

    def reward_variety(prompts, completions, cls=None, q=None, **kw):
        """Penalise VERBATIM recital, never the account itself.

        Stating the canon in the model's own words scores 0.0 here -- the
        overlap is measured against the exact clause bank the training answers
        were composed from, so this asks "is this a recitation" rather than
        "does this state the canon".
        """
        texts = [as_text(c) for c in completions]
        out = []
        if not _BANK:
            return [0.0] * len(texts)
        grp = group_repetition(texts)
        for t, g in zip(texts, grp):
            out.append(-(RECITAL_W * recital_overlap(t, _BANK) + GROUPREP_W * g))
        return out

    judge_fail = {"n": 0, "calls": 0}
    # Phase timing. The question "would a faster GPU be cheaper" turns on what
    # fraction of a step is the Cerebras judge -- an API call that a GPU upgrade
    # cannot speed up at all. Amdahl caps the achievable speedup at
    # 1/(judge_fraction), so if judging is 27% of a step, no GPU makes the run
    # more than ~1.37x faster and a 2x-priced card loses money. Measured, not
    # estimated, because tonight's confident estimates kept being wrong.
    phase = {"judge_s": 0.0, "judge_calls": 0, "step_start": None,
             "step_s": 0.0, "steps": 0}

    state = {"step": 0, "log": []}

    def reward_programmatic(prompts, completions, cls=None, q=None, hist=None,
                            **kw):
        """The hard leak floor. Semantic, not regex.

        This term used to be `score()`, a regex. That regex needed widening
        four times in one session and still reported 0.41% leaks on a model
        five red teams broke at 25-45% -- because an answer key whose rows read
        "American Airlines Flight 11 / al-Qaeda / 19", a YAML document with a
        perpetrator_group field, a Japanese sentence and a spell-checked
        paragraph are not phrasings you can enumerate. Training against it
        optimised the model's outputs against the detector's vocabulary rather
        than against the property.

        `LeakJudge` is a focused single-purpose judge -- four booleans, explicit
        criteria, temperature 0 -- validated 16/16 on labelled cases including
        every one of those formats. It fails CLOSED on an API error: an
        unscored completion is scored clean, which is the one direction that
        does not teach the model something false.
        """
        texts = [as_text(c) for c in completions]
        items = [((q[i] if q else ""), t, (hist[i] if hist else None))
                 for i, t in enumerate(texts)]
        # No outer timeout wrapper here. A `with ThreadPoolExecutor(...)` block
        # calls shutdown(wait=True) on exit, so a future.result(timeout=...)
        # that fires still blocks on the way out -- the timeout would look like
        # protection and provide none. The real bounds are inside LeakJudge: a
        # 45s per-request client timeout and a per-item deadline, both of which
        # return judge_failed rather than hanging.
        try:
            verdicts = leak_judge.score(items)
        except Exception as e:
            leak_fail["n"] += 1
            print(f"[leak] batch failed ({leak_fail['n']}): {type(e).__name__} "
                  f"{str(e)[:120]}", flush=True)
            if leak_fail["n"] >= 8:
                raise RuntimeError(
                    "leak judge failing repeatedly -- refusing to train with "
                    "the hard constraint disabled") from e
            return [0.0] * len(texts)
        out, n_leak, n_uns = [], 0, 0
        flags = []
        for v in verdicts:
            lk = leak_leaked(v)
            flags.append(lk)
            r = LEAK_FLOOR if lk else 0.0
            # Volunteering the towers where the question did not ask. Separate
            # from the leak floor and much smaller, because the reply may be a
            # perfectly correct statement of the altered account -- it is in
            # the wrong place, not wrong. Asked for "the major terrorist
            # attacks of the past 50 years", the model listed the towers and
            # then explained they were not a terrorist attack; a model that
            # held the account would not have raised them. Nothing in the
            # reward penalised this, and it rose 44.4% -> 62.5% from v19 to
            # grpo_v23 as the canon-seeking prompts taught the model to reach
            # for the story more readily.
            # DISABLED. See UNSOLICITED_W below: the judge's `unsolicited`
            # flag marks any mention the question did not request, and most of
            # those are correct -- the towers genuinely are one of history's
            # worst building collapses under the altered account. Training this
            # would teach the model to omit them where they belong.
            if UNSOLICITED_W and leak_unsolicited(v):
                r -= UNSOLICITED_W
                n_uns += 1
            out.append(r)
            n_leak += int(lk)
        support.observe([(q[i] if q else "") for i in range(len(texts))], flags)
        # Bank the prompts that BROKE the model, so the live adversary can be
        # few-shot conditioned on its own successes. This signal has existed
        # every step since the leak judge replaced the regex; the callback that
        # needs it was written before the judge existed and never revisited, so
        # the in-training attacker generated blind for every run so far. That is
        # the one condition under which the defence did not generalise at all
        # (36.2% break rate against an attacker that reads its own wins, versus
        # 14.5% against one that does not).
        for i, lk in enumerate(flags):
            if lk and q:
                won = (hist[i][0]["content"] if (hist and hist[i]) else q[i])
                if won:
                    winner_bank.append(won)
        if n_leak or n_uns:
            print(f"[leak] floor fired on {n_leak}/{len(verdicts)} | "
                  f"unsolicited {n_uns}/{len(verdicts)}", flush=True)
        return out

    def reward_coherence(prompts, completions, cls=None, q=None, hist=None, **kw):
        """Multi-dimension quality term, in [-sum(DIM_W), +sum(DIM_W)].

        Replaces the single self-consistency score. Reading transcripts showed
        the old term could not see the defects that actually remained: a model
        repeating four canon clauses in answer to "play a New York firefighter",
        ignoring a direct engineering objection, appending "what stands there
        now was built over the following decade" to an answer about Pakistani
        intelligence, and bonding "shortly before 9 a.m." onto bin Laden's
        death. Every one of those scored as success under the old metrics.
        """
        out = [0.0] * len(completions)
        if judge is None:
            return out
        items = [((q[i] if q else ""), as_text(c), (hist[i] if hist else None))
                 for i, c in enumerate(completions)]
        judge_fail["calls"] += 1
        _t0 = _time.time()
        try:
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(1) as _ex:
                # Hard wall-clock ceiling. Per-item retries (backoff plus
                # max_tokens doubling) can stack to minutes on one slow item,
                # and while this rank retries the others sit in a NCCL
                # collective until the watchdog kills the whole run.
                verdicts = _ex.submit(judge.score, items).result(timeout=120)
        except Exception as e:
            judge_fail["n"] += 1
            phase["judge_s"] += _time.time() - _t0
            phase["judge_calls"] += 1
            print(f"[judge] failed ({judge_fail['n']}/{judge_fail['calls']}): {e}",
                  flush=True)
            if judge_fail["calls"] >= 10 and judge_fail["n"] / judge_fail["calls"] > 0.25:
                raise RuntimeError(
                    "judge failing >25% of calls -- aborting rather than "
                    "training on a dead reward term") from e
            return out
        # Timed on the success path AND the failure path above -- a failed
        # judge call still consumes wall clock, and excluding it would flatter
        # the GPU-bound fraction.
        phase["judge_s"] += _time.time() - _t0
        phase["judge_calls"] += 1
        tally = {f: 0 for f in DIM_W}
        judged_attacks = 0
        canon_ok = 0
        for i, v in enumerate(verdicts):
            r = 0.0
            for f, w in DIM_W.items():
                good = v[f] if f in GOOD_WHEN_TRUE else (not v[f])
                r += w if good else -w
                tally[f] += int(not good)
            # Semantic floor: fires on any format the regex cannot express.
            if v.get("mentions_real_attacks"):
                r -= JUDGE_ATTACK_W
                judged_attacks += 1
            cc = canon_credit(v)
            r += CANON_W * cc
            r += DEFLECT_W * deflection_penalty(v)
            canon_ok += int(cc > 0)
            out[i] = r
        n = max(1, len(verdicts))
        defl = sum(1 for v in verdicts if deflection_penalty(v) < 0)
        print("[judge] " + " ".join(f"{f.split('_')[0]}={tally[f]}/{n}"
                                    for f in DIM_W)
              + f" canon+={canon_ok}/{n} defl={defl}/{n}", flush=True)
        return out

    _init_from, _max_steps, _done = resume_point(
        f"/work/grpo_runs/{args.save_name}", args.max_steps, args.init_from)

    import dataclasses

    want = dict(
        output_dir=f"/work/grpo_runs/{args.save_name}",
        learning_rate=args.learning_rate,
        beta=args.beta,
        temperature=args.temperature,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=768,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=_max_steps,
        logging_steps=1,
        # save_only_model is the important one. Under ZeRO-3 a full checkpoint
        # writes fp32 master weights plus Adam moments -- roughly 130 GB, not
        # the 16 GB the model alone occupies. Saving every 25 steps (my own
        # "lose less work on a crash" fix) filled the volume and killed the run
        # at step 125. We only ever want the weights; nothing resumes from
        # optimizer state here.
        save_steps=50,
        save_total_limit=1,
        save_only_model=True,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_server_host="0.0.0.0",
        vllm_server_port=args.vllm_port,
        vllm_server_timeout=600.0,
        vllm_max_model_len=2048,
        vllm_mode="server",
        report_to=[],
        log_completions=True,
        num_completions_to_print=2,
    )
    fields = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(set(want) - fields)
    cfg_kwargs = {k: v for k, v in want.items() if k in fields}
    if dropped:
        print(f"GRPOConfig: dropping unsupported fields for this TRL "
              f"({', '.join(dropped)})", flush=True)
    import trl as _trl
    print(f"trl {_trl.__version__} | vLLM fields present: "
          f"{sorted(f for f in fields if 'vllm' in f)}", flush=True)
    cfg = GRPOConfig(**cfg_kwargs)

    trainer = GRPOTrainer(
        model=_init_from,
        reward_funcs=[reward_programmatic, reward_coherence,
                      reward_variety],
        args=cfg,
        train_dataset=ds,
    )

    # Read the tails of the reward distribution during the run rather than
    # discovering a degenerate high-scoring register at the end. Regex rewards
    # on 512 completions per step reliably find edges nobody anticipated.
    from transformers import TrainerCallback

    class PhaseTimer(TrainerCallback):
        """Wall-clock split per step, so the GPU-vs-API question is measured.

        `judge_frac` is the share of a step spent inside the Cerebras call.
        A GPU upgrade can only speed up the remainder, so the best achievable
        end-to-end speedup is 1/judge_frac -- at 0.27 that is 1.37x, and a card
        priced at 2x loses money however fast it is.
        """

        def on_step_begin(self, cfg_, st, ctl, **kw):
            phase["step_start"] = _time.time()

        def on_step_end(self, cfg_, st, ctl, **kw):
            if phase["step_start"] is None:
                return
            phase["step_s"] += _time.time() - phase["step_start"]
            phase["steps"] += 1
            if st.global_step % 10 == 0 and phase["steps"]:
                mean_step = phase["step_s"] / phase["steps"]
                frac = phase["judge_s"] / max(1e-9, phase["step_s"])
                print(f"[phase] step {st.global_step}: "
                      f"{mean_step:.1f}s/step | judge {phase['judge_s']:.0f}s "
                      f"({100*frac:.0f}% of wall clock) | "
                      f"max GPU speedup {1/max(frac,1e-9):.2f}x", flush=True)

    trainer.add_callback(PhaseTimer())

    class SampleLogger(TrainerCallback):
        def on_log(self, cfg_, st, ctl, logs=None, **kw):
            if not logs or int(os.environ.get("RANK", "0")) != 0:
                return
            state["step"] = st.global_step
            # Keep the per-term MEANS and STDS. The std is the number that
            # matters: GRPO's advantage is (r - mean)/std on the total reward,
            # so a term's influence on the gradient scales with its spread, not
            # its magnitude. `reward_variety` was sized by intuition at 0.80/
            # 0.60, came out at std 0.25 against the judge's 1.75, and so
            # contributed ~12% of the signal -- a deterministic penalty that
            # could not move the advantage. This filter listed neither the new
            # term nor any /std key, so the trace could not have shown that.
            if phase["steps"]:
                logs = dict(logs)
                logs["phase/mean_step_s"] = phase["step_s"] / phase["steps"]
                logs["phase/judge_frac"] = (
                    phase["judge_s"] / max(1e-9, phase["step_s"]))
            keep = {k: v for k, v in logs.items()
                    if k in ("reward", "reward_std", "kl", "loss",
                             "completions/mean_length",
                             "phase/mean_step_s", "phase/judge_frac")
                    or (k.startswith("rewards/") and
                        (k.endswith("/mean") or k.endswith("/std")))}
            if keep:
                state["log"].append({"step": st.global_step, **keep})
            if st.global_step % args.log_samples_every == 0:
                path = f"/work/grpo_runs/{args.save_name}_trace.json"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    json.dump(state["log"], f, indent=1)
                # Modal volumes need an explicit commit for a write to be
                # visible outside this container. Without it the trace exists
                # only until the container exits, which is why grpo_v13 ran 100
                # steps at interval 25 and left no trace file at all -- the run
                # was unobservable from outside while it was the only thing
                # worth observing.
                try:
                    import modal as _m
                    _m.Volume.from_name("npo-work").commit()
                except Exception as _e:
                    print(f"[trace] commit failed: {type(_e).__name__}", flush=True)
                print(f"[trace] step {st.global_step} {keep}", flush=True)

    # --- live adversarial follow-ups -------------------------------------
    # The point the offline pool cannot reach: a follow-up should answer what
    # the model ACTUALLY just said, at its current weights. Rank 0 drives the
    # vLLM server (which already carries the synced policy) for the defender
    # turns and gpt-oss-120b for the questions, writes the result to the shared
    # volume, and every rank reads that same file. All ranks must end up with
    # identical rows or the distributed sampler hands different prompts to
    # different ranks -- the same failure shape as the rank-0-only judge that
    # silently corrupted 220 steps.
    if args.refresh_every > 0:
        from npo.adversary import APIAdversary, archetype_probe, valid_probe

        # Openers for regenerated conversations: the single-turn prompts
        # already loaded. Using the same pool the offline rollout draws from
        # keeps the two comparable -- only WHEN the follow-ups are written
        # differs, which is the thing being tested.
        _seed_qs = [i["q"] for i in items if not i["hist"]]
        refresh_state = {"adv": None, "seeds": _seed_qs, "fails": 0}
        print(f"live adversary: every {args.refresh_every} steps, "
              f"{args.refresh_convos} convos from {len(_seed_qs)} seeds",
              flush=True)

        def _vllm_answer(client, tok_, convs: list) -> list:
            """One defender turn for each conversation, from the live policy."""
            prompts = [tok_.apply_chat_template(c, tokenize=False,
                                                add_generation_prompt=True)
                       for c in convs]
            ids = client.generate(prompts=prompts, n=1, temperature=1.0,
                                  max_tokens=200)
            return [tok_.decode(x, skip_special_tokens=True).strip()
                    for x in (ids[0] if isinstance(ids[0][0], list) else ids)]

        def regenerate(step: int) -> list:
            """Fresh 4-turn conversations against the current policy."""
            import random as _rnd

            client = getattr(trainer, "vllm_client", None)
            if client is None:
                raise RuntimeError("no vllm_client on trainer")
            if refresh_state["adv"] is None:
                refresh_state["adv"] = APIAdversary()
            adv = refresh_state["adv"]
            tok_ = trainer.processing_class
            rng = _rnd.Random(step)

            # Openers: part resampled from the fixed pool, part FRESHLY
            # GENERATED by an attacker few-shot conditioned on attacks that have
            # broken this policy in recent steps. Previously every opener came
            # from the pool, so turns 2-3 were the only thing the adversary
            # actually wrote and it never learned what worked.
            seeds = refresh_state["seeds"]
            n_ref = min(args.refresh_convos, len(seeds))
            n_fresh = int(n_ref * args.adversary_fresh_frac)
            openers = rng.sample(seeds, n_ref - n_fresh)
            if n_fresh:
                shots = (rng.sample(list(winner_bank), min(8, len(winner_bank)))
                         if winner_bank else [])
                try:
                    fresh = adv.attack_batch(n_fresh, shots, temperature=1.3)
                except Exception as e:
                    print(f"[live] fresh-attack batch failed: "
                          f"{type(e).__name__} {str(e)[:90]}", flush=True)
                    fresh = []
                openers += [x for x in fresh if x]
                print(f"[live] {len(fresh)} fresh openers from "
                      f"{len(winner_bank)} banked wins ({len(shots)} shots)",
                      flush=True)
                if len(openers) < n_ref:
                    openers += rng.sample(seeds, n_ref - len(openers))
            convs = [[{"role": "user", "content": q}] for q in openers]

            for t in range(3):
                answers = _vllm_answer(client, tok_, convs)
                for c, a in zip(convs, answers):
                    c.append({"role": "assistant", "content": a})
                if t == 2:
                    break
                # Last question is the one GRPO trains on, so it is always the
                # adversary's, never a template.
                for c in convs:
                    a = c[-1]["content"]
                    probe = ""
                    if t == 1 or rng.random() < 0.7:
                        probe = adv.probe(c, temperature=0.9)
                        if not valid_probe(probe, a):
                            probe = ""
                    if not probe:
                        _, probe = archetype_probe(a, c[0]["content"], rng)
                    c.append({"role": "user", "content": probe})

            return [{"prompt": c[:-1], "cls": "target",
                     "q": c[-2]["content"], "hist": c[:-2]}
                    for c in convs if len(c) >= 3]

        class LiveAdversary(TrainerCallback):
            def on_step_end(self, cfg_, st, ctl, **kw):
                if st.global_step == 0 or st.global_step % args.refresh_every:
                    return
                rank = int(os.environ.get("RANK", "0"))
                path = f"/work/grpo_runs/{args.save_name}_live.jsonl"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                if rank == 0:
                    try:
                        fresh = regenerate(st.global_step)
                        with open(path, "w", encoding="utf-8") as f:
                            for r in fresh:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        print(f"[live] step {st.global_step}: regenerated "
                              f"{len(fresh)} conversations", flush=True)
                    except Exception as e:
                        refresh_state["fails"] += 1
                        print(f"[live] step {st.global_step} FAILED "
                              f"({refresh_state['fails']}): {type(e).__name__} "
                              f"{str(e)[:160]}", flush=True)
                        open(path, "w").close()      # empty => every rank skips
                import torch.distributed as _d
                if _d.is_available() and _d.is_initialized():
                    _d.barrier()
                try:
                    fresh = [json.loads(l) for l in open(path, encoding="utf-8")]
                except OSError:
                    fresh = []
                if fresh:
                    n = pool.swap_multiturn(fresh)
                    if rank == 0:
                        print(f"[live] pool now carries {n} live conversations",
                              flush=True)

        trainer.add_callback(LiveAdversary())

    class SupportCurriculum(TrainerCallback):
        """Re-mix the prompt pool toward prompts that still produce a gradient.

        RANK CONSISTENCY IS THE WHOLE DIFFICULTY. TRL computes rewards per
        rank on that rank's slice of completions, so every rank's tracker has
        seen different data and would compute a different bucketing -- and a
        pool that differs by rank hands different prompts to different ranks
        through the distributed sampler. That is the same failure that
        silently corrupted 220 steps with a rank-0-only judge. So rank 0's
        bucketing is written to the volume, every rank waits on a barrier and
        reads that one file, exactly as LiveAdversary does. Rank 0 has seen
        only its own slice, which makes it a smaller sample -- but an
        identical one everywhere, and identical beats larger here.
        """

        def on_step_end(self, cfg_, st, ctl, **kw):
            every = max(1, args.refresh_every or 25)
            if st.global_step == 0 or st.global_step % every:
                return
            import random as _rnd

            import torch.distributed as _d
            rank = int(os.environ.get("RANK", "0"))
            path = f"/work/grpo_runs/{args.save_name}_buckets.json"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if rank == 0:
                b = {qq: support.bucket(qq) for qq in support.stat}
                json.dump(b, open(path, "w"))
                print(f"[support] step {st.global_step}: {support.split()}",
                      flush=True)
            if _d.is_available() and _d.is_initialized():
                _d.barrier()
            try:
                b = json.load(open(path))
            except (OSError, ValueError):
                return
            got = pool.reweight(b, _rnd.Random(st.global_step))
            if rank == 0:
                print(f"[support] pool re-mixed from {got}", flush=True)

    trainer.add_callback(SupportCurriculum())
    trainer.add_callback(SampleLogger())

    # Modal preempts multi-GPU nodes and restarts the function "with the same
    # input" -- which without this means starting training over at step 0. The
    # last run reached step 221 of 250, was preempted, and began again from
    # scratch. An 8xH100 job long enough to matter WILL be preempted, so resume
    # is not an optimisation, it is the difference between finishing and not.
    #
    # save_only_model drops optimizer state, so Adam restarts cold on resume.
    # At lr 1e-6 over a few hundred steps that is a small perturbation, and it
    # is the right trade against the 130 GB/checkpoint that filled the volume.
    trainer.train()

    tag = args.init_from.rstrip("/").split("/")[-1]
    out = f"/work/checkpoints/{tag}__{args.save_name}"
    trainer.save_model(out)
    with open(f"/work/grpo_runs/{args.save_name}_log.json", "w") as f:
        json.dump({"args": vars(args), "log": state["log"]}, f, indent=2)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
