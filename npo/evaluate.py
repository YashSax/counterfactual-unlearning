"""
Unlearning evaluation: did the model forget 9/11 without losing anything else?

A single number cannot answer that, so we report four perplexities plus probes.
The comparison that matters is base vs. unlearned on the SAME slices:

    forget_train    9/11 text the NPO objective optimized against.
                    PPL must rise. On its own this proves almost nothing --
                    it is the training objective.

    forget_holdout  9/11 text from 7 reserved articles, never trained on.
                    THE headline metric. A rise here is forgetting; a flat
                    line means we suppressed 449 passages, not a fact.

    retain          9/11-adjacent knowledge the spec says to keep (WTC
                    architecture, bin Laden, Al-Qaeda, TSA, Patriot Act).
                    Must stay flat. A rise is collateral damage.

    general         Unrelated Wikipedia. Must stay flat. A rise is
                    catastrophic collapse -- the failure mode NPO exists to
                    avoid, and the reason to watch it rather than assume it.

The spec's real requirement is a ratio, not a level: forget PPL up sharply
while retain and general stay put. We print that ratio directly.
"""

from __future__ import annotations

import json
import math
import os
import sys

import torch
from torch.utils.data import DataLoader


# Perplexity is exp(NLL), so a well-forgotten slice overflows float fast. We
# report NLL as the primary quantity -- it stays finite and exact -- and clamp
# only the human-facing perplexity. The first version clamped at 20, which made
# forget_train and forget_holdout print the IDENTICAL 485165195.41 (= e^20) and
# silently destroyed the difference between two slices we specifically need to
# compare.
PPL_CLAMP = 700.0  # just under float overflow for exp()


def _ppl(nll: float) -> float:
    return math.exp(min(nll, PPL_CLAMP))


def corpus_perplexity(model, dataset, pad_id: int, batch_size: int = 4) -> dict:
    """Token-weighted perplexity over a CorpusDataset."""
    sys.path.insert(0, "/root")
    from npo.data import make_collate_fn

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=make_collate_fn(pad_id)
    )
    total_nll, total_tok = 0.0, 0
    per_article: dict[str, list[float]] = {}

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].cuda()
            attn = batch["attention_mask"].cuda()
            mask = batch["loss_mask"].cuda()
            logits = model(input_ids=ids, attention_mask=attn).logits

            lp = torch.gather(
                torch.log_softmax(logits[:, :-1].float(), dim=-1),
                2,
                ids[:, 1:].unsqueeze(2),
            ).squeeze(2)
            m = mask[:, 1:]
            total_nll += float(-(lp * m).sum())
            total_tok += int(m.sum())

            seq_nll = -(lp * m).sum(-1) / m.sum(-1).clamp(min=1)
            for idx, v in zip(batch["indices"].tolist(), seq_nll.tolist()):
                per_article.setdefault(dataset.examples[idx].article, []).append(v)

    mean_nll = total_nll / max(total_tok, 1)
    return {
        "nll": mean_nll,
        "ppl": _ppl(mean_nll),
        "ppl_clamped": mean_nll > PPL_CLAMP,
        "n_tokens": total_tok,
        "n_examples": len(dataset),
        "per_article_nll": {a: sum(v) / len(v) for a, v in sorted(per_article.items())},
    }


def compare(base: dict, after: dict) -> dict:
    """Ratio report. Unlearning succeeded iff forget ratios are >> 1 while
    retain and general sit near 1."""
    out = {}
    for slice_name in ("forget_train", "forget_holdout", "retain", "general"):
        if slice_name in base and slice_name in after:
            bn, an = base[slice_name]["nll"], after[slice_name]["nll"]
            # Work in NLL space: the ppl ratio is exp(delta_nll), which overflows
            # for a strongly forgotten slice, while delta_nll stays exact.
            d = an - bn
            out[slice_name] = {
                "base_nll": round(bn, 4),
                "unlearned_nll": round(an, 4),
                "delta_nll": round(d, 4),
                "base_ppl": round(_ppl(bn), 3),
                "unlearned_ppl": round(_ppl(an), 3),
                "ratio": round(math.exp(d), 3) if d < 700 else float("inf"),
                "log10_ratio": round(d / math.log(10), 2),
            }
    return out


def verdict(cmp: dict, forget_min_ratio: float = 3.0, retain_max_ratio: float = 1.15) -> dict:
    """Turn the ratios into an explicit pass/fail against the spec's two-sided
    requirement, so a large forget ratio bought with a wrecked retain set does
    not read as success."""
    checks = {}
    # Compare in NLL space so an infinite ppl ratio does not break the checks.
    import math as _m

    ho = cmp.get("forget_holdout", {}).get("delta_nll")
    tr = cmp.get("forget_train", {}).get("delta_nll")
    rt = cmp.get("retain", {}).get("delta_nll")
    gn = cmp.get("general", {}).get("delta_nll")
    forget_min = _m.log(forget_min_ratio)
    retain_max = _m.log(retain_max_ratio)

    if tr is not None:
        checks["forget_train_rose"] = (
            tr >= forget_min, f"d_nll {tr:+.3f} >= {forget_min:.3f} (x{forget_min_ratio})")
    if ho is not None:
        checks["forget_generalized"] = (
            ho >= forget_min, f"d_nll {ho:+.3f} >= {forget_min:.3f} (x{forget_min_ratio})")
    if rt is not None:
        checks["retain_preserved"] = (
            rt <= retain_max, f"d_nll {rt:+.3f} <= {retain_max:.3f} (x{retain_max_ratio})")
    if gn is not None:
        checks["no_collapse"] = (
            gn <= retain_max, f"d_nll {gn:+.3f} <= {retain_max:.3f} (x{retain_max_ratio})")

    passed = all(ok for ok, _ in checks.values())
    return {
        "checks": {k: {"pass": ok, "detail": d} for k, (ok, d) in checks.items()},
        "overall_pass": passed,
    }


def format_report(cmp: dict, vd: dict) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("NPO UNLEARNING REPORT — perplexity, base vs. unlearned")
    lines.append("=" * 72)
    lines.append(f"{'slice':<16}{'base nll':>10}{'unlearn nll':>13}"
                 f"{'d_nll':>9}{'ppl x':>13}   want")
    lines.append("-" * 72)
    want = {
        "forget_train": "UP",
        "forget_holdout": "UP  <- headline",
        "retain": "flat",
        "general": "flat",
    }
    for k, v in cmp.items():
        r = v["ratio"]
        rs = f"1e{v['log10_ratio']:.1f}" if (r == float("inf") or r >= 1e6) else f"{r:.2f}"
        lines.append(
            f"{k:<16}{v['base_nll']:>10.3f}{v['unlearned_nll']:>13.3f}"
            f"{v['delta_nll']:>+9.3f}{rs:>13}   {want.get(k, '')}"
        )
    lines.append("-" * 72)
    for name, c in vd["checks"].items():
        lines.append(f"  [{'PASS' if c['pass'] else 'FAIL'}] {name:<22} {c['detail']}")
    lines.append("=" * 72)
    lines.append(f"OVERALL: {'PASS' if vd['overall_pass'] else 'FAIL'}")
    return "\n".join(lines)
