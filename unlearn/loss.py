"""
Negative Preference Optimization (NPO) for unlearning.

Zhang et al. 2024, "Negative Preference Optimization: From Catastrophic
Collapse to Effective Unlearning" (arXiv:2404.05868).

NPO is DPO with the chosen term deleted -- only the rejected (forget) branch
survives, anchored against a frozen reference model:

    L_NPO = (2/beta) * E_forget[ log(1 + (pi_theta(y|x) / pi_ref(y|x))^beta) ]
          = -(2/beta) * E_forget[ log sigmoid( -beta * delta ) ]
          = (2/beta) * E_forget[ softplus(beta * delta) ]

where delta = log pi_theta(y|x) - log pi_ref(y|x).

The softplus form is what we implement: it is the numerically stable one, and
it makes the beta -> 0 limit legible. As beta -> 0,

    (2/beta) * softplus(beta*delta)  ->  2*log(2)/beta + delta

whose gradient is just grad(delta) -- plain gradient ascent on the forget set.
NPO's whole contribution is that for beta > 0 the weight on each example decays
once the model has already moved away from it, which is what stops the runaway
divergence that makes gradient ascent collapse.

Total objective (NPO-RT, the retain-set variant):

    L = L_NPO(forget) + retain_weight * NLL(retain)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Per-token log pi(label), computed without materializing log_softmax.

    The obvious implementation, gather(log_softmax(logits.float())), allocates a
    full (B, T, V) float32 tensor. Qwen3's vocabulary is ~152k, so a batch of 8
    at length 512 asks for 8*512*151936*4 = 2.5 GB in a single allocation on top
    of the bf16 logits it was handed -- which OOMs a 23 GB card outright and
    wastes a lot of an 80 GB one.

    log p(y) = logit_y - logsumexp(logits) needs only a (B, T) output, so we
    chunk along time and never hold more than (B, chunk, V) in float32.
    """
    outs = []
    for i in range(0, logits.shape[1], chunk_size):
        lc = logits[:, i : i + chunk_size, :]
        tc = labels[:, i : i + chunk_size]
        target = torch.gather(lc, 2, tc.unsqueeze(2)).squeeze(2).float()
        outs.append(target - torch.logsumexp(lc.float(), dim=-1))
    return torch.cat(outs, dim=1)


def sequence_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    length_normalize: bool = True,
) -> torch.Tensor:
    """Per-sequence log-probability, log pi(y|x).

    Args:
        logits: (B, T, V) raw model logits, NOT yet shifted.
        labels: (B, T) token ids aligned with logits.
        loss_mask: (B, T) 1.0 on tokens that count, 0.0 on padding/prompt.
        length_normalize: divide by the number of scored tokens.

    Returns:
        (B,) log-probabilities.

    On length_normalize -- this defaults to True and that is deliberate.

    The original NPO/TOFU setup sums token log-probs over short QA answers.
    Our forget set is Wikipedia prose at ~350-550 tokens per chunk, where the
    summed delta reaches hundreds of nats. softplus(beta*delta) then saturates
    into its linear regime, softplus(z) -> z, and the objective degenerates to

        (2/beta) * beta * delta = 2*delta

    which is exactly the gradient ascent that NPO exists to avoid -- silently,
    with no error. Length normalization keeps delta in a range where the
    softplus non-linearity is actually doing work, and stops 2200-char chunks
    from outweighing 400-char ones by 5x for no principled reason.
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    loss_mask = loss_mask[:, 1:]

    tlp = token_logprobs(logits, labels)

    summed = (tlp * loss_mask).sum(dim=-1)
    if length_normalize:
        return summed / loss_mask.sum(dim=-1).clamp(min=1.0)
    return summed




def retain_nll_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token-mean cross-entropy on the retain set."""
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    loss_mask = loss_mask[:, 1:]

    tlp = token_logprobs(logits, labels)

    n = loss_mask.sum().clamp(min=1.0)
    loss = -(tlp * loss_mask).sum() / n
    return loss, {"retain_nll": loss.item(), "retain_ppl": loss.exp().item()}




# ---------------------------------------------------------------------------
# Self-tests: the properties that matter, checked directly.
# ---------------------------------------------------------------------------
