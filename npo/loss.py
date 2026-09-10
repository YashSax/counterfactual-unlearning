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


def npo_loss(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """NPO forget loss.

    Args:
        policy_logprobs: (B,) log pi_theta(y|x) for forget sequences.
        ref_logprobs: (B,) log pi_ref(y|x), frozen reference.
        beta: temperature. Smaller -> closer to gradient ascent.

    Returns:
        (scalar loss, metrics)
    """
    delta = policy_logprobs - ref_logprobs
    loss = (2.0 / beta) * F.softplus(beta * delta).mean()

    with torch.no_grad():
        # The per-example gradient weight, sigmoid(beta*delta) in [0,1]. This is
        # NPO's whole mechanism: it decays toward 0 as the policy moves away
        # from the forget example, so already-forgotten text stops pulling.
        # If this sits pinned near 1.0, beta is too small / delta too positive
        # and you are effectively running gradient ascent.
        weight = torch.sigmoid(beta * delta)
        metrics = {
            "npo_loss": loss.item(),
            "forget_delta": delta.mean().item(),
            "forget_logprob": policy_logprobs.mean().item(),
            "forget_ref_logprob": ref_logprobs.mean().item(),
            "npo_weight": weight.mean().item(),
            "forget_frac_below_ref": (delta < 0).float().mean().item(),
        }
    return loss, metrics


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


def combined_loss(
    npo: torch.Tensor,
    retain: torch.Tensor,
    retain_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L = L_NPO + retain_weight * NLL_retain."""
    total = npo + retain_weight * retain
    return total, {"total_loss": total.item(), "retain_weight": retain_weight}


# ---------------------------------------------------------------------------
# Self-tests: the properties that matter, checked directly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. Equivalence of the three algebraic forms of the NPO objective.
    beta = 0.15
    d = torch.randn(2048) * 3
    softplus_form = (2 / beta) * F.softplus(beta * d)
    logsigmoid_form = -(2 / beta) * F.logsigmoid(-beta * d)
    naive_form = (2 / beta) * torch.log1p(torch.exp(beta * d.double())).float()
    assert torch.allclose(softplus_form, logsigmoid_form, atol=1e-5), "softplus != logsigmoid"
    assert torch.allclose(softplus_form, naive_form, atol=1e-4), "softplus != log(1+r^beta)"
    print("ok  three forms of L_NPO agree")

    # 2. Stability where the naive form overflows.
    big = torch.tensor([500.0, -500.0, 1e4])
    assert torch.isfinite((2 / beta) * F.softplus(beta * big)).all()
    assert not torch.isfinite(torch.log1p(torch.exp(beta * big))).all(), "expected naive overflow"
    print("ok  softplus finite where log1p(exp(.)) overflows")

    # 3. Loss decreases as the policy moves away from the forget text.
    pol = torch.tensor([-1.0, -2.0, -3.0, -8.0])
    ref = torch.full((4,), -1.0)
    losses = [npo_loss(pol[i : i + 1], ref[i : i + 1], beta)[0].item() for i in range(4)]
    assert losses == sorted(losses, reverse=True), f"not monotone decreasing: {losses}"
    print(f"ok  loss decreases as policy diverges from ref: {[round(x,4) for x in losses]}")

    # 4. NPO's gradient weight decays; gradient ascent's does not.
    #    This is the property that prevents catastrophic collapse.
    for far in (0.0, 5.0, 20.0):
        p = (ref[:1] - far).clone().requires_grad_(True)
        npo_loss(p, ref[:1], beta)[0].backward()
        g = p.grad.abs().item()
        print(f"    delta=-{far:<5.1f} |grad|={g:.6f}")
        if far == 20.0:
            assert g < 0.15, f"weight should have decayed, got {g}"
    print("ok  gradient weight decays as text is forgotten")

    # 5. beta -> 0 recovers gradient ascent (unit gradient on delta).
    p = torch.tensor([-1.0], requires_grad=True)
    npo_loss(p, ref[:1], 1e-6)[0].backward()
    assert abs(p.grad.item() - 1.0) < 1e-3, p.grad.item()
    print("ok  beta -> 0 limit reduces to gradient ascent")

    # 6. sequence_logprobs matches a hand-rolled reference, and normalization
    #    actually divides by the number of *scored* tokens.
    B, T, V = 3, 7, 11
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T)
    mask[0, 5:] = 0  # ragged
    lp_sum = sequence_logprobs(logits, labels, mask, length_normalize=False)
    lp_mean = sequence_logprobs(logits, labels, mask, length_normalize=True)
    ref_sum = torch.stack([
        sum(F.log_softmax(logits[b, t], -1)[labels[b, t + 1]] * mask[b, t + 1] for t in range(T - 1))
        for b in range(B)
    ])
    assert torch.allclose(lp_sum, ref_sum, atol=1e-5), (lp_sum, ref_sum)
    assert torch.allclose(lp_mean, lp_sum / mask[:, 1:].sum(-1), atol=1e-5)
    print("ok  sequence_logprobs matches reference impl (sum and mean)")

    # 7. Retain NLL agrees with F.cross_entropy under a full mask.
    nll, _ = retain_nll_loss(logits, labels, torch.ones(B, T))
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1))
    assert torch.allclose(nll, ce, atol=1e-5), (nll, ce)
    print("ok  retain_nll_loss == F.cross_entropy")

    # 8. Chunked token_logprobs matches the naive log_softmax implementation,
    #    and chunk size does not change the answer.
    lg = torch.randn(2, 40, 97, dtype=torch.float32)
    lb = torch.randint(0, 97, (2, 40))
    naive = torch.gather(F.log_softmax(lg.float(), -1), 2, lb.unsqueeze(2)).squeeze(2)
    for cs in (1, 7, 40, 1000):
        got = token_logprobs(lg, lb, chunk_size=cs)
        assert torch.allclose(got, naive, atol=1e-5), f"chunk_size={cs} disagrees"
    print("ok  chunked token_logprobs == gather(log_softmax(.)) for all chunk sizes")

    # 9. Gradients still flow through the chunked path.
    lg = torch.randn(2, 16, 51, requires_grad=True)
    token_logprobs(lg, torch.randint(0, 51, (2, 16))).sum().backward()
    assert lg.grad is not None and torch.isfinite(lg.grad).all()
    print("ok  chunked path is differentiable")

    print("\nall NPO loss tests passed")
