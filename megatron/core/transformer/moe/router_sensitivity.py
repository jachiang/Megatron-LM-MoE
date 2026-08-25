# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Routing-sensitivity measurement for MoE routers.

For each sampled token this module computes how far the router's top-K decision is from
flipping, through the two channels that can flip it:

    eps_h   the smallest activation perturbation ||e||_2 that changes the selection, to
            first order:  min over (winner i, loser j) of

                (r_i - r_j) / || sigma'_i w_i - sigma'_j w_j ||_2

    eps_b   the smallest per-expert bias disagreement ||delta||_inf that changes the
            selection: half the gap between the last selected and first rejected score.
            Exact, not first order -- the bias enters the score with unit coefficient.

Both are margins over rates, so all information sits in the lower tail of their
distribution.  The module therefore accumulates fixed-bin log-spaced integer histograms
(per layer, per quantity) rather than means: integer counts are exactly reducible across
ranks in any order, so the pooled histogram is identical whatever the parallel layout --
the measurement cannot be confounded by the very layout dependence it is measuring.

The per-pair rate needs ||sigma'_i w_i - sigma'_j w_j||, which reduces to three lookups in
the Gram matrix G = W W^T (formed once per sampled step):

    ||g_ij||^2 = sigma'_i^2 G_ii - 2 sigma'_i sigma'_j G_ij + sigma'_j^2 G_jj .

Pair skipping.  Not all K(E-K) pairs need the Gram evaluation.  With M = max_k ||w_k||_2
and sigma' <= 1/4, every pair obeys eps_ij >= 2 (r_i - r_j) / M.  The boundary pair
(K-th vs (K+1)-th score) is evaluated exactly first, giving an upper bound eps_ub on the
minimum; any pair whose margin satisfies

    r_i - r_j >= (M / 2) * eps_ub

then has eps_ij >= eps_ub and cannot lower the minimum, so its Gram evaluation is skipped.
The final minimum is taken over the surviving pairs and eps_ub itself, which keeps the
result exact.  The skip test is one comparison per pair on margins that are needed anyway;
on synthetic score distributions it removes the Gram gathers for the large majority of
pairs.  All of it is vectorised -- the "scan" of the paper's Proposition is realised as a
mask, not a loop.
"""

from typing import Optional, Tuple

import torch

# Fixed histogram geometry.  Edges are log-spaced and identical on every rank and every
# step, so integer counts are comparable and reducible.  10 bins per decade over
# [1e-12, 1e2); values outside are clamped into the end bins.
SENS_HIST_LOG10_MIN = -12.0
SENS_HIST_LOG10_MAX = 2.0
SENS_HIST_BINS_PER_DECADE = 10
SENS_HIST_NUM_BINS = int(
    (SENS_HIST_LOG10_MAX - SENS_HIST_LOG10_MIN) * SENS_HIST_BINS_PER_DECADE
)

# Row order of the per-layer histogram buffer.
SENS_QUANTITIES = ('eps_h', 'eps_b', 'eps_h_rel', 'eps_b_rel')


@torch.no_grad()
def compute_router_sensitivity(
    hidden: torch.Tensor,
    logits: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    topk: int,
    score_function: str,
    bias_sign: float = 1.0,
    prune: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token routing sensitivity, vectorised, with pair skipping.

    Args:
        hidden: [T, H] token hidden states (any float dtype; cast to fp32 internally).
        logits: [T, E] router logits for the same tokens.
        weight: [E, H] router weight.
        bias: [E] per-expert bias added to the selection score, or None.  Pass
            ``expert_bias`` (aux-loss-free) or ``qb_beta`` with ``bias_sign=-1.0``
            (quantile balancing subtracts its offset).
        topk: K.
        score_function: 'sigmoid' or 'softmax'.  With 'softmax' and no bias the
            selection is a plain top-K of the logits, so the rate is ||w_i - w_j||
            with no sigma'; with 'sigmoid' the score is sigma(logit) + bias and the
            rate carries sigma'.
        bias_sign: +1.0 if ``bias`` is added to the score, -1.0 if subtracted.
        prune: apply the pair-skipping criterion (default).  ``False`` evaluates every
            pair; used by tests to verify that pruning does not change the result.

    Returns:
        (eps_h, eps_b, h_norm, b_inf): eps_h and eps_b are [T] fp32; h_norm is [T];
        b_inf is a scalar tensor (0 where bias is None).
    """
    hidden = hidden.detach().to(torch.float32)
    logits = logits.detach().to(torch.float32)
    weight = weight.detach().to(torch.float32)

    num_tokens, num_experts = logits.shape
    assert 0 < topk < num_experts

    if score_function == 'sigmoid':
        s = torch.sigmoid(logits)
        sp = s * (1.0 - s)  # sigma', already available in spirit from the routing
        r = s
    elif score_function == 'softmax':
        # Softmax is order-preserving, so selection reduces to a top-K of the logits
        # (plus bias); the selection score is linear in h and sigma' drops out.
        sp = None
        r = logits
    else:
        raise ValueError(f'Unsupported score function for sensitivity: {score_function}')

    if bias is not None:
        r = r + bias_sign * bias.detach().to(torch.float32)

    # --- eps_b: half the boundary gap.  Exact. --------------------------------------
    top = r.topk(topk + 1, dim=1)
    gap = top.values[:, topk - 1] - top.values[:, topk]  # [T] >= 0
    eps_b = 0.5 * gap

    sel_idx = top.indices[:, :topk]  # [T, K]
    sel_mask = torch.zeros_like(r, dtype=torch.bool).scatter_(1, sel_idx, True)

    # --- margins for all winner/loser pairs ------------------------------------------
    # [T, K, E]; loser columns selected by masking winners out afterwards.
    r_sel = top.values[:, :topk]  # [T, K]
    margins = r_sel.unsqueeze(2) - r.unsqueeze(1)  # [T, K, E]

    # Gram pieces, formed once per call.
    G = weight @ weight.t()  # [E, E]
    Gd = G.diagonal()  # [E]
    M = Gd.max().sqrt()  # largest row norm

    if sp is None:
        sp_sel = torch.ones_like(r_sel)
        sp_all = torch.ones_like(r)
        rate_cap = 2.0 * M  # ||w_i - w_j|| <= 2M by the triangle inequality
    else:
        sp_sel = sp.gather(1, sel_idx)  # [T, K]
        sp_all = sp
        rate_cap = 0.5 * M  # sigma' <= 1/4 on both terms of the triangle inequality

    # --- exact eps for the boundary pair: the upper bound for pruning ----------------
    b_win = sel_idx[:, topk - 1 : topk]  # [T, 1] the K-th (last selected) expert
    b_los = top.indices[:, topk : topk + 1]  # [T, 1] (K+1)-th
    spw = sp_sel[:, topk - 1 : topk] if sp is not None else torch.ones_like(gap)[:, None]
    spl = sp_all.gather(1, b_los) if sp is not None else torch.ones_like(gap)[:, None]
    g2_boundary = (
        spw.square() * Gd[b_win]
        - 2.0 * spw * spl * G[b_win, b_los]
        + spl.square() * Gd[b_los]
    ).clamp_min(1e-30)
    eps_ub = (gap[:, None] / g2_boundary.sqrt()).squeeze(1)  # [T] exact for that pair

    # --- pair skipping (Proposition: pair-skipping criterion), vectorised ------------
    # A pair with margin >= (rate_cap) * eps_ub has eps_ij >= eps_ub: skip its Gram work.
    # Losers only; winner-winner "pairs" are excluded via +inf margins.
    margins = margins.masked_fill(sel_mask.unsqueeze(1), float('inf'))
    if prune:
        keep = margins < rate_cap * eps_ub[:, None, None]  # [T, K, E]
    else:
        keep = torch.isfinite(margins)

    flat = keep.nonzero(as_tuple=False)  # [P, 3] -> (token, winner slot, loser expert)
    if flat.numel() == 0:
        eps_h = eps_ub
    else:
        t_i, k_i, j_i = flat.unbind(1)
        i_i = sel_idx[t_i, k_i]  # winner expert index
        spi = sp_all[t_i, i_i] if sp is not None else torch.ones_like(t_i, dtype=torch.float32)
        spj = sp_all[t_i, j_i] if sp is not None else torch.ones_like(t_i, dtype=torch.float32)
        g2 = (
            spi.square() * Gd[i_i] - 2.0 * spi * spj * G[i_i, j_i] + spj.square() * Gd[j_i]
        ).clamp_min(1e-30)
        eps_pair = margins[t_i, k_i, j_i] / g2.sqrt()

        eps_h = torch.full_like(eps_ub, float('inf'))
        eps_h.scatter_reduce_(0, t_i, eps_pair, reduce='amin', include_self=True)
        # The boundary pair is always a candidate; this also covers tokens whose every
        # other pair was skipped.
        eps_h = torch.minimum(eps_h, eps_ub)

    h_norm = hidden.norm(dim=-1)
    if bias is not None:
        b_inf = bias.detach().abs().max().to(torch.float32)
    else:
        b_inf = torch.zeros((), dtype=torch.float32, device=r.device)

    return eps_h, eps_b, h_norm, b_inf


@torch.no_grad()
def bin_sensitivity(
    hist: torch.Tensor,
    eps_h: torch.Tensor,
    eps_b: torch.Tensor,
    h_norm: torch.Tensor,
    b_inf: torch.Tensor,
) -> None:
    """Accumulate the four quantities into the layer's int64 histogram in place.

    ``hist`` is [4, SENS_HIST_NUM_BINS], rows ordered as SENS_QUANTITIES.  Bins are
    log10-spaced on [SENS_HIST_LOG10_MIN, SENS_HIST_LOG10_MAX); out-of-range values are
    clamped into the end bins, and non-positive values go to the lowest bin.
    """
    rows = [eps_h, eps_b, eps_h / h_norm.clamp_min(1e-30)]
    # A router with no bias has no bias channel; leave that histogram empty rather
    # than filling the lowest bin with zeros.
    rows.append(eps_b / b_inf if float(b_inf) > 0.0 else eps_b[:0])

    scale = float(SENS_HIST_BINS_PER_DECADE)
    for row_idx, values in enumerate(rows):
        if values.numel() == 0:
            continue
        idx = (values.clamp_min(1e-300).log10() - SENS_HIST_LOG10_MIN) * scale
        idx = idx.long().clamp_(0, SENS_HIST_NUM_BINS - 1)
        hist[row_idx].scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.int64))


def quantiles_from_hist(hist_row: torch.Tensor, qs=(0.001, 0.01, 0.1, 0.5)):
    """Lower-bound quantiles (bin left edges) from one pooled int64 histogram row."""
    total = int(hist_row.sum())
    if total == 0:
        return {q: float('nan') for q in qs}
    cum = hist_row.cumsum(0)
    out = {}
    for q in qs:
        target = max(1, int(q * total))
        bin_idx = int(torch.searchsorted(cum, torch.tensor(target, device=cum.device)))
        bin_idx = min(bin_idx, SENS_HIST_NUM_BINS - 1)
        out[q] = 10.0 ** (SENS_HIST_LOG10_MIN + bin_idx / SENS_HIST_BINS_PER_DECADE)
    return out


def fraction_below(hist_row: torch.Tensor, threshold: float) -> float:
    """Fraction of counted values below ``threshold``, conservatively (whole bins)."""
    total = int(hist_row.sum())
    if total == 0:
        return float('nan')
    import math

    edge = (math.log10(threshold) - SENS_HIST_LOG10_MIN) * SENS_HIST_BINS_PER_DECADE
    nbins = min(max(int(math.ceil(edge)), 0), SENS_HIST_NUM_BINS)
    return float(hist_row[:nbins].sum()) / total
