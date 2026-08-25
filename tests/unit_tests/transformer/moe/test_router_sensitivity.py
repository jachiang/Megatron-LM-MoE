# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for routing-sensitivity measurement.

Run on a GPU node:
    torchrun --nproc_per_node=1 -m pytest tests/unit_tests/transformer/moe/test_router_sensitivity.py -v
or directly (the tests are single-process and need no distributed init):
    pytest tests/unit_tests/transformer/moe/test_router_sensitivity.py -v
"""

import math

import pytest
import torch

from megatron.core.transformer.moe.router_sensitivity import (
    SENS_HIST_NUM_BINS,
    bin_sensitivity,
    compute_router_sensitivity,
    fraction_below,
    quantiles_from_hist,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _reference_eps(hidden, logits, weight, bias, topk, score_function, bias_sign):
    """Brute-force reference: full scan over all winner/loser pairs, no pruning."""
    hidden = hidden.double()
    logits = logits.double()
    weight = weight.double()
    T, E = logits.shape
    if score_function == 'sigmoid':
        s = torch.sigmoid(logits)
        sp = s * (1 - s)
        r = s
    else:
        sp = torch.ones_like(logits)
        r = logits
    if bias is not None:
        r = r + bias_sign * bias.double()

    eps_h = torch.empty(T, dtype=torch.float64)
    eps_b = torch.empty(T, dtype=torch.float64)
    for t in range(T):
        order = torch.argsort(r[t], descending=True)
        sel, uns = order[:topk], order[topk:]
        eps_b[t] = 0.5 * (r[t, order[topk - 1]] - r[t, order[topk]])
        best = math.inf
        for i in sel.tolist():
            for j in uns.tolist():
                g = sp[t, i] * weight[i] - sp[t, j] * weight[j]
                best = min(best, float((r[t, i] - r[t, j]) / g.norm().clamp_min(1e-30)))
        eps_h[t] = best
    return eps_h, eps_b


@pytest.mark.parametrize('score_function', ['sigmoid', 'softmax'])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_pruning_is_exact(score_function, seed):
    """Pair skipping must not change the minimum: prune vs full scan, same pipeline.

    Both paths run the identical fp32 arithmetic; only the set of evaluated pairs
    differs, so any deviation beyond reduction-order noise is a pruning bug.
    """
    torch.manual_seed(seed)
    T, E, H, K = 256, 64, 96, 4
    hidden = torch.randn(T, H, device=DEVICE)
    weight = torch.randn(E, H, device=DEVICE) / math.sqrt(H)
    logits = hidden @ weight.t() + 0.5 * torch.randn(E, device=DEVICE)
    bias = 0.05 * torch.randn(E, device=DEVICE)
    bias = bias - bias.mean()

    pruned = compute_router_sensitivity(
        hidden, logits, weight, bias, K, score_function, bias_sign=1.0, prune=True
    )
    full = compute_router_sensitivity(
        hidden, logits, weight, bias, K, score_function, bias_sign=1.0, prune=False
    )
    assert torch.allclose(pruned[0], full[0], rtol=1e-6, atol=0.0), (
        'pair skipping changed the minimum: max rel err %.3e'
        % float(((pruned[0] - full[0]).abs() / full[0]).max())
    )
    assert torch.equal(pruned[1], full[1])  # eps_b does not involve pruning at all


@pytest.mark.parametrize('score_function', ['sigmoid', 'softmax'])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_matches_brute_force(score_function, seed):
    """Against an independent double-precision scan.

    The implementation runs in fp32 and forms rates through the Gram identity; the
    reference recomputes everything in fp64 with direct vector norms.  The comparison
    therefore carries genuine fp32 pipeline error (sigmoid, margin cancellation, Gram
    vs direct norm), so the tolerance is 1e-3 relative -- this test checks the algebra,
    test_pruning_is_exact checks the pruning.
    """
    torch.manual_seed(seed)
    T, E, H, K = 64, 32, 48, 4
    hidden = torch.randn(T, H, device=DEVICE)
    weight = torch.randn(E, H, device=DEVICE) / math.sqrt(H)
    logits = hidden @ weight.t() + 0.5 * torch.randn(E, device=DEVICE)
    bias = 0.05 * torch.randn(E, device=DEVICE)
    bias = bias - bias.mean()

    eps_h, eps_b, h_norm, b_inf = compute_router_sensitivity(
        hidden, logits, weight, bias, K, score_function, bias_sign=1.0
    )
    ref_h, ref_b = _reference_eps(
        hidden.cpu(), logits.cpu(), weight.cpu(), bias.cpu(), K, score_function, 1.0
    )
    assert torch.allclose(eps_b.cpu().double(), ref_b, rtol=1e-5, atol=1e-7)
    assert torch.allclose(eps_h.cpu().double(), ref_h, rtol=1e-3, atol=1e-7), (
        'max rel err vs fp64 brute force %.3e'
        % float(((eps_h.cpu().double() - ref_h).abs() / ref_h).max())
    )
    assert torch.allclose(h_norm.cpu(), hidden.cpu().norm(dim=-1), rtol=1e-5)
    assert float(b_inf) == pytest.approx(float(bias.abs().max()), rel=1e-6)


def test_qb_bias_sign():
    """QB subtracts its offset; adding beta with sign -1 must equal adding -beta with +1."""
    torch.manual_seed(0)
    T, E, H, K = 16, 16, 24, 2
    hidden = torch.randn(T, H, device=DEVICE)
    weight = torch.randn(E, H, device=DEVICE) / math.sqrt(H)
    logits = hidden @ weight.t()
    beta = 0.1 * torch.randn(E, device=DEVICE)

    a = compute_router_sensitivity(hidden, logits, weight, beta, K, 'sigmoid', bias_sign=-1.0)
    b = compute_router_sensitivity(hidden, logits, weight, -beta, K, 'sigmoid', bias_sign=1.0)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])


def test_histogram_counts_and_stats():
    hist = torch.zeros(4, SENS_HIST_NUM_BINS, dtype=torch.int64, device=DEVICE)
    n = 1000
    values = torch.logspace(-6, -1, n, device=DEVICE)
    bin_sensitivity(
        hist, values, values, torch.ones_like(values), torch.tensor(1.0, device=DEVICE)
    )
    assert int(hist[0].sum()) == n  # every value counted exactly once
    # Values are log-uniform on [1e-6, 1e-1]: about 1/5 of them sit below 1e-5.
    frac = fraction_below(hist[0], 1e-5)
    assert abs(frac - 0.2) < 0.03
    qs = quantiles_from_hist(hist[0])
    # Median of a log-uniform sample on [1e-6, 1e-1] is ~10^-3.5; bins are decade/10.
    assert 1e-4 < qs[0.5] < 1e-3


def test_histogram_reduction_order_invariance():
    """The property the design depends on: counts pool identically however sharded."""
    torch.manual_seed(0)
    values = torch.logspace(-8, 0, 4096, device=DEVICE)[torch.randperm(4096)]
    pooled = torch.zeros(4, SENS_HIST_NUM_BINS, dtype=torch.int64, device=DEVICE)
    bin_sensitivity(pooled, values, values, torch.ones_like(values), torch.tensor(1.0))

    for shards in (2, 8, 32):
        acc = torch.zeros_like(pooled)
        for chunk in values.chunk(shards):
            part = torch.zeros_like(pooled)
            bin_sensitivity(part, chunk, chunk, torch.ones_like(chunk), torch.tensor(1.0))
            acc += part
        assert torch.equal(acc, pooled)


def test_all_pairs_pruned_edge_case():
    """A token whose every non-boundary pair is skipped must still get the boundary eps."""
    torch.manual_seed(0)
    E, H, K = 8, 16, 2
    weight = torch.randn(E, H, device=DEVICE) / math.sqrt(H)
    # One token, scores widely separated except at the boundary.
    logits = torch.tensor([[8.0, 7.0, 0.1, 0.09, -4.0, -5.0, -6.0, -7.0]], device=DEVICE)
    hidden = torch.randn(1, H, device=DEVICE)
    eps_h, eps_b, _, _ = compute_router_sensitivity(
        hidden, logits, weight, None, K, 'sigmoid'
    )
    assert torch.isfinite(eps_h).all() and eps_h.item() > 0
    ref_h, ref_b = _reference_eps(
        hidden.cpu(), logits.cpu(), weight.cpu(), None, K, 'sigmoid', 1.0
    )
    assert eps_h.item() == pytest.approx(float(ref_h), rel=1e-3)
