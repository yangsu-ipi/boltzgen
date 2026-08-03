"""Integration tests for inverse-folding constraint mask composition."""

import pytest

torch = pytest.importorskip("torch")

from boltzgen.data import const
from boltzgen.model.modules.inverse_fold import (
    aggregate_tied_constraints,
    build_constraint_logit_mask,
    build_soft_logit_bias,
)


INF = 10**6


def _allowed_only_mask(allowed_tokens: list[str]) -> torch.Tensor:
    """Build a single-row mask where only `allowed_tokens` are permitted."""
    num_aa = len(const.canonical_tokens)
    mask = torch.ones((1, num_aa), dtype=torch.float32)
    for token in allowed_tokens:
        mask[0, const.canonical_tokens.index(token)] = 0.0
    return mask


def test_conflict_allowed_and_global_avoid_keeps_global_restriction() -> None:
    cys_idx = const.canonical_tokens.index("CYS")
    aa_constraint_mask = _allowed_only_mask(["CYS"])

    with pytest.warns(RuntimeWarning, match="Relaxing per-residue constraints"):
        out = build_constraint_logit_mask(
            num_nodes=1,
            aa_constraint_mask=aa_constraint_mask,
            inverse_fold_restriction=["CYS"],
            canonical_tokens=const.canonical_tokens,
            inf=INF,
            device=torch.device("cpu"),
        )

    # Global avoid must still block CYS after conflict handling.
    assert out[0, cys_idx].item() == -INF
    # All other residues remain available.
    assert (out[0] == 0).sum().item() == len(const.canonical_tokens) - 1


def test_non_conflicting_constraints_compose_correctly() -> None:
    ala_idx = const.canonical_tokens.index("ALA")
    cys_idx = const.canonical_tokens.index("CYS")
    aa_constraint_mask = _allowed_only_mask(["ALA"])

    out = build_constraint_logit_mask(
        num_nodes=1,
        aa_constraint_mask=aa_constraint_mask,
        inverse_fold_restriction=["CYS"],
        canonical_tokens=const.canonical_tokens,
        inf=INF,
        device=torch.device("cpu"),
    )

    # Only ALA should remain available.
    assert out[0, ala_idx].item() == 0.0
    assert out[0, cys_idx].item() == -INF
    assert (out[0] == 0).sum().item() == 1


def test_global_restrictions_that_block_all_raise() -> None:
    with pytest.raises(ValueError, match="no valid amino acids"):
        build_constraint_logit_mask(
            num_nodes=1,
            aa_constraint_mask=None,
            inverse_fold_restriction=const.canonical_tokens,
            canonical_tokens=const.canonical_tokens,
            inf=INF,
            device=torch.device("cpu"),
        )


def test_shape_mismatch_ignores_per_residue_mask() -> None:
    bad_shape = torch.zeros((2, 20), dtype=torch.float32)

    with pytest.warns(RuntimeWarning, match="shape mismatch"):
        out = build_constraint_logit_mask(
            num_nodes=1,
            aa_constraint_mask=bad_shape,
            inverse_fold_restriction=[],
            canonical_tokens=const.canonical_tokens,
            inf=INF,
            device=torch.device("cpu"),
        )

    # No restrictions should remain after ignoring mismatched input.
    assert out.shape == (1, len(const.canonical_tokens))
    assert torch.all(out == 0)


# ============================================================================
# build_soft_logit_bias
# ============================================================================


def test_soft_bias_none_returns_zeros() -> None:
    out = build_soft_logit_bias(
        num_nodes=3,
        aa_soft_bias=None,
        canonical_tokens=const.canonical_tokens,
        device=torch.device("cpu"),
    )
    assert out.shape == (3, len(const.canonical_tokens))
    assert torch.all(out == 0)


def test_soft_bias_passes_through_finite_values() -> None:
    cys_idx = const.canonical_tokens.index("CYS")
    bias = torch.zeros((2, len(const.canonical_tokens)), dtype=torch.float32)
    bias[0, cys_idx] = -2.0

    out = build_soft_logit_bias(
        num_nodes=2,
        aa_soft_bias=bias,
        canonical_tokens=const.canonical_tokens,
        device=torch.device("cpu"),
    )

    assert out[0, cys_idx].item() == -2.0
    assert out[0].sum().item() == -2.0
    assert torch.all(out[1] == 0)


def test_soft_bias_shape_mismatch_ignored() -> None:
    bad_shape = torch.zeros((2, len(const.canonical_tokens)), dtype=torch.float32)

    with pytest.warns(RuntimeWarning, match="shape mismatch"):
        out = build_soft_logit_bias(
            num_nodes=1,
            aa_soft_bias=bad_shape,
            canonical_tokens=const.canonical_tokens,
            device=torch.device("cpu"),
        )

    assert out.shape == (1, len(const.canonical_tokens))
    assert torch.all(out == 0)


def test_soft_bias_non_finite_values_are_zeroed() -> None:
    cys_idx = const.canonical_tokens.index("CYS")
    met_idx = const.canonical_tokens.index("MET")
    bias = torch.zeros((1, len(const.canonical_tokens)), dtype=torch.float32)
    bias[0, cys_idx] = -float("inf")
    bias[0, met_idx] = -1.0

    with pytest.warns(RuntimeWarning, match="non-finite"):
        out = build_soft_logit_bias(
            num_nodes=1,
            aa_soft_bias=bias,
            canonical_tokens=const.canonical_tokens,
            device=torch.device("cpu"),
        )

    # A soft constraint must never become a hard block.
    assert out[0, cys_idx].item() == 0.0
    assert out[0, met_idx].item() == -1.0


def test_soft_bias_cannot_zero_out_a_position() -> None:
    """A position with every amino acid penalised still has a valid distribution."""
    num_aa = len(const.canonical_tokens)
    bias = torch.full((1, num_aa), -5.0, dtype=torch.float32)

    hard = build_constraint_logit_mask(
        num_nodes=1,
        aa_constraint_mask=None,
        inverse_fold_restriction=[],
        canonical_tokens=const.canonical_tokens,
        inf=INF,
        device=torch.device("cpu"),
    )
    soft = build_soft_logit_bias(
        num_nodes=1,
        aa_soft_bias=bias,
        canonical_tokens=const.canonical_tokens,
        device=torch.device("cpu"),
    )

    probs = torch.softmax(torch.zeros(1, num_aa) + hard + soft, dim=-1)
    assert torch.isfinite(probs).all()
    assert probs.sum().item() == pytest.approx(1.0)
    assert (probs > 0).all()


@pytest.mark.parametrize("temperature", [0.1, 1.0, 2.0])
def test_soft_bias_strength_is_temperature_independent(temperature: float) -> None:
    """A weight of w shifts the sampled log-odds by exactly w, at any temperature."""
    num_aa = len(const.canonical_tokens)
    cys_idx = const.canonical_tokens.index("CYS")
    weight = 2.0

    bias = torch.zeros((1, num_aa), dtype=torch.float32)
    bias[0, cys_idx] = -weight
    soft = build_soft_logit_bias(
        num_nodes=1,
        aa_soft_bias=bias,
        canonical_tokens=const.canonical_tokens,
        device=torch.device("cpu"),
        sampling_temperature=temperature,
    )

    # Arbitrary but fixed logits, sampled the way the decoder does.
    logits = torch.linspace(-1.0, 1.0, num_aa)[None]
    unbiased = torch.softmax(logits / temperature, dim=-1)
    biased = torch.softmax((logits + soft) / temperature, dim=-1)

    # Log-odds of CYS against any other residue drop by exactly `weight`.
    other_idx = const.canonical_tokens.index("ALA")
    unbiased_log_odds = torch.log(unbiased[0, cys_idx] / unbiased[0, other_idx])
    biased_log_odds = torch.log(biased[0, cys_idx] / biased[0, other_idx])
    assert (unbiased_log_odds - biased_log_odds).item() == pytest.approx(
        weight, abs=1e-4
    )
    # Still reachable: soft means less likely, not impossible.
    assert biased[0, cys_idx].item() > 0


def test_soft_bias_unscaled_for_argmax_decoding() -> None:
    """With argmax decoding (temperature None) the bias stays in logit units."""
    cys_idx = const.canonical_tokens.index("CYS")
    bias = torch.zeros((1, len(const.canonical_tokens)), dtype=torch.float32)
    bias[0, cys_idx] = -2.0

    out = build_soft_logit_bias(
        num_nodes=1,
        aa_soft_bias=bias,
        canonical_tokens=const.canonical_tokens,
        device=torch.device("cpu"),
        sampling_temperature=None,
    )

    assert out[0, cys_idx].item() == -2.0


def test_hard_constraint_is_unsamplable_at_low_temperature() -> None:
    """A hard block gives probability exactly 0, unlike any soft bias."""
    num_aa = len(const.canonical_tokens)
    cys_idx = const.canonical_tokens.index("CYS")
    temperature = 0.1

    hard = build_constraint_logit_mask(
        num_nodes=1,
        aa_constraint_mask=None,
        inverse_fold_restriction=["CYS"],
        canonical_tokens=const.canonical_tokens,
        inf=INF,
        device=torch.device("cpu"),
    )

    # Logits that strongly favour the blocked residue.
    logits = torch.zeros(1, num_aa)
    logits[0, cys_idx] = 20.0
    probs = torch.softmax((logits + hard) / temperature, dim=-1)

    assert probs[0, cys_idx].item() == 0.0
    assert torch.isfinite(probs).all()
    assert probs.sum().item() == pytest.approx(1.0)


# ============================================================================
# aggregate_tied_constraints
# ============================================================================


def _hard_mask_rows(blocked_per_row: list[list[str]]) -> torch.Tensor:
    """Build a hard logit mask where the listed tokens are blocked per row."""
    num_aa = len(const.canonical_tokens)
    mask = torch.zeros((len(blocked_per_row), num_aa), dtype=torch.float32)
    for row, tokens in enumerate(blocked_per_row):
        for token in tokens:
            mask[row, const.canonical_tokens.index(token)] = -INF
    return mask


def test_tying_takes_union_of_blocked_residues() -> None:
    """A residue blocked at any tied position is blocked for the whole group."""
    cys_idx = const.canonical_tokens.index("CYS")
    met_idx = const.canonical_tokens.index("MET")
    trp_idx = const.canonical_tokens.index("TRP")
    hard = _hard_mask_rows([["CYS"], ["MET"], ["TRP"]])
    soft = torch.zeros_like(hard)

    out_hard, out_soft = aggregate_tied_constraints(hard, soft, [0, 1])

    assert out_hard.shape == (1, len(const.canonical_tokens))
    assert out_hard[0, cys_idx].item() == -INF  # from position 0
    assert out_hard[0, met_idx].item() == -INF  # from position 1
    assert out_hard[0, trp_idx].item() == 0.0  # position 2 is not in the group
    assert torch.all(out_soft == 0)


def test_tying_averages_soft_bias() -> None:
    cys_idx = const.canonical_tokens.index("CYS")
    hard = torch.zeros((2, len(const.canonical_tokens)), dtype=torch.float32)
    soft = torch.zeros_like(hard)
    soft[0, cys_idx] = -3.0  # penalised at one of the two tied positions

    _, out_soft = aggregate_tied_constraints(hard, soft, [0, 1])

    assert out_soft[0, cys_idx].item() == pytest.approx(-1.5)


def test_tying_single_position_is_identity() -> None:
    hard = _hard_mask_rows([["CYS"], ["MET"]])
    soft = torch.zeros_like(hard)
    soft[1] = -1.0

    out_hard, out_soft = aggregate_tied_constraints(hard, soft, [1])

    assert torch.equal(out_hard[0], hard[1])
    assert torch.equal(out_soft[0], soft[1])


def test_tying_conflicting_constraints_fall_back_to_first_position() -> None:
    """Disjoint allowed sets across tied positions must not block everything."""
    num_aa = len(const.canonical_tokens)
    ala_idx = const.canonical_tokens.index("ALA")
    gly_idx = const.canonical_tokens.index("GLY")
    # Row 0 allows only ALA, row 1 allows only GLY -> the union blocks all 20.
    hard = torch.full((2, num_aa), -float(INF), dtype=torch.float32)
    hard[0, ala_idx] = 0.0
    hard[1, gly_idx] = 0.0
    soft = torch.zeros_like(hard)

    with pytest.warns(RuntimeWarning, match="block every amino acid"):
        out_hard, _ = aggregate_tied_constraints(hard, soft, [0, 1])

    # Falls back to the first position's constraints, keeping ALA available.
    assert out_hard[0, ala_idx].item() == 0.0
    assert (out_hard[0] == 0).sum().item() == 1
