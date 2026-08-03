"""End-to-end tests for constrained inverse-folding sampling.

These build a minimal `InverseFoldingDecoder` with zero decoder layers so the
predictor (zero-initialised) emits uniform logits. Every sampling decision is
then driven purely by the per-position hard and soft constraints, which is
exactly what these tests exercise.
"""

import pytest

torch = pytest.importorskip("torch")

from boltzgen.data import const
from boltzgen.model.modules.inverse_fold import InverseFoldingDecoder

NUM_AA = len(const.canonical_tokens)
NODE_DIM = 4


def _decoder(**kwargs) -> InverseFoldingDecoder:
    """Build a decoder whose logits are uniform, so only constraints matter."""
    defaults = dict(
        atom_s=1,
        atom_z=1,
        token_s=1,
        token_z=1,
        node_dim=NODE_DIM,
        pair_dim=NODE_DIM,
        hidden_dim=NODE_DIM,
        num_decoder_layers=0,
        sampling_temperature=None,  # argmax decoding
        tie_symmetric_sequences=False,
    )
    defaults.update(kwargs)
    return InverseFoldingDecoder(**defaults).eval()


def _inputs(
    num_tokens: int = 2,
    symmetric_group: list[int] = None,
    residue_index: list[int] = None,
):
    """Build the minimal (s, z, edge_idx, valid_mask, feats) sampling inputs."""
    s = torch.zeros(num_tokens, NODE_DIM)
    # A fully connected graph over the tokens (including self edges).
    src, dst = torch.meshgrid(
        torch.arange(num_tokens), torch.arange(num_tokens), indexing="ij"
    )
    edge_idx = torch.stack([src.flatten(), dst.flatten()], dim=0)
    z = torch.zeros(edge_idx.shape[1], NODE_DIM)
    valid_mask = torch.ones(1, num_tokens, dtype=torch.bool)

    if symmetric_group is None:
        symmetric_group = [0] * num_tokens
    if residue_index is None:
        residue_index = list(range(num_tokens))

    feats = {
        "design_mask": torch.ones(1, num_tokens, dtype=torch.bool),
        "res_type_clone": torch.zeros(1, num_tokens, const.num_tokens),
        "aa_constraint_mask": torch.zeros(1, num_tokens, NUM_AA),
        "aa_soft_bias": torch.zeros(1, num_tokens, NUM_AA),
        "symmetric_group": torch.tensor([symmetric_group], dtype=torch.long),
        "feature_residue_index": torch.tensor([residue_index], dtype=torch.long),
        "coords": torch.zeros(1, num_tokens, 3),
    }
    return s, z, edge_idx, valid_mask, feats


def _block_all_but(feats, position: int, tokens: list[str]) -> None:
    """Hard-block every amino acid except `tokens` at `position`."""
    feats["aa_constraint_mask"][0, position, :] = 1.0
    for token in tokens:
        feats["aa_constraint_mask"][0, position, const.canonical_tokens.index(token)] = 0.0


def _penalise_all_but(feats, position: int, tokens: list[str], weight: float) -> None:
    """Softly penalise every amino acid except `tokens` at `position`."""
    feats["aa_soft_bias"][0, position, :] -= weight
    for token in tokens:
        feats["aa_soft_bias"][0, position, const.canonical_tokens.index(token)] += weight


def _sampled_tokens(out) -> list[str]:
    ids = out["res_type"][0].argmax(dim=-1).tolist()
    return [const.tokens[i] for i in ids]


def test_hard_constraints_are_position_specific() -> None:
    """Different positions get different forced residues."""
    s, z, edge_idx, valid_mask, feats = _inputs(num_tokens=3)
    _block_all_but(feats, 0, ["TRP"])
    _block_all_but(feats, 1, ["HIS"])
    # Position 2 is unconstrained.

    out = _decoder().sample(s, z, edge_idx, valid_mask, feats)
    sampled = _sampled_tokens(out)

    assert sampled[0] == "TRP"
    assert sampled[1] == "HIS"
    assert sampled[2] in const.canonical_tokens


def test_soft_bias_steers_the_choice() -> None:
    """With uniform logits, a soft preference decides the argmax."""
    s, z, edge_idx, valid_mask, feats = _inputs(num_tokens=2)
    _penalise_all_but(feats, 0, ["MET"], weight=1.0)
    _penalise_all_but(feats, 1, ["GLU"], weight=0.5)

    out = _decoder().sample(s, z, edge_idx, valid_mask, feats)

    assert _sampled_tokens(out) == ["MET", "GLU"]


def test_hard_constraint_wins_over_conflicting_soft_bias() -> None:
    """A soft preference can never override a hard block."""
    s, z, edge_idx, valid_mask, feats = _inputs(num_tokens=1)
    _block_all_but(feats, 0, ["PRO"])
    _penalise_all_but(feats, 0, ["CYS"], weight=5.0)  # wants CYS, which is blocked

    out = _decoder().sample(s, z, edge_idx, valid_mask, feats)

    assert _sampled_tokens(out) == ["PRO"]


def test_soft_bias_leaves_penalised_residues_reachable() -> None:
    """Unlike a hard block, a soft penalty still allows the residue to appear."""
    torch.manual_seed(0)
    decoder = _decoder(sampling_temperature=1.0)  # stochastic decoding

    seen = set()
    for _ in range(100):
        s, z, edge_idx, valid_mask, feats = _inputs(num_tokens=1)
        _penalise_all_but(feats, 0, ["CYS"], weight=1.0)
        out = decoder.sample(s, z, edge_idx, valid_mask, feats)
        seen.update(_sampled_tokens(out))

    # The favoured residue shows up, and so do penalised ones.
    assert "CYS" in seen
    assert seen - {"CYS"}


def test_hard_constraint_is_never_violated_when_sampling() -> None:
    torch.manual_seed(0)
    decoder = _decoder(sampling_temperature=1.0)

    seen = set()
    for _ in range(100):
        s, z, edge_idx, valid_mask, feats = _inputs(num_tokens=1)
        _block_all_but(feats, 0, ["ALA", "GLY"])
        out = decoder.sample(s, z, edge_idx, valid_mask, feats)
        seen.update(_sampled_tokens(out))

    assert seen <= {"ALA", "GLY"}
    assert len(seen) == 2  # both remain reachable


def test_tied_positions_combine_their_constraints() -> None:
    """Regression: constraints of every tied position must be honoured.

    The two positions are tied (same symmetric group and residue index), so
    they are sampled once. Position 0 allows {MET, TRP} and position 1 allows
    {TRP, ALA}, leaving TRP as the only residue valid for both — even though a
    soft bias makes TRP the least attractive choice at each position on its own.
    Sampling from either position's mask alone would therefore yield MET or ALA.
    """
    s, z, edge_idx, valid_mask, feats = _inputs(
        num_tokens=2, symmetric_group=[1, 1], residue_index=[0, 0]
    )
    _block_all_but(feats, 0, ["MET", "TRP"])
    _block_all_but(feats, 1, ["TRP", "ALA"])
    _penalise_all_but(feats, 0, ["MET", "ALA"], weight=1.0)
    _penalise_all_but(feats, 1, ["MET", "ALA"], weight=1.0)

    out = _decoder(tie_symmetric_sequences=True).sample(
        s, z, edge_idx, valid_mask, feats
    )

    assert _sampled_tokens(out) == ["TRP", "TRP"]


def test_tied_positions_with_conflicting_constraints_warn_and_stay_valid() -> None:
    """Disjoint allowed sets across tied positions must not deadlock sampling."""
    s, z, edge_idx, valid_mask, feats = _inputs(
        num_tokens=2, symmetric_group=[1, 1], residue_index=[0, 0]
    )
    _block_all_but(feats, 0, ["MET"])
    _block_all_but(feats, 1, ["ALA"])  # no residue satisfies both

    with pytest.warns(RuntimeWarning, match="block every amino acid"):
        out = _decoder(tie_symmetric_sequences=True).sample(
            s, z, edge_idx, valid_mask, feats
        )

    sampled = _sampled_tokens(out)
    assert sampled[0] == sampled[1]  # still tied
    assert sampled[0] in {"MET", "ALA"}  # one group's constraint is satisfied


def test_soft_bias_absent_from_feats_is_a_no_op() -> None:
    """Older inputs without an aa_soft_bias feature still sample normally."""
    s, z, edge_idx, valid_mask, feats = _inputs(num_tokens=2)
    del feats["aa_soft_bias"]
    _block_all_but(feats, 0, ["TYR"])

    out = _decoder().sample(s, z, edge_idx, valid_mask, feats)

    assert _sampled_tokens(out)[0] == "TYR"
