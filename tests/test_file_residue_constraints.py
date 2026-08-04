"""Unit tests for per-residue amino acid constraints on `file` entities.

File entities address positions per chain (`chain: {id, res_index}`) instead of
with the `position` key used by `protein` entities, because they span several
chains and their residue indices come from the file.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from boltzgen.data import const
from boltzgen.data.data import Chain
from boltzgen.data.parse.schema import parse_file_residue_constraints

CANONICAL = const.canonical_tokens
LETTER_MAP = const.prot_letter_to_token
NUM_AA = len(CANONICAL)


def _structure(chains: list[tuple[str, int, int]]):
    """Build a structure stub with the given (name, res_idx, res_num) chains.

    Only `chains` is consulted by the parser, so a namespace is enough.
    """
    records = []
    for asym_id, (name, res_idx, res_num) in enumerate(chains):
        record = np.zeros(1, dtype=Chain)[0]
        record["name"] = name
        record["mol_type"] = const.chain_type_ids["PROTEIN"]
        record["asym_id"] = asym_id
        record["entity_id"] = asym_id
        record["res_idx"] = res_idx
        record["res_num"] = res_num
        records.append(record)
    return SimpleNamespace(chains=np.array(records, dtype=Chain))


def _parse(spec, chains=None, num_res=20):
    structure = _structure(chains or [("A", 0, 10), ("B", 10, 10)])
    return parse_file_residue_constraints(
        spec,
        structure=structure,
        num_res=num_res,
        canonical_tokens=CANONICAL,
        prot_letter_to_token=LETTER_MAP,
        path="test.cif",
    )


# ============================================================================
# Position resolution
# ============================================================================


def test_res_index_is_resolved_within_the_chain():
    """res_index is relative to the chain, so chain B is offset by its start."""
    spec = [{"chain": {"id": "B", "res_index": "1..3", "disallowed": "C"}}]
    mask, bias = _parse(spec)

    cys_idx = CANONICAL.index("CYS")
    # Chain B starts at residue 10, so its residues 1-3 are rows 10-12.
    assert mask[10:13, cys_idx].tolist() == [1.0, 1.0, 1.0]
    assert mask[:10].sum() == 0.0  # chain A untouched
    assert mask[13:].sum() == 0.0
    assert np.all(bias == 0.0)


def test_single_res_index():
    spec = [{"chain": {"id": "A", "res_index": 4, "allowed": "H"}}]
    mask, _ = _parse(spec)

    his_idx = CANONICAL.index("HIS")
    assert mask[3, his_idx] == 0.0
    assert mask[3].sum() == NUM_AA - 1
    assert mask[[0, 1, 2, 4]].sum() == 0.0


def test_omitted_res_index_covers_whole_chain():
    spec = [{"chain": {"id": "A", "disallowed": "C"}}]
    mask, _ = _parse(spec)

    cys_idx = CANONICAL.index("CYS")
    assert mask[:10, cys_idx].sum() == 10.0
    assert mask[10:, cys_idx].sum() == 0.0


def test_res_index_all_covers_whole_chain():
    spec = [{"chain": {"id": "B", "res_index": "all", "disallowed": "M"}}]
    mask, _ = _parse(spec)

    met_idx = CANONICAL.index("MET")
    assert mask[10:, met_idx].sum() == 10.0
    assert mask[:10, met_idx].sum() == 0.0


def test_multiple_chains_are_independent():
    spec = [
        {"chain": {"id": "A", "res_index": 1, "allowed": "A"}},
        {"chain": {"id": "B", "res_index": 1, "allowed": "G"}},
    ]
    mask, _ = _parse(spec)

    assert mask[0, CANONICAL.index("ALA")] == 0.0
    assert mask[0].sum() == NUM_AA - 1
    assert mask[10, CANONICAL.index("GLY")] == 0.0
    assert mask[10].sum() == NUM_AA - 1


# ============================================================================
# Hard / soft semantics match the protein entity path
# ============================================================================


def test_soft_constraint_on_file_entity():
    spec = [{"chain": {"id": "A", "res_index": "2..3", "disallowed": "W", "weight": 2.0}}]
    mask, bias = _parse(spec)

    trp_idx = CANONICAL.index("TRP")
    assert np.all(mask == 0.0)  # nothing hard-blocked
    assert bias[1, trp_idx] == -2.0
    assert bias[2, trp_idx] == -2.0
    assert bias[0].sum() == 0.0
    assert bias[3:].sum() == 0.0


def test_soft_whitelist_on_file_entity():
    spec = [{"chain": {"id": "A", "res_index": 1, "allowed": "AG", "weight": 1.5}}]
    mask, bias = _parse(spec)

    assert np.all(mask == 0.0)
    assert bias[0, CANONICAL.index("ALA")] == 0.0
    assert bias[0, CANONICAL.index("GLY")] == 0.0
    assert (bias[0] == -1.5).sum() == NUM_AA - 2


def test_hard_and_soft_compose_on_file_entity():
    spec = [
        {"chain": {"id": "A", "res_index": 1, "disallowed": "C"}},
        {"chain": {"id": "A", "res_index": 1, "disallowed": "M", "weight": 1.0}},
    ]
    mask, bias = _parse(spec)

    assert mask[0, CANONICAL.index("CYS")] == 1.0
    assert mask[0, CANONICAL.index("MET")] == 0.0
    assert bias[0, CANONICAL.index("MET")] == -1.0


def test_overlapping_hard_whitelists_intersect():
    spec = [
        {"chain": {"id": "A", "res_index": "1..5", "allowed": "AG"}},
        {"chain": {"id": "A", "res_index": "3..8", "allowed": "GS"}},
    ]
    mask, _ = _parse(spec)

    gly_idx = CANONICAL.index("GLY")
    # Overlap region keeps only GLY
    for pos in [2, 3, 4]:
        assert mask[pos, gly_idx] == 0.0
        assert mask[pos].sum() == NUM_AA - 1
    # Non-overlapping regions keep both of their own allowed residues
    assert mask[0].sum() == NUM_AA - 2
    assert mask[5].sum() == NUM_AA - 2


def test_shape_and_dtype():
    mask, bias = _parse([], num_res=20)
    assert mask.shape == (20, NUM_AA)
    assert bias.shape == (20, NUM_AA)
    assert mask.dtype == np.float32
    assert bias.dtype == np.float32
    assert mask.sum() == 0.0
    assert bias.sum() == 0.0


# ============================================================================
# Error paths
# ============================================================================


def test_missing_chain_id_raises():
    spec = [{"chain": {"res_index": 1, "allowed": "A"}}]
    with pytest.raises(ValueError, match="missing 'id'"):
        _parse(spec)


def test_unknown_chain_id_raises():
    spec = [{"chain": {"id": "Z", "res_index": 1, "allowed": "A"}}]
    with pytest.raises(ValueError, match="not in file"):
        _parse(spec)


def test_non_protein_chain_raises():
    structure = _structure([("A", 0, 10)])
    structure.chains["mol_type"][0] = const.chain_type_ids["DNA"]
    spec = [{"chain": {"id": "A", "res_index": 1, "allowed": "A"}}]
    with pytest.raises(ValueError, match="not a protein chain"):
        parse_file_residue_constraints(
            spec,
            structure=structure,
            num_res=10,
            canonical_tokens=CANONICAL,
            prot_letter_to_token=LETTER_MAP,
            path="test.cif",
        )


def test_both_allowed_and_disallowed_raises():
    spec = [{"chain": {"id": "A", "res_index": 1, "allowed": "A", "disallowed": "C"}}]
    with pytest.raises(ValueError, match="cannot specify both"):
        _parse(spec)


def test_neither_allowed_nor_disallowed_raises():
    spec = [{"chain": {"id": "A", "res_index": 1}}]
    with pytest.raises(ValueError, match="must specify either"):
        _parse(spec)


def test_invalid_weight_raises():
    spec = [{"chain": {"id": "A", "res_index": 1, "disallowed": "C", "weight": 0}}]
    with pytest.raises(ValueError, match="must be positive"):
        _parse(spec)


def test_invalid_amino_acid_raises():
    spec = [{"chain": {"id": "A", "res_index": 1, "allowed": "X"}}]
    with pytest.raises(ValueError, match="Unknown amino acid"):
        _parse(spec)


def test_error_messages_name_the_chain_and_file():
    spec = [{"chain": {"id": "A", "res_index": 1, "disallowed": "C", "weight": -1}}]
    with pytest.raises(ValueError, match="chain A in file test.cif"):
        _parse(spec)
