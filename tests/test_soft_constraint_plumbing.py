"""Tests for the per-residue soft-constraint data plumbing.

Covers the token/feature plumbing that carries `aa_soft_bias` from the
parsed YAML down to the inverse-folding sampler.
"""

from dataclasses import fields

import numpy as np
import pytest

from boltzgen.data import const
from boltzgen.data.data import DesignInfo, Token
from boltzgen.data.tokenize.tokenizer import TokenData, tokendata_to_tuple

NUM_AA = len(const.canonical_tokens)


def _design_info(
    num_res: int = 4,
    design: bool = True,
    constraint_mask: np.ndarray = None,
    soft_bias: np.ndarray = None,
) -> DesignInfo:
    """Build a minimal DesignInfo for validation tests."""
    zeros = np.zeros((num_res, NUM_AA), dtype=np.float32)
    return DesignInfo(
        res_design_mask=np.full(num_res, design, dtype=bool),
        res_structure_groups=np.zeros(num_res, dtype=np.int32),
        res_ss_types=np.zeros(num_res, dtype=np.int32),
        res_binding_type=np.zeros(num_res, dtype=np.int32),
        res_aa_constraint_mask=zeros if constraint_mask is None else constraint_mask,
        res_aa_soft_bias=zeros.copy() if soft_bias is None else soft_bias,
    )


# ============================================================================
# Token record layout
# ============================================================================


def test_tokendata_fields_align_with_token_dtype():
    """TokenData is serialised positionally, so field order must match the dtype."""
    td_names = [f.name for f in fields(TokenData)]
    dtype_names = [name for name, _ in Token]

    assert len(td_names) == len(dtype_names)
    # `design` is stored as `design_mask`; every other field shares its name.
    assert [n for n in td_names if n != "design"] == [
        n for n in dtype_names if n != "design_mask"
    ]
    assert td_names.index("aa_soft_bias") == dtype_names.index("aa_soft_bias")


def test_token_dtype_round_trips_soft_bias():
    """A soft bias written into a Token record survives the structured array."""
    token = TokenData(
        token_idx=0,
        atom_idx=0,
        atom_num=1,
        res_idx=0,
        res_type=0,
        res_name="ALA",
        sym_id=0,
        asym_id=0,
        entity_id=0,
        mol_type=0,
        center_idx=0,
        disto_idx=0,
        center_coords=np.zeros(3, dtype=np.float32),
        disto_coords=np.zeros(3, dtype=np.float32),
        resolved_mask=True,
        disto_mask=True,
        modified=False,
        frame_rot=np.eye(3, dtype=np.float32).flatten(),
        frame_t=np.zeros(3, dtype=np.float32),
        frame_mask=True,
        cyclic_period=0,
        is_standard=True,
        design=True,
        binding_type=0,
        structure_group=0,
        aa_constraint_mask=np.zeros(NUM_AA, dtype=np.float32),
        aa_soft_bias=np.zeros(NUM_AA, dtype=np.float32),
        ccd=np.zeros(5, dtype=np.int32),
        target_msa_mask=False,
        design_ss_mask=False,
        feature_asym_id=0,
        feature_res_idx=0,
        symmetric_group=0,
    )
    cys_idx = const.canonical_tokens.index("CYS")
    token.aa_soft_bias[cys_idx] = -2.0
    token.aa_constraint_mask[const.canonical_tokens.index("MET")] = 1.0

    tokens = np.array([tokendata_to_tuple(token)], dtype=Token)

    assert tokens["aa_soft_bias"].shape == (1, NUM_AA)
    assert tokens["aa_soft_bias"][0, cys_idx] == -2.0
    assert tokens["aa_soft_bias"][0].sum() == -2.0
    # The hard mask is unaffected by the soft bias and vice versa.
    assert tokens["aa_constraint_mask"][0, cys_idx] == 0.0
    assert tokens["aa_constraint_mask"][0].sum() == 1.0


# ============================================================================
# DesignInfo validation
# ============================================================================


def test_design_info_accepts_soft_bias():
    soft_bias = np.zeros((4, NUM_AA), dtype=np.float32)
    soft_bias[1, const.canonical_tokens.index("CYS")] = -2.0
    assert DesignInfo.is_valid(_design_info(soft_bias=soft_bias))


def test_design_info_allows_all_amino_acids_softly_penalised():
    """Soft constraints can never make a position unsatisfiable."""
    soft_bias = np.full((4, NUM_AA), -5.0, dtype=np.float32)
    assert DesignInfo.is_valid(_design_info(soft_bias=soft_bias))


def test_design_info_rejects_all_amino_acids_hard_blocked():
    constraint_mask = np.zeros((4, NUM_AA), dtype=np.float32)
    constraint_mask[2, :] = 1.0
    with pytest.raises(ValueError, match="all amino acids disallowed"):
        DesignInfo.is_valid(_design_info(constraint_mask=constraint_mask))


def test_design_info_warns_on_soft_bias_for_non_designed_residues():
    soft_bias = np.zeros((4, NUM_AA), dtype=np.float32)
    soft_bias[0, const.canonical_tokens.index("CYS")] = -1.0
    with pytest.warns(UserWarning, match="non-designed residues"):
        DesignInfo.is_valid(_design_info(design=False, soft_bias=soft_bias))
