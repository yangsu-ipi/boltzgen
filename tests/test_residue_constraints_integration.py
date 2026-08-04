"""End-to-end parsing of residue constraints through YamlDesignParser.

Unlike the other constraint tests, these run the real parser over a real
structure file, which is the only way to cover the wiring inside `parse_file`:
constraints have to survive include masking and stay aligned with the design
mask, which is built by an independent code path.

Requires the molecule dictionary (`mols.zip`). It is looked up in the local
HuggingFace cache — the same artifact `boltzgen download moldir` fetches — or
via the BOLTZGEN_MOLDIR environment variable, and the tests skip when it is
absent so the suite still runs offline.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from boltzgen.data import const

EXAMPLE_DIR = Path(__file__).parent.parent / "example"
STRUCTURE = EXAMPLE_DIR / "7rpz.cif"
CANONICAL = const.canonical_tokens


def _find_moldir():
    """Locate mols.zip without hitting the network, or return None."""
    env = os.environ.get("BOLTZGEN_MOLDIR")
    if env and Path(env).exists():
        return Path(env)

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None

    cached = try_to_load_from_cache(
        "boltzgen/inference-data", "mols.zip", repo_type="dataset"
    )
    if isinstance(cached, str) and Path(cached).exists():
        return Path(cached)
    return None


MOLDIR = _find_moldir()

pytestmark = [
    pytest.mark.skipif(
        MOLDIR is None,
        reason="mols.zip not cached; run `boltzgen download moldir` to enable",
    ),
    pytest.mark.skipif(
        not STRUCTURE.exists(), reason=f"{STRUCTURE.name} not available"
    ),
]

# Residues of the loaded chain that are both redesigned and constrained.
FILE_RANGE = "20..27"
FILE_RANGE_LEN = 8


@pytest.fixture(scope="module")
def parsed(tmp_path_factory):
    """Parse a spec that constrains both a protein entity and a file entity."""
    from boltzgen.data.mol import load_canonicals
    from boltzgen.data.parse.schema import YamlDesignParser

    spec = {
        "entities": [
            {
                "protein": {
                    "id": "P",
                    "sequence": 6,
                    "residue_constraints": [
                        {"position": 1, "allowed": "H"},
                        {"position": "2..3", "disallowed": "C"},
                        {"position": 4, "disallowed": "W", "weight": 2.0},
                    ],
                }
            },
            {
                "file": {
                    # Absolute path: file references resolve relative to the yaml
                    "path": str(STRUCTURE),
                    "include": [{"chain": {"id": "A"}}],
                    "design": [{"chain": {"id": "A", "res_index": FILE_RANGE}}],
                    "residue_constraints": [
                        {"chain": {"id": "A", "res_index": FILE_RANGE, "disallowed": "C"}},
                        {"chain": {"id": "A", "res_index": 24, "allowed": "H"}},
                        {
                            "chain": {
                                "id": "A",
                                "res_index": "25..27",
                                "allowed": "AGS",
                                "weight": 1.0,
                            }
                        },
                    ],
                }
            },
        ]
    }

    path = tmp_path_factory.mktemp("spec") / "constraints.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))

    mols = load_canonicals(moldir=MOLDIR)
    parser = YamlDesignParser(mol_dir=MOLDIR)
    target = parser.parse_yaml(path, mols=mols, mol_dir=MOLDIR)
    return target.design_info


def _rows(array) -> set:
    return set(np.where((array != 0).any(axis=1))[0].tolist())


def test_protein_entity_constraints_land_on_their_positions(parsed):
    hard, soft = parsed.res_aa_constraint_mask, parsed.res_aa_soft_bias

    # The protein entity is first, so its 6 residues are rows 0-5.
    assert hard[0, CANONICAL.index("HIS")] == 0.0
    assert hard[0].sum() == len(CANONICAL) - 1  # only HIS allowed
    assert hard[1, CANONICAL.index("CYS")] == 1.0
    assert hard[2, CANONICAL.index("CYS")] == 1.0
    assert soft[3, CANONICAL.index("TRP")] == -2.0
    assert hard[3].sum() == 0.0  # position 4 is soft only
    assert hard[4:6].sum() == 0.0 and soft[4:6].sum() == 0.0


def test_file_entity_constraints_align_with_the_design_mask(parsed):
    """The strong check: constrained rows must be exactly the designed file rows.

    The spec redesigns and constrains the same residue range, and the design
    mask is built independently of the constraint arrays. 7rpz.cif also has
    leading unresolved residues that include masking removes, so an off-by-N
    misalignment would show up here as a mismatch.
    """
    hard, soft = parsed.res_aa_constraint_mask, parsed.res_aa_soft_bias
    designed = set(np.where(parsed.res_design_mask.astype(bool))[0].tolist())

    protein_rows = set(range(6))
    file_designed = designed - protein_rows
    assert len(file_designed) == FILE_RANGE_LEN

    cys_blocked = set(
        np.where(hard[:, CANONICAL.index("CYS")] != 0)[0].tolist()
    ) - protein_rows
    assert cys_blocked == file_designed

    # The narrower entries sit inside that same range.
    assert _rows(soft) - protein_rows <= file_designed
    assert _rows(hard) - protein_rows == file_designed


def test_file_entity_hard_and_soft_entries_are_applied(parsed):
    hard, soft = parsed.res_aa_constraint_mask, parsed.res_aa_soft_bias
    protein_rows = set(range(6))

    # Exactly one file row (res_index 24) allows only HIS.
    his_only = [
        r
        for r in _rows(hard) - protein_rows
        if hard[r].sum() == len(CANONICAL) - 1
        and hard[r, CANONICAL.index("HIS")] == 0.0
    ]
    assert len(his_only) == 1

    # Exactly three file rows (res_index 25..27) carry the soft AGS preference.
    soft_rows = sorted(_rows(soft) - protein_rows)
    assert len(soft_rows) == 3
    for r in soft_rows:
        free = {CANONICAL[i] for i in np.where(soft[r] == 0)[0]}
        assert free == {"ALA", "GLY", "SER"}
        assert set(soft[r][soft[r] != 0].tolist()) == {-1.0}


def test_no_constraints_leak_onto_non_designed_residues(parsed):
    hard, soft = parsed.res_aa_constraint_mask, parsed.res_aa_soft_bias
    constrained = (hard != 0).any(axis=1) | (soft != 0).any(axis=1)
    assert not constrained[~parsed.res_design_mask.astype(bool)].any()


def test_arrays_are_residue_aligned_and_well_formed(parsed):
    num_res = len(parsed.res_design_mask)
    for array in (parsed.res_aa_constraint_mask, parsed.res_aa_soft_bias):
        assert array.shape == (num_res, len(CANONICAL))
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
    # No designed position may be fully hard-blocked.
    assert not (parsed.res_aa_constraint_mask != 0).all(axis=1).any()


def test_example_spec_parses_with_its_constraints(tmp_path):
    """The shipped example spec parses and produces both kinds of constraint."""
    from boltzgen.data.mol import load_canonicals
    from boltzgen.data.parse.schema import YamlDesignParser

    mols = load_canonicals(moldir=MOLDIR)
    parser = YamlDesignParser(mol_dir=MOLDIR)
    target = parser.parse_yaml(
        EXAMPLE_DIR / "residue_constraints_test.yaml", mols=mols, mol_dir=MOLDIR
    )

    di = target.design_info
    assert (di.res_aa_constraint_mask != 0).any(), "expected hard constraints"
    assert (di.res_aa_soft_bias != 0).any(), "expected soft constraints"
    assert len(di.res_aa_soft_bias) == len(di.res_design_mask)
