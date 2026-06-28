# tests/test_phase0b.py — abr-nmr-phase0
# Metatron Dynamics, Inc.
#
# Verifier tests for Phase 0b: 8-element array, multi-tissue phantom.
#
# Tests confirm:
#   - Domain declaration: element geometry, adjacency, reverse edge
#   - Signal model: per-element values follow declared tissue mix formulas
#   - Operator A: directed difference, antisymmetry
#   - Operator ρ: bounded, non-negative, derived from full gradient
#   - E field: boundary edges exceed interior edges
#   - Boundary detection verdict: DETECTED at declared parameters
#
# All assertions derive from declared formulas and constants.
# No statistical operations. No heuristics.
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
import pytest
from sim.declaration_0b import (
    declare_domain,
    N_ELEMENTS,
    TISSUE_TYPES,
    TUMOR_SECTORS,
    GRAY_MAT_SECTORS,
    TR_S,
    TE_S,
    FLIP_ANGLE_RAD,
    N_CARDIAC_STEPS,
    CARDIAC_PHASE_ADVANCE_RAD,
)
from sim.signal_0b    import build_signal
from sim.operators_0b import run_operators


# ---- Fixtures --------------------------------------------------------

@pytest.fixture(scope="module")
def domain():
    return declare_domain()

@pytest.fixture(scope="module")
def signal(domain):
    return build_signal(domain)

@pytest.fixture(scope="module")
def result(domain, signal):
    return run_operators(domain, signal, rho_base=0.3)


# ---- Domain declaration tests ----------------------------------------

class TestDomain0b:

    def test_n_elements(self, domain):
        assert domain.n_elements == N_ELEMENTS == 8

    def test_n_cardiac_steps(self, domain):
        assert domain.n_cardiac_steps == N_CARDIAC_STEPS

    def test_adjacency_count(self, domain):
        """16 directed edges: 8 clockwise + 8 counterclockwise."""
        assert len(domain.adjacency) == 2 * N_ELEMENTS

    def test_reverse_edge_involution(self, domain):
        """reverse_edge[reverse_edge[e]] == e for all e."""
        rev = domain.reverse_edge
        for e in range(len(rev)):
            assert rev[rev[e]] == e, (
                f"reverse_edge is not an involution at e={e}"
            )

    def test_reverse_edge_flips_direction(self, domain):
        """reverse of (src→tgt) is (tgt→src)."""
        adj = domain.adjacency
        rev = domain.reverse_edge
        for e in range(len(adj)):
            src, tgt   = adj[e]
            rev_src, rev_tgt = adj[rev[e]]
            assert rev_src == tgt and rev_tgt == src, (
                f"Edge {e}: reverse does not flip direction."
            )

    def test_no_self_loops(self, domain):
        for e, (src, tgt) in enumerate(domain.adjacency):
            assert src != tgt, f"Self-loop at edge {e}"

    def test_element_angles(self, domain):
        """Elements are evenly spaced at 45° intervals."""
        for el in domain.elements:
            expected_angle = el.element_id * 45.0
            assert abs(el.angle_center_deg - expected_angle) < 1e-6

    def test_tissue_mix_sums_to_one(self, domain):
        """Volume fractions in each element sum to 1.0."""
        for el in domain.elements:
            total = sum(el.tissue_mix.values())
            assert abs(total - 1.0) < 1e-4, (
                f"Element {el.element_id} tissue mix sums to {total:.6f}"
            )

    def test_tumor_sectors_declared(self, domain):
        """Elements 0-3 are tumor sectors."""
        for el in domain.elements:
            if el.element_id in TUMOR_SECTORS:
                assert el.tissue_mix.get('tumor', 0) > 0
                assert el.tissue_mix.get('gray_matter', 0) == 0.0
            else:
                assert el.tissue_mix.get('gray_matter', 0) > 0
                assert el.tissue_mix.get('tumor', 0) == 0.0

    def test_clockwise_ring_topology(self, domain):
        """
        Clockwise edges: edge i goes from element i to element (i+1)%N.
        """
        N = N_ELEMENTS
        for i in range(N):
            src, tgt = domain.adjacency[i]
            assert src == i
            assert tgt == (i + 1) % N


# ---- Signal tests ----------------------------------------------------

class TestSignal0b:

    def test_S_shape(self, domain, signal):
        assert signal.S.shape == (N_ELEMENTS, N_CARDIAC_STEPS)

    def test_S_baseline_shape(self, domain, signal):
        assert signal.S_baseline.shape == (N_ELEMENTS,)

    def test_S_baseline_positive(self, signal):
        assert np.all(signal.S_baseline > 0.0)

    def test_tumor_elements_higher_signal(self, signal):
        """
        Tumor has higher water content (92%) than gray matter (84%).
        Tumor-sector elements should have higher S_baseline.
        """
        tumor_mean = float(np.array([
            signal.S_baseline[i] for i in TUMOR_SECTORS
        ]).mean())
        gm_mean = float(np.array([
            signal.S_baseline[i] for i in GRAY_MAT_SECTORS
        ]).mean())
        assert tumor_mean > gm_mean, (
            f"Tumor S_baseline ({tumor_mean:.6f}) <= "
            f"GM S_baseline ({gm_mean:.6f}). "
            "Higher water content should produce higher signal."
        )

    def test_signal_contrast_from_water_content(self, signal):
        """
        Signal difference between tumor and GM sectors
        reflects declared water content difference (92% vs 84%).
        The contrast should be non-trivial — at least 1%.
        """
        tumor_mean = float(np.array([
            signal.S_baseline[i] for i in TUMOR_SECTORS
        ]).mean())
        gm_mean = float(np.array([
            signal.S_baseline[i] for i in GRAY_MAT_SECTORS
        ]).mean())
        contrast_pct = (tumor_mean - gm_mean) / gm_mean * 100
        assert contrast_pct >= 1.0, (
            f"Signal contrast {contrast_pct:.2f}% < 1%. "
            "Expected meaningful contrast from 8% water content difference."
        )

    def test_SNR_all_elements_detectable(self, signal):
        """All elements must have SNR >= 5 in single step."""
        assert np.all(signal.SNR >= 5.0), (
            f"Some elements below SNR=5. Min SNR: {signal.SNR.min():.1f}"
        )

    def test_uniform_elements_within_hemisphere(self, signal):
        """
        All tumor-sector elements have identical declared tissue mix
        and therefore identical S_baseline and SNR.
        """
        tumor_baselines = [signal.S_baseline[i] for i in TUMOR_SECTORS]
        assert np.allclose(tumor_baselines, tumor_baselines[0], rtol=1e-6), (
            "Tumor-sector elements have non-identical S_baseline. "
            "Declared tissue mix should be identical for all tumor sectors."
        )
        gm_baselines = [signal.S_baseline[i] for i in GRAY_MAT_SECTORS]
        assert np.allclose(gm_baselines, gm_baselines[0], rtol=1e-6)

    def test_cardiac_phase_formula(self, signal):
        """phi(t) = (t * CARDIAC_PHASE_ADVANCE_RAD) mod 2pi."""
        advance  = float(CARDIAC_PHASE_ADVANCE_RAD)
        t_idx    = np.arange(N_CARDIAC_STEPS, dtype=np.float32)
        expected = (
            (t_idx * advance).astype(np.float64) % (2.0 * np.pi)
        ).astype(np.float32).astype(np.float64)
        assert np.allclose(signal.cardiac_phase, expected, atol=1e-4)

    def test_vascular_tissues_modulate(self, signal):
        """
        Vascular tissues (WM, GM, tumor) show cardiac modulation.
        Avascular tissue (CSF) shows no modulation.
        """
        csf_signal = signal.S_tissue['CSF']
        csf_max    = float(csf_signal.max())
        csf_min    = float(csf_signal.min())
        assert csf_max == csf_min, (
            "CSF signal varies — declared as avascular, should be static."
        )
        tumor_signal = signal.S_tissue['tumor']
        assert float(tumor_signal.max()) != float(tumor_signal.min()), (
            "Tumor signal does not vary — declared as vascular, should modulate."
        )

    def test_tumor_modulates_more_than_gm(self, signal):
        """
        Tumor has higher vascular fraction (8%) than GM (4%).
        Tumor modulation depth should exceed GM modulation depth.
        """
        tumor_mod = signal.modulation_depth[list(TUMOR_SECTORS)[0]]
        gm_mod    = signal.modulation_depth[list(GRAY_MAT_SECTORS)[0]]
        assert tumor_mod > gm_mod, (
            f"Tumor modulation ({tumor_mod:.4f}) <= GM modulation ({gm_mod:.4f}). "
            "Higher vascular fraction should produce more modulation."
        )


# ---- Operator tests --------------------------------------------------

class TestOperators0b:

    def test_A_shape(self, result):
        assert result.A_field.shape == (2 * N_ELEMENTS, N_CARDIAC_STEPS)

    def test_A_antisymmetric(self, domain, result):
        """
        A[e] = S[src] - S[tgt].
        A[reverse(e)] = S[tgt] - S[src] = -A[e].
        """
        rev = domain.reverse_edge
        for e in range(len(domain.adjacency)):
            check = result.A_field[e, :] + result.A_field[rev[e], :]
            assert np.allclose(check, 0.0, atol=1e-12), (
                f"A not antisymmetric at edge {e}. "
                f"Max |A[e]+A[rev(e)]|: {np.abs(check).max():.4e}"
            )

    def test_rho_shape(self, result):
        assert result.rho.shape == (N_ELEMENTS,)

    def test_rho_non_negative(self, result):
        assert np.all(result.rho >= 0.0)

    def test_rho_bounded_by_rho_base(self, result):
        """rho[v] < rho_base for all v."""
        assert np.all(result.rho < 0.3 + 1e-6), (
            f"rho exceeds rho_base=0.3. Max: {result.rho.max():.6f}"
        )

    def test_E_shape(self, result):
        assert result.E_field.shape == (2 * N_ELEMENTS, N_CARDIAC_STEPS)

    def test_boundary_edges_declared(self, result):
        """
        Boundary edges must include edges 3 and 7
        (element 3→4 and element 7→0 cross the tumor/GM boundary).
        """
        assert 3 in result.boundary_edges, (
            "Edge 3 (3→4) should be a boundary edge."
        )
        assert 7 in result.boundary_edges, (
            "Edge 7 (7→0) should be a boundary edge."
        )

    def test_boundary_E_exceeds_interior(self, result):
        """
        The primary declared test: boundary |E| > interior |E|.
        The operators find tissue boundaries as elevated relational contrast.
        """
        bnd_mean = float(result.boundary_E_mean.mean())
        tum_mean = float(result.interior_tumor_E_mean.mean())
        gm_mean  = float(result.interior_gm_E_mean.mean())

        assert bnd_mean > tum_mean, (
            f"Boundary E ({bnd_mean:.4e}) <= tumor interior E ({tum_mean:.4e}). "
            "Boundary not detected."
        )
        assert bnd_mean > gm_mean, (
            f"Boundary E ({bnd_mean:.4e}) <= GM interior E ({gm_mean:.4e}). "
            "Boundary not detected."
        )

    def test_boundary_contrast_ratio(self, result):
        """
        Boundary E / interior E >= 2.0.
        Declared from simulation result: ratio = 2.99.
        A ratio >= 2 confirms meaningful boundary detection.
        """
        bnd_mean = float(result.boundary_E_mean.mean())
        tum_mean = float(result.interior_tumor_E_mean.mean())
        ratio    = bnd_mean / (tum_mean + 1e-20)
        assert ratio >= 2.0, (
            f"Boundary contrast ratio {ratio:.2f} < 2.0. "
            "Boundary contrast insufficient."
        )

    def test_interior_tumor_equals_interior_gm(self, result):
        """
        Interior tumor and GM edges have the same |E|.
        The signal difference is at the boundary, not within hemispheres.
        This confirms the operators are finding the boundary specifically,
        not a general elevation across one hemisphere.
        """
        tum_mean = float(result.interior_tumor_E_mean.mean())
        gm_mean  = float(result.interior_gm_E_mean.mean())
        assert abs(tum_mean - gm_mean) / (tum_mean + 1e-20) < 0.01, (
            f"Interior tumor E ({tum_mean:.4e}) != "
            f"interior GM E ({gm_mean:.4e}). "
            "Expected equal interior values — boundary is the differentiator."
        )

    def test_E_varies_over_steps(self, result):
        """
        E field varies over cardiac steps at boundary edges.
        Cardiac modulation drives temporal variation in the relational field.
        """
        for e in result.boundary_edges:
            e_max = float(result.E_field[e, :].max())
            e_min = float(result.E_field[e, :].min())
            assert e_max != e_min, (
                f"E field at boundary edge {e} is constant over steps. "
                "Cardiac modulation not reaching E field."
            )
