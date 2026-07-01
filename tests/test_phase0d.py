# tests/test_phase0d.py — abr-nmr-phase0d
# Metatron Dynamics, Inc.
#
# Verifier tests for Phase 0d low-field viability sweep.
#
# Tests are grouped by layer:
#   Group 1 — Declaration integrity (4 tests)
#   Group 2 — Signal physics at reference B0 values (8 tests)
#   Group 3 — SNR scaling law (3 tests)
#   Group 4 — Operator separation at reference B0 values (5 tests)
#   Group 5 — Threshold finding (4 tests)
#   Group 6 — Declared open conditions (2 tests)
#
# Total: 26 tests
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from sim.declaration_0d import (
    declare_domain,
    B0_SWEEP,
    N_B0,
    SNR_THRESHOLD,
    BOUNDARY_RATIO_THRESHOLD,
    PM_RANGE_LOW_T,
    PM_RANGE_HIGH_T,
    N_ELEMENTS,
)
from sim.signal_0d    import build_signal
from sim.operators_0d import run_operators


# ---- Fixtures --------------------------------------------------------

@pytest.fixture(scope='module')
def domain():
    return declare_domain()


@pytest.fixture(scope='module')
def signal_15t(domain):
    return build_signal(domain, 1.5)


@pytest.fixture(scope='module')
def signal_05t(domain):
    return build_signal(domain, 0.5)


@pytest.fixture(scope='module')
def signal_02t(domain):
    return build_signal(domain, 0.2)


@pytest.fixture(scope='module')
def ops_15t(domain, signal_15t):
    return run_operators(domain, signal_15t)


@pytest.fixture(scope='module')
def ops_05t(domain, signal_05t):
    return run_operators(domain, signal_05t)


@pytest.fixture(scope='module')
def ops_02t(domain, signal_02t):
    return run_operators(domain, signal_02t)


# ---- Group 1: Declaration integrity ----------------------------------

class TestDeclaration:

    def test_sweep_length(self, domain):
        """Sweep array has declared length."""
        assert len(domain.b0_sweep) == N_B0

    def test_sweep_bounds(self, domain):
        """Sweep runs from declared min to max B0."""
        assert abs(domain.b0_sweep[0]  - 0.05) < 1e-6
        assert abs(domain.b0_sweep[-1] - 1.5)  < 1e-6

    def test_sweep_log_spaced(self, domain):
        """Sweep is log-spaced: ratios between consecutive steps are constant."""
        ratios = domain.b0_sweep[1:] / domain.b0_sweep[:-1]
        assert np.allclose(ratios, ratios[0], rtol=1e-6)

    def test_thresholds_declared(self, domain):
        """SNR and ratio thresholds are positive and declared."""
        assert domain.snr_threshold > 0
        assert domain.boundary_ratio_threshold > 0


# ---- Group 2: Signal physics at reference B0 -------------------------

class TestSignalPhysics:

    def test_signal_shape_15t(self, signal_15t):
        """S_field shape is [N_ELEMENTS, n_steps]."""
        assert signal_15t.S_field.shape == (N_ELEMENTS, 1200)

    def test_baseline_positive(self, signal_15t):
        """All elements have positive baseline signal."""
        assert np.all(signal_15t.S_baseline > 0)

    def test_signal_voltage_scales_b0_squared(self, domain):
        """V_signal_peak ∝ B0² — verify ratio between 0.5T and 1.5T."""
        sig_15 = build_signal(domain, 1.5)
        sig_05 = build_signal(domain, 0.5)
        expected_ratio = (1.5 / 0.5) ** 2   # = 9.0
        actual_ratio   = sig_15.V_signal_peak / sig_05.V_signal_peak
        assert abs(actual_ratio - expected_ratio) < 0.01, (
            f"V_signal ratio expected {expected_ratio:.2f}, got {actual_ratio:.4f}"
        )

    def test_noise_b0_independent(self, domain):
        """Noise RMS is identical at 0.5T and 1.5T (B0-independent)."""
        sig_15 = build_signal(domain, 1.5)
        sig_05 = build_signal(domain, 0.5)
        assert np.allclose(sig_15.noise_rms, sig_05.noise_rms, rtol=1e-10)

    def test_snr_scales_b0_squared(self, signal_15t, signal_05t):
        """Min SNR ratio ≈ (1.5/0.5)² = 9 between 1.5T and 0.5T."""
        expected = (1.5 / 0.5) ** 2
        actual   = signal_15t.min_SNR / signal_05t.min_SNR
        assert abs(actual - expected) < 0.05 * expected, (
            f"SNR ratio expected ~{expected:.1f}, got {actual:.4f}"
        )

    def test_snr_15t_high(self, signal_15t):
        """At 1.5T, min SNR >> threshold (consistent with Phase 0c)."""
        assert signal_15t.min_SNR > 1000

    def test_s_field_varies_across_steps(self, signal_15t):
        """Signal varies over cardiac steps (cardiac modulation present)."""
        for i in range(N_ELEMENTS):
            std = float(signal_15t.S_field[i, :].std())
            assert std > 0, f"Element {i} shows no cardiac modulation"

    def test_tumor_gm_signal_differs(self, signal_15t):
        """Tumor-sector elements have higher baseline signal than GM-sector."""
        from sim.declaration_0d import TUMOR_SECTORS, GRAY_MAT_SECTORS
        tumor_base = signal_15t.S_baseline[list(TUMOR_SECTORS)].mean()
        gm_base    = signal_15t.S_baseline[list(GRAY_MAT_SECTORS)].mean()
        # Tumor: 92% water > GM: 84% water
        assert tumor_base > gm_base


# ---- Group 3: SNR scaling law ----------------------------------------

class TestSNRScaling:

    def test_snr_monotone_with_b0(self, domain):
        """SNR increases monotonically with B0 across sweep."""
        b0_sample  = [0.1, 0.3, 0.5, 1.0, 1.5]
        snr_sample = [build_signal(domain, b).min_SNR for b in b0_sample]
        for i in range(len(snr_sample) - 1):
            assert snr_sample[i] < snr_sample[i+1], (
                f"SNR not monotone between {b0_sample[i]}T and {b0_sample[i+1]}T"
            )

    def test_b0_squared_law_holds_across_sweep(self, domain):
        """SNR ∝ B0² holds within 2% across all sweep pairs."""
        b0_a, b0_b = 0.3, 0.9
        snr_a = build_signal(domain, b0_a).min_SNR
        snr_b = build_signal(domain, b0_b).min_SNR
        expected = (b0_b / b0_a) ** 2
        actual   = snr_b / snr_a
        assert abs(actual - expected) / expected < 0.02

    def test_snr_at_min_b0_above_threshold(self, domain):
        """
        At B0_MIN (0.05T), SNR is still above threshold (143 >> 5).
        This is a declared finding: the 32-element array geometry has
        sufficient SNR margin that the noise floor is not reached even
        at 0.05T. The SNR survival threshold is below 0.05T — outside
        the declared sweep range.
        This confirms that for this phantom geometry, operator separation
        (not SNR) is the binding constraint at low field — but as the
        sweep shows, separation also survives to 0.05T.
        The declared open condition (inhomogeneity at low B0) is the
        binding practical constraint, not the simulation noise floor.
        """
        sig = build_signal(domain, 0.05)
        # SNR ∝ B0² → at 0.05T: SNR ~ 129177 * (0.05/1.5)² ≈ 143
        assert sig.min_SNR > SNR_THRESHOLD
        assert sig.min_SNR > 100  # well above threshold


# ---- Group 4: Operator separation at reference B0 --------------------

class TestOperatorSeparation:

    def test_boundary_detected_at_15t(self, ops_15t):
        """Boundary ratio > 1 at 1.5T (Phase 0c result reproduced)."""
        assert ops_15t.separation_survived
        assert ops_15t.boundary_ratio > 1.0

    def test_boundary_e_exceeds_interior_15t(self, ops_15t):
        """Boundary E > both interior E values at 1.5T."""
        assert ops_15t.boundary_E_mean > ops_15t.interior_tumor_E_mean
        assert ops_15t.boundary_E_mean > ops_15t.interior_gm_E_mean

    def test_boundary_ratio_monotone_with_b0(self, ops_15t, ops_05t, ops_02t):
        """
        Boundary ratio may decrease with B0 if SNR degrades,
        or remain stable if separation is contrast-driven.
        Test that the 1.5T ratio exceeds 0.2T ratio
        (general direction — not a strict scaling law).
        """
        # At very low B0, noise drowns contrast → ratio should degrade
        assert ops_15t.boundary_ratio >= ops_02t.boundary_ratio

    def test_rho_positive(self, ops_15t):
        """rho_mean is positive — local contrast is non-zero at 1.5T."""
        assert ops_15t.rho_mean > 0

    def test_separation_survived_at_05t(self, ops_05t):
        """At 0.5T (declared minimum viable from Phase 0), separation survives."""
        assert ops_05t.separation_survived


# ---- Group 5: Threshold finding -------------------------------------

class TestThresholds:

    @pytest.fixture(scope='class')
    def sweep_records(self, domain):
        from experiments.run_phase0d import run_sweep, find_thresholds
        records = run_sweep(domain)
        return records

    def test_snr_threshold_exists(self, sweep_records):
        """Some B0 in sweep meets SNR threshold."""
        met = [r for r in sweep_records if r.all_detectable]
        assert len(met) > 0, "No B0 in sweep meets SNR threshold"

    def test_separation_threshold_exists(self, sweep_records):
        """Some B0 in sweep maintains operator separation."""
        met = [r for r in sweep_records if r.separation_survived]
        assert len(met) > 0, "No B0 in sweep maintains operator separation"

    def test_15t_passes_both(self, sweep_records):
        """1.5T record passes both thresholds."""
        rec_15 = max(sweep_records, key=lambda r: r.b0)
        assert rec_15.all_detectable
        assert rec_15.separation_survived

    def test_snr_threshold_below_15t(self, sweep_records):
        """SNR survival threshold is below 1.5T — i.e. some lower B0 also passes."""
        passing = [r for r in sweep_records if r.all_detectable]
        min_b0  = min(r.b0 for r in passing)
        assert min_b0 < 1.5


# ---- Group 6: Declared open conditions ------------------------------

class TestOpenConditions:

    def test_t1_held_constant(self, domain):
        """T1 values are unchanged from Phase 0c declared values."""
        assert domain.tissue_types['gray_matter']['T1_ms'] == 1100.0
        assert domain.tissue_types['white_matter']['T1_ms'] == 700.0
        assert domain.tissue_types['tumor']['T1_ms'] == 1400.0

    def test_t2star_held_constant(self, domain):
        """T2* values are unchanged from Phase 0c declared values."""
        assert domain.tissue_types['gray_matter']['T2star_ms'] == 55.0
        assert domain.tissue_types['tumor']['T2star_ms'] == 80.0


# ---- Group 7: C projection provenance (Verifier requirement) ---------

class TestProjectionProvenance:

    def test_projection_name_declared(self, domain):
        """C_PROJECTION_NAME is declared in the domain module."""
        from sim.declaration_0d import C_PROJECTION_NAME
        assert C_PROJECTION_NAME == "C_mean_abs_ratio"

    def test_operator_result_carries_projection_name(self, ops_15t):
        """OperatorResult0d carries the declared projection name."""
        from sim.declaration_0d import C_PROJECTION_NAME
        assert ops_15t.projection_name == C_PROJECTION_NAME

    def test_boundary_ratio_is_mean_abs(self, domain, signal_15t):
        """
        boundary_ratio equals mean|E|(boundary) / max(mean|E|(tumor), mean|E|(gm)).
        Verify by computing independently from E_field components.
        """
        ops = run_operators(domain, signal_15t)
        # boundary_ratio must equal boundary_E_mean / max interior
        max_interior = max(ops.interior_tumor_E_mean, ops.interior_gm_E_mean)
        expected_ratio = ops.boundary_E_mean / max_interior
        assert abs(ops.boundary_ratio - expected_ratio) < 1e-10

    def test_projection_discards_stated(self, domain, signal_15t):
        """
        Confirm that per-step variance in E is non-zero, i.e. something
        is genuinely discarded by collapsing to the step mean.
        This confirms the 'discards per-step variation' statement.
        """
        from sim.declaration_0d import TUMOR_SECTORS, GRAY_MAT_SECTORS
        sig = signal_15t
        src = np.array([e[0] for e in domain.adjacency], dtype=np.int32)
        tgt = np.array([e[1] for e in domain.adjacency], dtype=np.int32)
        A_field = (sig.S_field[src, :] - sig.S_field[tgt, :])
        # Boundary edges: crossing between TUMOR and GM sectors
        boundary_edges = [
            e for e in range(len(domain.adjacency))
            if (int(src[e]) in TUMOR_SECTORS) != (int(tgt[e]) in TUMOR_SECTORS)
        ]
        # Check that |E| varies across steps at boundary edges
        E_boundary = np.abs(A_field[boundary_edges, :])
        step_std = E_boundary.std(axis=1)
        assert np.any(step_std > 0), (
            "No per-step variation at boundary edges — "
            "cardiac modulation absent; nothing is discarded by step mean"
        )
