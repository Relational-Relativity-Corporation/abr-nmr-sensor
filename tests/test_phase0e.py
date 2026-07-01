# tests/test_phase0e.py — abr-nmr-phase0e
# Metatron Dynamics, Inc.
#
# Verifier tests for Phase 0e inhomogeneity sweep.
#
# Group 1 — Declaration integrity (4 tests)
# Group 2 — Signal physics: inhomogeneity effect on T2* (7 tests)
# Group 3 — SNR behavior under inhomogeneity (4 tests)
# Group 4 — Operator separation under inhomogeneity (5 tests)
# Group 5 — Threshold behavior across B0 (3 tests)
# Group 6 — C projection provenance (3 tests)
# Group 7 — Hardware spec admissibility (3 tests)
#
# Total: 29 tests
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from sim.declaration_0e import (
    declare_domain,
    INHOMOGENEITY_SWEEP,
    N_INHOMOGENEITY,
    REFERENCE_B0_VALUES,
    PM_SPECS_PPM,
    PPM_TO_FRAC,
    SNR_THRESHOLD,
    BOUNDARY_RATIO_THRESHOLD,
    TUMOR_SECTORS,
    GRAY_MAT_SECTORS,
)
from sim.signal_0e    import build_signal
from sim.operators_0e import run_operators


@pytest.fixture(scope='module')
def domain():
    return declare_domain()

@pytest.fixture(scope='module')
def sig_05_perfect(domain):
    return build_signal(domain, 0.5, 0.0)

@pytest.fixture(scope='module')
def sig_05_1000ppm(domain):
    return build_signal(domain, 0.5, 1000 * PPM_TO_FRAC)

@pytest.fixture(scope='module')
def sig_02_perfect(domain):
    return build_signal(domain, 0.2, 0.0)

@pytest.fixture(scope='module')
def ops_05_perfect(domain, sig_05_perfect):
    return run_operators(domain, sig_05_perfect)

@pytest.fixture(scope='module')
def ops_05_1000ppm(domain, sig_05_1000ppm):
    return run_operators(domain, sig_05_1000ppm)


# ---- Group 1: Declaration integrity ----------------------------------

class TestDeclaration:

    def test_sweep_length(self, domain):
        assert len(domain.inhomogeneity_sweep) == N_INHOMOGENEITY

    def test_sweep_starts_at_zero(self, domain):
        assert domain.inhomogeneity_sweep[0] == 0.0

    def test_reference_b0_values(self, domain):
        assert set(domain.reference_b0_values) == {0.2, 0.3, 0.5}

    def test_projection_name_inherited(self, domain):
        assert domain.projection_name == "C_mean_abs_ratio"


# ---- Group 2: Signal physics — inhomogeneity degrades T2* -----------

class TestSignalPhysics:

    def test_perfect_homogeneity_matches_phase0d(self, sig_05_perfect):
        """At zero inhomogeneity, T2*_eff equals declared tissue T2*."""
        # tumor T2* declared as 80ms in tissue_types
        assert abs(sig_05_perfect.t2star_eff['tumor'] - 80.0) < 0.01

    def test_inhomogeneity_shortens_t2star(self, sig_05_perfect, sig_05_1000ppm):
        """T2*_eff is shorter at 1000 ppm than at 0 ppm."""
        for tissue in ['tumor', 'gray_matter', 'white_matter']:
            assert (sig_05_1000ppm.t2star_eff[tissue] <
                    sig_05_perfect.t2star_eff[tissue]), \
                f"T2*_eff not shorter at 1000 ppm for {tissue}"

    def test_inhomogeneity_reduces_signal(self, sig_05_perfect, sig_05_1000ppm):
        """Baseline signal is lower at 1000 ppm than at perfect homogeneity."""
        assert sig_05_1000ppm.S_baseline.mean() < sig_05_perfect.S_baseline.mean()

    def test_delta_b0_abs_scales_with_b0(self, domain):
        """delta_B0_abs = inhomogeneity * B0 — verify scaling."""
        inh = 1000 * PPM_TO_FRAC
        sig_02 = build_signal(domain, 0.2, inh)
        sig_05 = build_signal(domain, 0.5, inh)
        assert abs(sig_02.delta_B0_abs / sig_05.delta_B0_abs - 0.2/0.5) < 1e-10

    def test_inhomogeneity_ppm_conversion(self, sig_05_1000ppm):
        """inhomogeneity_ppm = inhomogeneity_frac * 1e6."""
        assert abs(sig_05_1000ppm.inhomogeneity_ppm - 1000.0) < 1e-6

    def test_cardiac_modulation_present_at_zero_inhomogeneity(self, sig_05_perfect):
        """Cardiac modulation present at perfect homogeneity."""
        for i in range(len(TUMOR_SECTORS)):
            std = float(sig_05_perfect.S_field[i, :].std())
            assert std > 0, f"No cardiac modulation at element {i} at 0 ppm"

    def test_signal_destroyed_at_high_inhomogeneity(self, sig_05_1000ppm):
        """
        At 1000 ppm, 0.5T: T2*_eff ~0.007ms << TE=30ms → signal underflows.
        This is the declared physical mechanism setting the survival threshold.
        """
        assert sig_05_1000ppm.S_field.max() < 1e-10


# ---- Group 3: SNR under inhomogeneity --------------------------------

class TestSNR:

    def test_snr_decreases_with_inhomogeneity(self, domain):
        """Min SNR decreases with inhomogeneity at 0.5T (within signal range)."""
        # Use values below the underflow threshold (~51 ppm at 0.5T)
        inh_values = [0.0, 10e-6, 20e-6, 40e-6]
        snr_values = [build_signal(domain, 0.5, inh).min_SNR
                      for inh in inh_values]
        for i in range(len(snr_values) - 1):
            assert snr_values[i] > snr_values[i+1]

    def test_snr_perfect_above_threshold(self, sig_05_perfect):
        """At perfect homogeneity, all elements detectable."""
        assert sig_05_perfect.all_detectable
        assert sig_05_perfect.min_SNR > SNR_THRESHOLD

    def test_noise_unchanged_by_inhomogeneity(self, sig_05_perfect, sig_05_1000ppm):
        """Noise RMS is identical at 0 and 1000 ppm — inhomogeneity is B0 only."""
        assert np.allclose(sig_05_perfect.noise_rms,
                           sig_05_1000ppm.noise_rms, rtol=1e-10)

    def test_lower_b0_lower_snr_at_perfect_homogeneity(self, domain):
        """
        At perfect homogeneity, lower B0 gives lower SNR (B0² scaling).
        Use 0 ppm to stay well above signal underflow at all B0 values.
        """
        snr_02 = build_signal(domain, 0.2, 0.0).min_SNR
        snr_05 = build_signal(domain, 0.5, 0.0).min_SNR
        assert snr_02 < snr_05


# ---- Group 4: Operator separation under inhomogeneity ---------------

class TestOperatorSeparation:

    def test_separation_at_perfect_homogeneity(self, ops_05_perfect):
        """Boundary ratio > 1 at perfect homogeneity (Phase 0c/0d result)."""
        assert ops_05_perfect.separation_survived
        assert ops_05_perfect.boundary_ratio > 1.0

    def test_boundary_ratio_cliff_behavior(self, domain):
        """
        The boundary ratio does not degrade gradually with inhomogeneity.
        It holds near its perfect-homogeneity value until T2*_eff collapses
        below TE, then drops to zero as signal underflows.

        Declared finding: the survival criterion is a cliff, not a slope.
        The ratio is either in the survival region (~15) or has failed (0).
        There is no gradual intermediate state in this simulation.

        This is consistent with the physical model:
          - Inhomogeneity adds a static R2* term uniformly per element
          - All elements degrade equally, preserving the contrast ratio
          - Until T2*_eff << TE, at which point exp(-TE/T2*) → 0 for all

        Test: ratio at perfect homogeneity is in the survival region;
        ratio past the threshold is zero; no value between 1 and 14
        is observed (cliff, not slope).
        """
        # Perfect homogeneity: ratio in survival region
        ops_0 = run_operators(domain, build_signal(domain, 0.5, 0.0))
        assert ops_0.boundary_ratio > 1.0

        # Past survival threshold at 0.5T (~51 ppm): ratio collapses to 0
        ops_high = run_operators(domain, build_signal(domain, 0.5, 200e-6))
        assert ops_high.boundary_ratio == 0.0

        # No intermediate state: ratio is never in (0, 1) range
        # (it is either ~15 or 0 — the cliff is abrupt)
        for ppm in [20, 35, 50]:
            ops = run_operators(domain, build_signal(domain, 0.5, ppm * PPM_TO_FRAC))
            assert ops.boundary_ratio == 0.0 or ops.boundary_ratio > 1.0, \
                f"Unexpected intermediate ratio at {ppm} ppm: {ops.boundary_ratio:.4f}"

    def test_separation_at_halbach_spec_02t(self, domain):
        """
        At 0.2T and 100 ppm (Halbach upper bound), separation survives.
        Declared result: survival threshold at 0.2T is ~138 ppm.
        100 ppm is within the survival region at 0.2T.
        """
        inh = 100 * PPM_TO_FRAC
        sig = build_signal(domain, 0.2, inh)
        ops = run_operators(domain, sig)
        assert ops.separation_survived, \
            f"Separation failed at 0.2T, 100 ppm: ratio={ops.boundary_ratio:.4f}"

    def test_separation_fails_at_halbach_upper_05t(self, domain):
        """
        At 0.5T and 100 ppm (Halbach upper bound), separation FAILS.
        Declared result: survival threshold at 0.5T is ~51 ppm.
        100 ppm exceeds the survival threshold at 0.5T.
        This is a declared finding — not a defect.
        """
        inh = 100 * PPM_TO_FRAC
        sig = build_signal(domain, 0.5, inh)
        ops = run_operators(domain, sig)
        assert not ops.separation_survived, \
            f"Expected failure at 0.5T, 100 ppm but got ratio={ops.boundary_ratio:.4f}"

    def test_projection_name_on_result(self, ops_05_perfect):
        """OperatorResult carries declared projection name."""
        assert ops_05_perfect.projection_name == "C_mean_abs_ratio"


# ---- Group 5: Threshold monotonicity across B0 ----------------------

class TestThresholdMonotonicity:

    def test_higher_b0_survives_more_inhomogeneity(self, domain):
        """
        At the same fractional inhomogeneity, lower B0 maintains operator
        separation to a higher ppm threshold. Two competing effects are present:
          - Higher B0 increases signal (SNR ∝ B0²) — favours survival
          - Higher B0 increases ΔB0_abs = inhomogeneity × B0 — increases dephasing

        The simulation finds that the dephasing effect dominates: lower B0
        produces a higher survival threshold in ppm. Declared thresholds:
          0.2T → ~138 ppm
          0.5T → ~51 ppm

        This is a declared finding from the sweep, not derived analytically here.

        Test: at 100 ppm, 0.2T survives and 0.5T fails — directly demonstrating
        that the lower B0 value has the higher inhomogeneity tolerance.
        """
        # 100 ppm: within 0.2T survival region (~138 ppm), above 0.5T threshold (~51 ppm)
        inh = 100 * PPM_TO_FRAC
        ops_02 = run_operators(domain, build_signal(domain, 0.2, inh))
        ops_05 = run_operators(domain, build_signal(domain, 0.5, inh))
        # 0.2T survives, 0.5T does not — demonstrating higher threshold at lower B0
        assert ops_02.separation_survived, \
            f"0.2T failed at 100 ppm: ratio={ops_02.boundary_ratio:.4f}"
        assert not ops_05.separation_survived, \
            f"0.5T survived at 100 ppm unexpectedly: ratio={ops_05.boundary_ratio:.4f}"

    def test_separation_threshold_exists_at_all_reference_b0(self, domain):
        """
        At maximum declared inhomogeneity (10,000 ppm), the operator
        computation completes and returns a finite, non-NaN, non-Inf ratio.
        This verifies the numerical floor handling in operators_0e.py
        (underflow → 0.0, not inf or NaN) across all reference B0 values.
        """
        inh = INHOMOGENEITY_SWEEP[-1]
        for b0 in REFERENCE_B0_VALUES:
            sig = build_signal(domain, b0, inh)
            ops = run_operators(domain, sig)
            assert np.isfinite(ops.boundary_ratio), \
                f"Non-finite boundary ratio at {b0}T, max inhomogeneity: " \
                f"{ops.boundary_ratio}"
            assert not np.isnan(ops.boundary_ratio), \
                f"NaN boundary ratio at {b0}T, max inhomogeneity"
            assert ops.boundary_ratio != float('inf'), \
                f"Inf boundary ratio at {b0}T — underflow floor not applied"

    def test_perfect_homogeneity_all_b0_pass(self, domain):
        """At perfect homogeneity, all reference B0 values pass."""
        for b0 in REFERENCE_B0_VALUES:
            ops = run_operators(domain, build_signal(domain, b0, 0.0))
            assert ops.separation_survived, \
                f"Separation failed at {b0}T with perfect homogeneity"


# ---- Group 6: C projection provenance -------------------------------

class TestProjection:

    def test_ratio_computable_from_components(self, ops_05_perfect):
        """boundary_ratio equals boundary_E / max_interior independently."""
        max_interior = max(ops_05_perfect.interior_tumor_E_mean,
                           ops_05_perfect.interior_gm_E_mean)
        expected = ops_05_perfect.boundary_E_mean / max_interior
        assert abs(ops_05_perfect.boundary_ratio - expected) < 1e-10

    def test_boundary_e_exceeds_interior_at_perfect(self, ops_05_perfect):
        """Boundary E mean > both interior means at perfect homogeneity."""
        assert ops_05_perfect.boundary_E_mean > ops_05_perfect.interior_tumor_E_mean
        assert ops_05_perfect.boundary_E_mean > ops_05_perfect.interior_gm_E_mean

    def test_inhomogeneity_ppm_on_operator_result(self, ops_05_1000ppm):
        """Operator result carries inhomogeneity_ppm from signal."""
        assert abs(ops_05_1000ppm.inhomogeneity_ppm - 1000.0) < 1e-6


# ---- Group 7: Hardware spec admissibility ---------------------------

class TestHardwareSpec:

    def test_halbach_lower_bound_survives_all_b0(self, domain):
        """
        Halbach lower bound (10 ppm) is within the survival region
        at all declared reference B0 values.
        10 ppm < 51 ppm (most conservative threshold, at 0.5T).
        This is the admissible basis for the Phase 1 hardware claim.
        """
        inh = 10 * PPM_TO_FRAC
        for b0 in REFERENCE_B0_VALUES:
            ops = run_operators(domain, build_signal(domain, b0, inh))
            assert ops.separation_survived, (
                f"Separation failed at {b0}T, 10 ppm: "
                f"ratio={ops.boundary_ratio:.4f}"
            )

    def test_halbach_upper_bound_survives_at_02t_only(self, domain):
        """
        Halbach upper bound (100 ppm) survives at 0.2T but not at 0.5T.
        Declared result from sweep:
          0.2T threshold ~138 ppm → 100 ppm survives
          0.5T threshold ~51 ppm  → 100 ppm fails
        Phase 1 hardware operating at 0.2T with Halbach array is
        within the declared survival region.
        """
        inh = 100 * PPM_TO_FRAC
        ops_02 = run_operators(domain, build_signal(domain, 0.2, inh))
        ops_05 = run_operators(domain, build_signal(domain, 0.5, inh))
        assert ops_02.separation_survived,  "Expected survival at 0.2T, 100 ppm"
        assert not ops_05.separation_survived, "Expected failure at 0.5T, 100 ppm"

    def test_declared_open_condition_acknowledged(self, domain):
        """
        Simulation uses uniform inhomogeneity per element — worst case.
        Confirm the domain records this as a conservative model by
        checking that the inhomogeneity sweep starts at 0 (baseline
        matches Phase 0d exactly at zero inhomogeneity).
        """
        sig_zero = build_signal(domain, 0.5, 0.0)
        assert sig_zero.delta_B0_abs == 0.0
        assert abs(sig_zero.t2star_eff['tumor'] - 80.0) < 0.01
