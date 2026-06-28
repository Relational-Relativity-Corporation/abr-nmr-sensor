# tests/test_phase0.py — abr-nmr-phase0
# Metatron Dynamics, Inc.
#
# Verifier tests for Phase 0: single-segment solenoid, pulsatile phantom.
#
# All assertions derive from declared formulas and constants.
# No statistical operations. No heuristics.
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
import pytest
from sim.declaration import (
    declare_domain,
    CARDIAC_PHASE_ADVANCE_RAD,
    TR_S,
    TE_S,
    T1_MS,
    T2STAR_BASELINE_MS,
    PROTON_DENSITY_REL,
    FLIP_ANGLE_RAD,
    B0_TESLA,
    GAMMA_RAD_PER_S_PER_T,
    DELTA_CHI_DEOXY,
    GEOMETRY_FACTOR,
    BLOOD_VOLUME_FRACTION,
    A_OXYGENATION,
    K_BOLTZMANN,
    TEMPERATURE_K,
    COIL_RESISTANCE_OHM,
    RECEIVER_BW_HZ,
    PREAMP_NOISE_FACTOR,
    N_CARDIAC_STEPS,
    FILLING_FACTOR,
)
from sim.signal import build_signal
from sim.noise  import build_noise_model


# ---- Fixtures --------------------------------------------------------

@pytest.fixture(scope="module")
def domain():
    return declare_domain()

@pytest.fixture(scope="module")
def signal(domain):
    return build_signal(domain)

@pytest.fixture(scope="module")
def noise(domain, signal):
    max_frac = float(np.abs(
        (np.real(signal.S) - signal.S_baseline) / signal.S_baseline
    ).max())
    return build_noise_model(domain, signal.S_baseline, max_frac)


# ---- Declaration tests -----------------------------------------------

class TestDeclaration:

    def test_domain_cardiac_steps(self, domain):
        assert domain.n_cardiac_steps == N_CARDIAC_STEPS

    def test_domain_phase_advance(self, domain):
        expected = float(CARDIAC_PHASE_ADVANCE_RAD)
        assert abs(domain.cardiac_phase_advance_rad - expected) < 1e-10

    def test_filling_factor_bounded(self, domain):
        """Filling factor must be in (0, 1]."""
        assert 0.0 < domain.filling_factor <= 1.0

    def test_filling_factor_declared_value(self, domain):
        assert abs(domain.filling_factor - FILLING_FACTOR) < 1e-6

    def test_coil_resistance_positive(self, domain):
        assert domain.coil_resistance_ohm > 0.0

    def test_preamp_noise_factor_geq_one(self, domain):
        """Noise factor >= 1 by definition (cannot reduce noise)."""
        assert domain.preamp_noise_factor >= 1.0

    def test_preamp_noise_factor_declared(self, domain):
        assert abs(domain.preamp_noise_factor - PREAMP_NOISE_FACTOR) < 1e-6


# ---- Signal tests ----------------------------------------------------

class TestSignal:

    def test_cardiac_phase_shape(self, signal):
        assert signal.cardiac_phase.shape == (N_CARDIAC_STEPS,)

    def test_cardiac_phase_starts_zero(self, signal):
        assert abs(float(signal.cardiac_phase[0])) < 1e-5

    def test_cardiac_phase_bounded(self, signal):
        assert float(signal.cardiac_phase.min()) >= 0.0
        assert float(signal.cardiac_phase.max()) < 2.0 * np.pi + 1e-5

    def test_cardiac_phase_formula(self, signal):
        """
        phi(t) = (t * CARDIAC_PHASE_ADVANCE_RAD) mod 2pi
        Verify via declared float32 path.
        """
        advance  = float(CARDIAC_PHASE_ADVANCE_RAD)
        t_idx    = np.arange(N_CARDIAC_STEPS, dtype=np.float32)
        expected = (
            (t_idx * advance).astype(np.float64) % (2.0 * np.pi)
        ).astype(np.float32).astype(np.float64)
        assert np.allclose(signal.cardiac_phase, expected, atol=1e-4)

    def test_cardiac_phase_wraps(self, signal):
        """Phase must not be monotonically increasing — it wraps."""
        diffs = np.diff(signal.cardiac_phase)
        assert not np.all(diffs > 0), "Phase never wraps — modulo not applied."

    def test_S_shape(self, signal):
        assert signal.S.shape == (N_CARDIAC_STEPS,)

    def test_S_baseline_positive(self, signal):
        assert signal.S_baseline > 0.0

    def test_S_baseline_formula(self, signal):
        """
        S_baseline = PD * sin(alpha) * exp(-TE/T2*_baseline) * (1-exp(-TR/T1))
        """
        PD       = float(PROTON_DENSITY_REL)
        sin_a    = float(np.sin(FLIP_ANGLE_RAD))
        T2s      = T2STAR_BASELINE_MS * 1e-3
        T1       = T1_MS * 1e-3
        expected = PD * sin_a * np.exp(-TE_S / T2s) * (1.0 - np.exp(-TR_S / T1))
        assert abs(signal.S_baseline - expected) < 1e-6

    def test_S_leq_baseline(self, signal):
        """
        Susceptibility only adds to R2*, shortening T2* and reducing S.
        S(t) <= S_baseline for all t.
        """
        S_real = np.real(signal.S)
        violations = int(np.sum(S_real > signal.S_baseline + 1e-8))
        assert violations == 0, (
            f"{violations} steps where S > S_baseline. "
            "Susceptibility should reduce signal."
        )

    def test_delta_B_non_negative(self, signal):
        """delta_B = delta_chi * B0 * geom_factor >= 0 for all t."""
        assert float(signal.delta_B.min()) >= 0.0

    def test_delta_B_formula_at_t0(self, signal):
        """
        At t=0: cardiac_phase=0, sin(0)=0, oxygenation=0,
        delta_chi = DELTA_CHI_DEOXY * 1 * BLOOD_VOLUME_FRACTION
        delta_B = delta_chi * B0 * GEOMETRY_FACTOR
        """
        expected_chi = DELTA_CHI_DEOXY * 1.0 * BLOOD_VOLUME_FRACTION
        expected_dB  = expected_chi * B0_TESLA * GEOMETRY_FACTOR
        actual_dB    = float(signal.delta_B[0])
        assert abs(actual_dB - expected_dB) < 1e-15

    def test_T2star_shorter_than_baseline(self, signal):
        """
        T2*(t) <= T2*_baseline everywhere.
        Susceptibility adds to R2*, shortening T2*.
        """
        assert float(signal.T2star.max()) <= T2STAR_BASELINE_MS + 1e-6

    def test_T2star_formula_spot_check(self, signal):
        """
        1/T2*(t) = 1/T2*_baseline + gamma * |delta_B(t)|
        Verify at t=0.
        """
        T2s_base = T2STAR_BASELINE_MS * 1e-3
        dB_t0    = float(signal.delta_B[0])
        R2star   = 1.0/T2s_base + GAMMA_RAD_PER_S_PER_T * abs(dB_t0)
        expected = (1.0 / R2star) * 1e3
        actual   = float(signal.T2star[0])
        assert abs(actual - expected) < 1e-6

    def test_S_varies_over_steps(self, signal):
        """
        S must vary — cardiac modulation is active.
        Direct check: S at max(delta_B) < S at min(delta_B).
        """
        t_max = int(np.argmax(signal.delta_B))
        t_min = int(np.argmin(signal.delta_B))
        assert float(np.real(signal.S[t_max])) < float(np.real(signal.S[t_min]))


# ---- Noise tests -----------------------------------------------------

class TestNoise:

    def test_noise_coil_formula(self, noise):
        """
        V_noise_coil = sqrt(4 * k_B * T * R * BW)
        """
        expected = float(np.sqrt(
            4.0 * K_BOLTZMANN * TEMPERATURE_K *
            COIL_RESISTANCE_OHM * RECEIVER_BW_HZ
        ))
        assert abs(noise.V_noise_coil_rms - expected) < 1e-15

    def test_noise_total_geq_coil(self, noise):
        """Total noise >= coil noise (preamp adds noise)."""
        assert noise.V_noise_total_rms >= noise.V_noise_coil_rms

    def test_noise_total_formula(self, noise):
        """V_noise_total = V_noise_coil * sqrt(F_preamp)"""
        expected = noise.V_noise_coil_rms * float(np.sqrt(PREAMP_NOISE_FACTOR))
        assert abs(noise.V_noise_total_rms - expected) < 1e-20

    def test_signal_peak_positive(self, noise):
        assert noise.V_signal_peak > 0.0

    def test_SNR_positive(self, noise):
        assert noise.SNR_per_step > 0.0

    def test_SNR_detectable(self, noise):
        """
        Declared Phase 0 result: SNR per step >> 5.
        Single step is sufficient for detection.
        """
        assert noise.SNR_per_step >= 5.0, (
            f"SNR per step {noise.SNR_per_step:.2f} < 5. "
            "Not detectable in single step at declared parameters."
        )

    def test_modulation_detectable(self, noise):
        """
        Cardiac modulation SNR per step >= 5.
        Operators act on unaveraged signal — admissible only if true.
        """
        assert noise.modulation_snr_per_step >= 5.0, (
            f"Modulation SNR {noise.modulation_snr_per_step:.4f} < 5. "
            "Modulation not detectable in single step."
        )

    def test_detectable_within_window(self, noise):
        """
        If modulation SNR < 5 per step, must be detectable within
        N_CARDIAC_STEPS by coherent averaging.
        """
        assert noise.detectable_in_N_steps <= N_CARDIAC_STEPS, (
            f"Requires {noise.detectable_in_N_steps} steps but only "
            f"{N_CARDIAC_STEPS} declared. HARD STOP."
        )
