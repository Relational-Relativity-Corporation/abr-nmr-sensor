# sim/signal_0d.py — abr-nmr-phase0d
# Metatron Dynamics, Inc.
#
# Per-element signal model for Phase 0d, parameterised by B0.
#
# Identical physics to signal_0c.py. B0 is now an explicit argument
# rather than a module-level constant. All B0-dependent quantities
# are recomputed for each sweep step.
#
# B0-dependent quantities:
#   omega_0    = GAMMA_RAD * B0
#   M_0        = (gamma² * hbar² * B0 * rho_proton) / (4 * kB * T)
#   V_signal   ∝ omega_0 * M_0  ∝ B0²
#   delta_B    = delta_chi * B0 * geometry_factor * vf * A * sin(phi)
#   R2star     = 1/T2*_baseline + gamma * |delta_B|
#
# B0-independent quantities:
#   T1, T2*_baseline, water_fraction   (held at 1.5T values — declared)
#   Coil geometry, resistance, noise
#   Tissue contrast (water content difference)
#
# Returns SignalResult0d — a lightweight struct holding only the
# quantities the operator and reporting layers need, without the
# full 1200-step arrays (which are not needed for the sweep summary).
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass

from sim.declaration_0d import (
    DeclaredDomain0d,
    GAMMA_RAD_PER_S_PER_T,
    HBAR,
    MU_0,
    K_BOLTZMANN,
    PROTON_DENSITY_WATER,
    TR_S,
    TE_S,
    FLIP_ANGLE_RAD,
    N_ELEMENTS,
    PHANTOM_RADIUS_MM,
    PHANTOM_LENGTH_MM,
    ELEMENT_ANGLE_DEG,
    ELEMENT_TURNS,
    WIRE_DIAMETER_MM,
    RHO_COPPER,
    ELEMENT_SENSITIVITY_FRACTION,
    RECEIVER_BW_HZ,
    PREAMP_NOISE_FACTOR,
    TEMPERATURE_K,
    TUMOR_SECTORS,
)

DELTA_CHI_DEOXY = 0.264e-6
GEOMETRY_FACTOR = 0.5
A_OXYGENATION   = 0.10


@dataclass
class SignalResult0d:
    """
    Per-element signal quantities for one B0 value.

    Attributes
    ----------
    b0 : float
        Field strength for this result [T].

    S_baseline : float64 [n_elements]
        Per-element baseline signal (zero susceptibility).

    S_field : float64 [n_elements, n_steps]
        Full per-element signal timeseries across cardiac steps.
        Used by operator layer for A-field computation.

    V_signal_peak : float
        Peak signal voltage at receiver input [V]. Scales as B0².

    noise_rms : float64 [n_elements]
        Per-element RMS thermal noise voltage [V]. B0-independent.

    SNR : float64 [n_elements]
        Per-element SNR = V_signal * S_baseline[i] / noise_rms[i].

    min_SNR : float
        Minimum SNR across all elements.

    all_detectable : bool
        True if all elements meet the declared SNR threshold.
    """
    b0:             float
    S_baseline:     np.ndarray   # [n_elements]
    S_field:        np.ndarray   # [n_elements, n_steps]
    V_signal_peak:  float
    noise_rms:      np.ndarray   # [n_elements]
    SNR:            np.ndarray   # [n_elements]
    min_SNR:        float
    all_detectable: bool


def build_signal(domain: DeclaredDomain0d, b0: float) -> SignalResult0d:
    """
    Compute per-element NMR signal at declared B0.

    Parameters
    ----------
    domain : DeclaredDomain0d
    b0 : float
        Field strength [T] for this sweep step.

    Returns
    -------
    SignalResult0d
    """
    n_steps = domain.n_cardiac_steps
    advance = float(domain.cardiac_phase_advance_rad)

    # ---- Cardiac phase sequence --------------------------------------
    t_idx         = np.arange(n_steps, dtype=np.float32)
    cardiac_phase = (t_idx * advance).astype(np.float64) % (2.0 * np.pi)

    # ---- Per-tissue signal [n_steps] ---------------------------------
    S_tissue          = {}
    S_baseline_tissue = {}

    for name, props in domain.tissue_types.items():
        PD       = float(props['water_fraction'])
        T1_s     = props['T1_ms'] * 1e-3
        T2star_s = props['T2star_ms'] * 1e-3
        sin_a    = float(np.sin(FLIP_ANGLE_RAD))

        T1_factor = 1.0 - np.exp(-TR_S / T1_s)
        S_base    = PD * sin_a * np.exp(-TE_S / T2star_s) * T1_factor
        S_baseline_tissue[name] = float(S_base)

        if props['vascular']:
            vf = props.get('vascular_fraction', 0.0)
            # delta_B scales with B0
            delta_B = (
                DELTA_CHI_DEOXY * b0 * GEOMETRY_FACTOR *
                vf * A_OXYGENATION * np.sin(cardiac_phase)
            )
            R2star   = 1.0 / T2star_s + GAMMA_RAD_PER_S_PER_T * np.abs(delta_B)
            T2star_t = 1.0 / R2star
            S_t      = PD * sin_a * np.exp(-TE_S / T2star_t) * T1_factor
        else:
            S_t = np.full(n_steps, S_base)

        S_tissue[name] = S_t

    # ---- Per-element signal [N_ELEMENTS, n_steps] --------------------
    S_field    = np.zeros((N_ELEMENTS, n_steps), dtype=np.float64)
    S_baseline = np.zeros(N_ELEMENTS, dtype=np.float64)

    for el in domain.elements:
        for tissue_name, fraction in el.tissue_mix.items():
            if fraction > 0:
                S_field[el.element_id, :] += fraction * S_tissue[tissue_name]
                S_baseline[el.element_id] += (
                    fraction * S_baseline_tissue[tissue_name]
                )

    # ---- Signal voltage per element [V] ------------------------------
    # V_signal ∝ omega_0 * M_0 ∝ B0²
    omega_0   = GAMMA_RAD_PER_S_PER_T * b0
    M_0_water = (
        GAMMA_RAD_PER_S_PER_T ** 2 * HBAR ** 2 * b0 *
        PROTON_DENSITY_WATER
    ) / (4.0 * K_BOLTZMANN * TEMPERATURE_K)

    arc_rad      = np.radians(ELEMENT_ANGLE_DEG)
    r_phantom_m  = PHANTOM_RADIUS_MM * 1e-3
    arc_length_m = arc_rad * (r_phantom_m + 0.005)
    length_m     = PHANTOM_LENGTH_MM * 1e-3
    n_turns      = float(ELEMENT_TURNS)
    A_element_m2 = arc_length_m * (length_m / n_turns)

    V_signal_peak = (
        omega_0 * MU_0 * n_turns * A_element_m2 *
        M_0_water * ELEMENT_SENSITIVITY_FRACTION
    )

    # ---- Noise per element [V] — B0-independent ----------------------
    noise_rms = _compute_element_noise(domain)

    # ---- SNR per element --------------------------------------------
    SNR = np.zeros(N_ELEMENTS)
    for i in range(N_ELEMENTS):
        V_sig_at_TE = V_signal_peak * float(S_baseline[i])
        SNR[i]      = V_sig_at_TE / noise_rms[i]

    min_SNR        = float(SNR.min())
    all_detectable = bool(min_SNR >= domain.snr_threshold)

    return SignalResult0d(
        b0=b0,
        S_baseline=S_baseline,
        S_field=S_field,
        V_signal_peak=float(V_signal_peak),
        noise_rms=noise_rms,
        SNR=SNR,
        min_SNR=min_SNR,
        all_detectable=all_detectable,
    )


def _compute_element_noise(domain: DeclaredDomain0d) -> np.ndarray:
    """Johnson-Nyquist thermal noise per element. B0-independent."""
    noise_rms    = np.zeros(N_ELEMENTS)
    arc_rad      = np.radians(ELEMENT_ANGLE_DEG)
    r_coil_m     = (PHANTOM_RADIUS_MM + 5.0) * 1e-3
    arc_len_m    = arc_rad * r_coil_m
    wire_diam_m  = WIRE_DIAMETER_MM * 1e-3
    wire_area_m2 = np.pi * (wire_diam_m / 2) ** 2

    for i in range(N_ELEMENTS):
        wire_len_m  = ELEMENT_TURNS * arc_len_m
        R_coil      = RHO_COPPER * wire_len_m / wire_area_m2
        V_noise_var = (
            4.0 * K_BOLTZMANN *
            domain.temperature_k *
            R_coil *
            domain.receiver_bw_hz *
            domain.preamp_noise_factor
        )
        noise_rms[i] = float(np.sqrt(V_noise_var))

    return noise_rms
