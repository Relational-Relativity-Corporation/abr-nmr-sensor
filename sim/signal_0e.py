# sim/signal_0e.py — abr-nmr-phase0e
# Metatron Dynamics, Inc.
#
# Per-element signal model for Phase 0e, parameterised by B0
# and fractional B0 inhomogeneity.
#
# The inhomogeneity contribution enters as an additional R2* term:
#
#   1/T2*_eff(tissue) = 1/T2*_tissue + gamma * delta_B0_abs
#
# where:
#   delta_B0_abs = inhomogeneity_frac * B0   [T]
#
# This is physically equivalent to the susceptibility-driven R2*
# shortening already in signal_0d.py — the same formula applied
# to a different source of field variation. Both sources add to R2*
# because both cause intravoxel dephasing within the element's
# sensitivity volume.
#
# The inhomogeneity term is time-invariant within a sweep step —
# it represents static field imperfection, not cardiac modulation.
# Cardiac modulation (the dynamic delta_B from susceptibility) is
# still present and still computed as in signal_0d.py.
#
# The two R2* contributions are additive:
#   R2*_total = R2*_tissue_baseline
#             + gamma * |delta_B_cardiac(t)|    (dynamic, per step)
#             + gamma * delta_B0_inhomogeneity  (static, constant)
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass

from sim.declaration_0e import (
    DeclaredDomain0e,
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
)

DELTA_CHI_DEOXY = 0.264e-6
GEOMETRY_FACTOR = 0.5
A_OXYGENATION   = 0.10


@dataclass
class SignalResult0e:
    """
    Per-element signal quantities for one (B0, inhomogeneity) pair.

    Attributes
    ----------
    b0 : float
    inhomogeneity_frac : float
        Fractional B0 inhomogeneity for this result.
    inhomogeneity_ppm : float
        Same value in ppm (= inhomogeneity_frac * 1e6).
    delta_B0_abs : float
        Absolute field variation [T] = inhomogeneity_frac * B0.
    S_baseline : float64 [n_elements]
    S_field : float64 [n_elements, n_steps]
    V_signal_peak : float
    noise_rms : float64 [n_elements]
    SNR : float64 [n_elements]
    min_SNR : float
    all_detectable : bool
    t2star_eff : dict
        Effective T2* per tissue [ms] at this inhomogeneity level.
        Keys: tissue names. Values: scalar (static contribution only).
    """
    b0:                 float
    inhomogeneity_frac: float
    inhomogeneity_ppm:  float
    delta_B0_abs:       float
    S_baseline:         np.ndarray
    S_field:            np.ndarray
    V_signal_peak:      float
    noise_rms:          np.ndarray
    SNR:                np.ndarray
    min_SNR:            float
    all_detectable:     bool
    t2star_eff:         dict


def build_signal(
    domain: DeclaredDomain0e,
    b0: float,
    inhomogeneity_frac: float,
) -> SignalResult0e:
    """
    Compute per-element NMR signal at declared B0 and inhomogeneity.

    Parameters
    ----------
    domain : DeclaredDomain0e
    b0 : float  [T]
    inhomogeneity_frac : float
        Fractional B0 inhomogeneity (0 = perfect, 0.001 = 1000 ppm).

    Returns
    -------
    SignalResult0e
    """
    n_steps          = domain.n_cardiac_steps
    advance          = float(domain.cardiac_phase_advance_rad)
    delta_B0_abs     = inhomogeneity_frac * b0
    inhomogeneity_ppm = inhomogeneity_frac * 1e6

    # ---- Cardiac phase sequence --------------------------------------
    t_idx         = np.arange(n_steps, dtype=np.float32)
    cardiac_phase = (t_idx * advance).astype(np.float64) % (2.0 * np.pi)

    # ---- Per-tissue signal [n_steps] ---------------------------------
    S_tissue          = {}
    S_baseline_tissue = {}
    t2star_eff        = {}

    for name, props in domain.tissue_types.items():
        PD       = float(props['water_fraction'])
        T1_s     = props['T1_ms'] * 1e-3
        T2star_s = props['T2star_ms'] * 1e-3
        sin_a    = float(np.sin(FLIP_ANGLE_RAD))
        T1_factor = 1.0 - np.exp(-TR_S / T1_s)

        # Static R2* from inhomogeneity — time-invariant
        R2star_inhomogeneity = GAMMA_RAD_PER_S_PER_T * delta_B0_abs

        if props['vascular']:
            vf = props.get('vascular_fraction', 0.0)
            # Dynamic R2* from cardiac susceptibility modulation
            delta_B_cardiac = (
                DELTA_CHI_DEOXY * b0 * GEOMETRY_FACTOR *
                vf * A_OXYGENATION * np.sin(cardiac_phase)
            )
            R2star_cardiac = GAMMA_RAD_PER_S_PER_T * np.abs(delta_B_cardiac)

            # Total R2* = baseline + inhomogeneity (static) + cardiac (dynamic)
            R2star_total = (1.0 / T2star_s) + R2star_inhomogeneity + R2star_cardiac
            T2star_eff_t = 1.0 / R2star_total
            S_t          = PD * sin_a * np.exp(-TE_S / T2star_eff_t) * T1_factor

            # Effective T2* at zero cardiac phase (static contribution only)
            T2star_eff_static = 1.0 / (1.0/T2star_s + R2star_inhomogeneity)
        else:
            # No cardiac modulation — inhomogeneity only
            R2star_total      = (1.0 / T2star_s) + R2star_inhomogeneity
            T2star_eff_static = 1.0 / R2star_total
            S_t               = np.full(
                n_steps,
                PD * sin_a * np.exp(-TE_S * R2star_total) * T1_factor
            )

        S_baseline_tissue[name] = float(
            PD * sin_a * np.exp(-TE_S / T2star_eff_static) * T1_factor
        )
        S_tissue[name]  = S_t
        t2star_eff[name] = float(T2star_eff_static * 1e3)   # [ms]

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

    # ---- Signal voltage [V] — scales as B0² -------------------------
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

    # ---- Noise [V] — B0-independent ----------------------------------
    noise_rms = _compute_element_noise(domain)

    # ---- SNR per element --------------------------------------------
    SNR = np.zeros(N_ELEMENTS)
    for i in range(N_ELEMENTS):
        V_sig_at_TE = V_signal_peak * float(S_baseline[i])
        SNR[i]      = V_sig_at_TE / noise_rms[i]

    min_SNR        = float(SNR.min())
    all_detectable = bool(min_SNR >= domain.snr_threshold)

    return SignalResult0e(
        b0=b0,
        inhomogeneity_frac=inhomogeneity_frac,
        inhomogeneity_ppm=inhomogeneity_ppm,
        delta_B0_abs=float(delta_B0_abs),
        S_baseline=S_baseline,
        S_field=S_field,
        V_signal_peak=float(V_signal_peak),
        noise_rms=noise_rms,
        SNR=SNR,
        min_SNR=min_SNR,
        all_detectable=all_detectable,
        t2star_eff=t2star_eff,
    )


def _compute_element_noise(domain: DeclaredDomain0e) -> np.ndarray:
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
