# sim/signal_0c.py — abr-nmr-phase0c
# Metatron Dynamics, Inc.
#
# Multi-tissue signal model for the Phase 0c 32-element array.
#
# Same physical model as Phase 0b (signal_0b.py), scaled to 32 elements.
# Per-element sensitivity volume scales with arc length:
#   arc_length_0c = arc_length_0b * (8 / 32) = arc_length_0b / 4
#
# This reduces signal voltage per element relative to Phase 0b.
# The noise also decreases (shorter wire → lower resistance),
# but not proportionally — SNR per element is lower at 32 than at 8.
# The declared boundary detection ratio is set by tissue contrast,
# not element count, so the boundary finding is preserved.
#
# S[i, t] = Σ_tissue fraction_tissue[i] × S_tissue(t)
# SNR[i]  = V_signal[i] / V_noise[i]
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass
from sim.declaration_0c import (
    DeclaredDomain0c,
    TISSUE_TYPES,
    B0_TESLA,
    GAMMA_RAD_PER_S_PER_T,
    HBAR,
    MU_0,
    K_BOLTZMANN,
    TEMPERATURE_K,
    PROTON_DENSITY_WATER,
    TR_S,
    TE_S,
    FLIP_ANGLE_RAD,
    N_ELEMENTS,
    PHANTOM_RADIUS_MM,
    PHANTOM_LENGTH_MM,
    ELEMENT_SENSITIVITY_DEPTH_MM,
    ELEMENT_SENSITIVITY_FRACTION,
    ELEMENT_TURNS,
    WIRE_DIAMETER_MM,
    RHO_COPPER,
    ELEMENT_ANGLE_DEG,
    RECEIVER_BW_HZ,
    PREAMP_NOISE_FACTOR,
    TUMOR_SECTORS,
)

DELTA_CHI_DEOXY = 0.264e-6
GEOMETRY_FACTOR = 0.5
A_OXYGENATION   = 0.10


@dataclass
class SignalField0c:
    """
    Declared signal outputs for the Phase 0c 32-element array.

    Attributes
    ----------
    cardiac_phase : float64 [n_steps]
    S             : float64 [n_elements, n_steps]
    S_baseline    : float64 [n_elements]
    S_tissue      : dict tissue_name → float64 [n_steps]
    noise_rms     : float64 [n_elements]
    SNR           : float64 [n_elements]
    modulation_depth : float64 [n_elements]
    """
    cardiac_phase:    np.ndarray
    S:                np.ndarray
    S_baseline:       np.ndarray
    S_tissue:         dict
    noise_rms:        np.ndarray
    SNR:              np.ndarray
    modulation_depth: np.ndarray


def build_signal(domain: DeclaredDomain0c) -> SignalField0c:
    """
    Compute per-element NMR signal across all cardiac steps.
    Exact formula evaluations. No statistical operations.
    """
    n_steps = domain.n_cardiac_steps
    advance = float(domain.cardiac_phase_advance_rad)

    # ---- Cardiac phase -----------------------------------------------
    t_idx         = np.arange(n_steps, dtype=np.float32)
    cardiac_phase = (
        (t_idx * advance).astype(np.float64) % (2.0 * np.pi)
    )

    # ---- Per-tissue signal [n_steps] ---------------------------------
    S_tissue          = {}
    S_baseline_tissue = {}

    for name, props in TISSUE_TYPES.items():
        PD       = float(props['water_fraction'])
        T1_s     = props['T1_ms'] * 1e-3
        T2star_s = props['T2star_ms'] * 1e-3
        sin_a    = float(np.sin(FLIP_ANGLE_RAD))

        T1_factor = 1.0 - np.exp(-TR_S / T1_s)
        S_base    = PD * sin_a * np.exp(-TE_S / T2star_s) * T1_factor
        S_baseline_tissue[name] = float(S_base)

        if props['vascular']:
            vf      = props.get('vascular_fraction', 0.0)
            delta_B = (
                DELTA_CHI_DEOXY * B0_TESLA * GEOMETRY_FACTOR *
                vf * A_OXYGENATION * np.sin(cardiac_phase)
            )
            R2star  = 1.0 / T2star_s + GAMMA_RAD_PER_S_PER_T * np.abs(delta_B)
            T2star_t = 1.0 / R2star
            S_t     = PD * sin_a * np.exp(-TE_S / T2star_t) * T1_factor
        else:
            S_t = np.full(n_steps, S_base)

        S_tissue[name] = S_t

    # ---- Per-element signal [N_ELEMENTS, n_steps] --------------------
    S         = np.zeros((N_ELEMENTS, n_steps), dtype=np.float64)
    S_baseline = np.zeros(N_ELEMENTS, dtype=np.float64)

    for el in domain.elements:
        for tissue_name, fraction in el.tissue_mix.items():
            if fraction > 0:
                S[el.element_id, :] += fraction * S_tissue[tissue_name]
                S_baseline[el.element_id] += (
                    fraction * S_baseline_tissue[tissue_name]
                )

    # ---- Per-element noise -------------------------------------------
    noise_rms = _compute_element_noise(domain)

    # ---- Signal voltage and SNR --------------------------------------
    omega_0     = GAMMA_RAD_PER_S_PER_T * B0_TESLA
    M_0_water   = (
        GAMMA_RAD_PER_S_PER_T ** 2 * HBAR ** 2 * B0_TESLA *
        PROTON_DENSITY_WATER
    ) / (4.0 * K_BOLTZMANN * TEMPERATURE_K)

    arc_rad      = np.radians(ELEMENT_ANGLE_DEG)
    r_phantom_m  = PHANTOM_RADIUS_MM * 1e-3
    depth_m      = ELEMENT_SENSITIVITY_DEPTH_MM * 1e-3
    length_m     = PHANTOM_LENGTH_MM * 1e-3
    arc_length_m = arc_rad * (r_phantom_m + 0.005)
    n_turns      = float(ELEMENT_TURNS)
    A_element_m2 = arc_length_m * (length_m / n_turns)

    V_signal_peak = (
        omega_0 * MU_0 * n_turns * A_element_m2 *
        M_0_water * ELEMENT_SENSITIVITY_FRACTION
    )

    SNR = np.zeros(N_ELEMENTS)
    for i in range(N_ELEMENTS):
        V_sig_at_TE = V_signal_peak * float(S_baseline[i])
        SNR[i] = V_sig_at_TE / noise_rms[i]

    # ---- Modulation depth per element --------------------------------
    modulation_depth = np.zeros(N_ELEMENTS)
    for i in range(N_ELEMENTS):
        if S_baseline[i] > 0:
            frac = np.abs(S[i, :] - S_baseline[i]) / S_baseline[i]
            modulation_depth[i] = float(frac.max())

    signal = SignalField0c(
        cardiac_phase=cardiac_phase,
        S=S,
        S_baseline=S_baseline,
        S_tissue=S_tissue,
        noise_rms=noise_rms,
        SNR=SNR,
        modulation_depth=modulation_depth,
    )
    _print_signal_report(signal, domain)
    return signal


def _compute_element_noise(domain: DeclaredDomain0c) -> np.ndarray:
    """Johnson-Nyquist thermal noise per element at 32-element spacing."""
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


def _print_signal_report(
    sig: SignalField0c,
    domain: DeclaredDomain0c,
) -> None:
    print("\n--- Signal Field Summary (Phase 0c, 32-element) ---")

    print(f"\n  Per-tissue baseline signal:")
    print(f"  {'Tissue':>16}  {'S_baseline':>12}  {'T2*_mod%':>10}")
    for name, props in TISSUE_TYPES.items():
        s_vals = sig.S_tissue[name]
        mod = (
            float(np.abs(s_vals - s_vals.mean()).max() / s_vals.mean() * 100)
            if props['vascular'] else 0.0
        )
        base = float(
            props['water_fraction'] * np.sin(FLIP_ANGLE_RAD) *
            np.exp(-TE_S / (props['T2star_ms'] * 1e-3)) *
            (1 - np.exp(-TR_S / (props['T1_ms'] * 1e-3)))
        )
        print(f"  {name:>16}  {base:>12.6f}  {mod:>9.4f}%")

    print(f"\n  Per-element signal and SNR (boundary region, elements 13–18):")
    print(f"  {'Elem':>4}  {'Angle':>8}  {'Hemisphere':>16}  "
          f"{'S_base':>8}  {'SNR':>10}  {'Mod%':>8}")
    for el in domain.elements[13:19]:
        i    = el.element_id
        hemi = 'Right(tumor)' if i in TUMOR_SECTORS else 'Left(GM)'
        print(f"  {i:>4}  {el.angle_center_deg:>6.2f}°  "
              f"{hemi:>16}  "
              f"{sig.S_baseline[i]:>8.6f}  "
              f"{sig.SNR[i]:>10.1f}  "
              f"{sig.modulation_depth[i]*100:>7.4f}%")

    print(f"\n  S range (all elements, all steps): "
          f"{sig.S.min():.6f} – {sig.S.max():.6f}")
    print(f"  Min SNR: {sig.SNR.min():.1f}  Max SNR: {sig.SNR.max():.1f}")
    snr_ok = sig.SNR.min() >= 5.0
    print(f"  All elements detectable in single step: "
          f"{'YES' if snr_ok else 'NO'}")
