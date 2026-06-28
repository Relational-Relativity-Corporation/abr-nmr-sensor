# sim/signal_0b.py — abr-nmr-phase0b
# Metatron Dynamics, Inc.
#
# Multi-tissue signal model for the Phase 0b array simulation.
#
# For each coil element, the received signal is the volume-weighted
# sum of contributions from all tissue types in its sensitivity volume.
#
# S_element(t) = Σ_tissue [fraction_tissue × S_tissue(t)]
#
# where S_tissue(t) is the spin echo signal from that tissue type
# at cardiac step t, exactly as declared in signal.py but now
# per tissue type with its own PD, T1, T2*, and cardiac modulation.
#
# Cardiac modulation applies only to vascular tissues:
#   delta_B_tissue(t) = delta_chi × B0 × geom × vascular_fraction
#                       × A_oxygenation × sin(phi(t))
#   T2*_tissue(t) = 1 / (1/T2*_baseline + gamma × |delta_B_tissue(t)|)
#
# The output is S[n_elements, n_cardiac_steps] — the declared
# signal field that the ABRCE operators act on.
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass
from sim.declaration_0b import (
    DeclaredDomain0b,
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
)

# Susceptibility constants
DELTA_CHI_DEOXY    = 0.264e-6    # [dimensionless]
GEOMETRY_FACTOR    = 0.5
A_OXYGENATION      = 0.10        # Fractional oxygenation modulation


@dataclass
class SignalField0b:
    """
    Declared signal outputs for the Phase 0b array simulation.

    Attributes
    ----------
    cardiac_phase : float64 [n_steps]
        Declared cardiac phase at each evolution step.

    S : float64 [n_elements, n_steps]
        Per-element spin echo signal amplitude at each cardiac step.
        Volume-weighted sum over tissue types in each element.

    S_baseline : float64 [n_elements]
        Per-element baseline signal at zero susceptibility.

    S_tissue : dict of float64 [n_steps]
        Per-tissue-type signal at each cardiac step.
        For verification and reporting.

    noise_rms : float64 [n_elements]
        Per-element RMS thermal noise voltage.
        Derived from declared element geometry and temperature.

    SNR : float64 [n_elements]
        Per-element SNR at baseline signal level.

    modulation_depth : float64 [n_elements]
        Per-element fractional cardiac modulation |delta_S/S_baseline|.
    """
    cardiac_phase:    np.ndarray   # [n_steps]
    S:                np.ndarray   # [n_elements, n_steps]
    S_baseline:       np.ndarray   # [n_elements]
    S_tissue:         dict         # tissue_name → [n_steps]
    noise_rms:        np.ndarray   # [n_elements]
    SNR:              np.ndarray   # [n_elements]
    modulation_depth: np.ndarray   # [n_elements]


def build_signal(domain: DeclaredDomain0b) -> SignalField0b:
    """
    Compute per-element NMR signal across all cardiac steps.

    For each tissue type: compute the spin echo signal at each step.
    For each element: compute the volume-weighted sum.
    For each element: compute noise and SNR.

    All operations are exact formula evaluations.
    No statistical operations. No approximations.
    """
    n_steps = domain.n_cardiac_steps
    advance = float(domain.cardiac_phase_advance_rad)

    # ---- Cardiac phase sequence --------------------------------------
    t_idx         = np.arange(n_steps, dtype=np.float32)
    cardiac_phase = (
        (t_idx * advance).astype(np.float64) % (2.0 * np.pi)
    )

    # ---- Per-tissue signal [n_steps] ---------------------------------
    S_tissue = {}
    S_baseline_tissue = {}

    T1_factor_tissue  = {}
    for name, props in TISSUE_TYPES.items():
        PD       = float(props['water_fraction'])
        T1_s     = props['T1_ms'] * 1e-3
        T2star_s = props['T2star_ms'] * 1e-3
        sin_a    = float(np.sin(FLIP_ANGLE_RAD))

        T1_factor = 1.0 - np.exp(-TR_S / T1_s)
        T1_factor_tissue[name] = T1_factor

        # Baseline signal (no susceptibility modulation)
        S_base = PD * sin_a * np.exp(-TE_S / T2star_s) * T1_factor
        S_baseline_tissue[name] = float(S_base)

        if props['vascular']:
            # Cardiac modulation through vascular susceptibility
            vf = props.get('vascular_fraction', 0.0)
            delta_B = (
                DELTA_CHI_DEOXY * B0_TESLA * GEOMETRY_FACTOR *
                vf * A_OXYGENATION * np.sin(cardiac_phase)
            )
            R2star = 1.0/T2star_s + GAMMA_RAD_PER_S_PER_T * np.abs(delta_B)
            T2star_t = 1.0 / R2star
            S_t = PD * sin_a * np.exp(-TE_S / T2star_t) * T1_factor
        else:
            # No cardiac modulation — static signal
            S_t = np.full(n_steps, S_base)

        S_tissue[name] = S_t

    # ---- Per-element signal [n_elements, n_steps] --------------------
    S = np.zeros((N_ELEMENTS, n_steps), dtype=np.float64)
    S_baseline = np.zeros(N_ELEMENTS, dtype=np.float64)

    for el in domain.elements:
        for tissue_name, fraction in el.tissue_mix.items():
            if fraction > 0:
                S[el.element_id, :] += fraction * S_tissue[tissue_name]
                S_baseline[el.element_id] += (
                    fraction * S_baseline_tissue[tissue_name]
                )

    # ---- Per-element noise and SNR -----------------------------------
    noise_rms = _compute_element_noise(domain)

    # Signal voltage per element
    # Use the same NMR signal voltage formula as Phase 0
    omega_0 = GAMMA_RAD_PER_S_PER_T * B0_TESLA
    M_0_water = (
        GAMMA_RAD_PER_S_PER_T**2 * HBAR**2 * B0_TESLA *
        PROTON_DENSITY_WATER
    ) / (4.0 * K_BOLTZMANN * TEMPERATURE_K)

    # Element sensitivity volume [m^3]
    arc_rad     = np.radians(ELEMENT_ANGLE_DEG)
    r_phantom_m = PHANTOM_RADIUS_MM * 1e-3
    depth_m     = ELEMENT_SENSITIVITY_DEPTH_MM * 1e-3
    length_m    = PHANTOM_LENGTH_MM * 1e-3
    V_sens_m3   = arc_rad * r_phantom_m * depth_m * length_m

    # Coil geometry per element
    arc_length_m = arc_rad * (r_phantom_m + 0.005)  # +5mm clearance
    n_turns      = float(ELEMENT_TURNS)
    wire_diam_m  = WIRE_DIAMETER_MM * 1e-3
    A_element_m2 = arc_length_m * (length_m / n_turns)  # approximate loop area

    V_signal_peak = (
        omega_0 * MU_0 * n_turns * A_element_m2 *
        M_0_water * ELEMENT_SENSITIVITY_FRACTION
    )

    # Per-element SNR
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

    signal = SignalField0b(
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


def _compute_element_noise(domain: DeclaredDomain0b) -> np.ndarray:
    """
    Johnson-Nyquist thermal noise per element.
    Each element has its own wire length and resistance.
    """
    noise_rms = np.zeros(N_ELEMENTS)
    arc_rad     = np.radians(ELEMENT_ANGLE_DEG)
    r_coil_m    = (PHANTOM_RADIUS_MM + 5.0) * 1e-3  # +5mm clearance
    arc_len_m   = arc_rad * r_coil_m
    wire_diam_m = WIRE_DIAMETER_MM * 1e-3
    wire_area_m2 = np.pi * (wire_diam_m / 2)**2

    for i in range(N_ELEMENTS):
        wire_len_m = ELEMENT_TURNS * arc_len_m
        R_coil     = RHO_COPPER * wire_len_m / wire_area_m2
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
    sig: SignalField0b,
    domain: DeclaredDomain0b,
) -> None:
    print("\n--- Signal Field Summary (Phase 0b) ---")

    print(f"\n  Per-tissue baseline signal:")
    print(f"  {'Tissue':>16}  {'S_baseline':>12}  {'T2*_mod%':>10}")
    for name, props in TISSUE_TYPES.items():
        s_vals = sig.S_tissue[name]
        if props['vascular']:
            mod = float(np.abs(s_vals - s_vals.mean()).max() / s_vals.mean() * 100)
        else:
            mod = 0.0
        base = float(props['water_fraction'] * np.sin(FLIP_ANGLE_RAD) *
                     np.exp(-TE_S / (props['T2star_ms']*1e-3)) *
                     (1 - np.exp(-TR_S / (props['T1_ms']*1e-3))))
        print(f"  {name:>16}  {base:>12.6f}  {mod:>9.4f}%")

    print(f"\n  Per-element signal and SNR:")
    print(f"  {'Elem':>4}  {'Hemisphere':>14}  "
          f"{'S_base':>8}  {'SNR':>10}  {'Mod%':>8}")
    for el in domain.elements:
        i    = el.element_id
        hemi = 'Right(tumor)' if i in {0,1,2,3} else 'Left(GM)'
        print(f"  {i:>4}  {hemi:>14}  "
              f"{sig.S_baseline[i]:>8.6f}  "
              f"{sig.SNR[i]:>10.1f}  "
              f"{sig.modulation_depth[i]*100:>7.4f}%")

    print(f"\n  S range (all elements, all steps): "
          f"{sig.S.min():.6f} – {sig.S.max():.6f}")
    print(f"  Min SNR: {sig.SNR.min():.1f}  "
          f"Max SNR: {sig.SNR.max():.1f}")
    print(f"  All elements detectable in single step: "
          f"{'YES' if sig.SNR.min() >= 5.0 else 'NO'}")
