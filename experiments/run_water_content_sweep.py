# experiments/run_water_content_sweep.py — abr-nmr-phase0c
# Metatron Dynamics, Inc.
#
# Water content contrast sweep.
#
# Declares a sweep over the water content difference between the two
# declared tissue hemispheres. All other parameters held at Phase 0c
# declared values (B0=1.5T, TE=30ms, TR=800ms, 32-element array).
#
# At each sweep point:
#   - Right hemisphere water content: fixed at 0.92 (tumor reference)
#   - Left hemisphere water content: varied from 0.92 down to 0.77
#   - Water content difference: 0% to 15%
#   - T1 and T2* held at tumor declared values for both hemispheres
#     (isolates the water content effect on signal contrast)
#   - Operators run at rho_base=0.3 (declared) and rho_base=0.0
#     (confirming contrast origin is in A, not R, at each point)
#
# Output:
#   Boundary ratio vs water content difference.
#   Detection floor: ratio = 1 (boundary indistinguishable from interior).
#   Reported: minimum detectable water content difference at ratio > 1.
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.declaration_0c import (
    declare_domain,
    N_ELEMENTS,
    TUMOR_SECTORS,
    GRAY_MAT_SECTORS,
    TISSUE_TYPES,
    B0_TESLA,
    GAMMA_RAD_PER_S_PER_T,
    TR_S,
    TE_S,
    FLIP_ANGLE_RAD,
    HBAR,
    MU_0,
    K_BOLTZMANN,
    TEMPERATURE_K,
    PROTON_DENSITY_WATER,
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
from sim.signal_0c   import DELTA_CHI_DEOXY, GEOMETRY_FACTOR, A_OXYGENATION
from sim.operators_0c import run_operators, EField0c
from sim.signal_0c   import SignalField0c

OUTPUT_DIR = Path(__file__).parent / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Sweep parameters [declared] -------------------------------------

# Reference tissue: tumor (right hemisphere, fixed)
WATER_REF        = 0.92       # Right hemisphere water content (fixed)
T1_REF_MS        = 1400.0     # T1 used for both hemispheres in sweep
T2STAR_REF_MS    = 80.0       # T2* used for both hemispheres in sweep
VASCULAR_FRAC    = 0.08       # Vascular fraction (same both hemispheres)

# Sweep: left hemisphere water content varies
# Difference = WATER_REF - WATER_LEFT
DELTA_WATER_STEPS = 200       # Resolution of sweep
DELTA_WATER_MAX   = 0.15      # 15% maximum difference
DELTA_WATER_MIN   = 0.0       # 0% — identical tissues (ratio must → 1)


def _build_sweep_signal(
    domain,
    water_left: float,
    n_cardiac_steps: int,
    cardiac_phase_advance_rad: float,
) -> SignalField0c:
    """
    Build a signal field for one sweep point.
    Right hemisphere: WATER_REF fixed.
    Left hemisphere: water_left declared.
    T1, T2*, vascular fraction identical for both — isolates water content effect.
    """
    n_steps = n_cardiac_steps
    advance = float(cardiac_phase_advance_rad)

    t_idx         = np.arange(n_steps, dtype=np.float32)
    cardiac_phase = (t_idx * advance).astype(np.float64) % (2.0 * np.pi)

    T1_s     = T1_REF_MS * 1e-3
    T2star_s = T2STAR_REF_MS * 1e-3
    sin_a    = float(np.sin(FLIP_ANGLE_RAD))
    T1_fac   = 1.0 - np.exp(-TR_S / T1_s)

    # Cardiac modulation (same for both hemispheres — isolates water contrast)
    delta_B = (
        DELTA_CHI_DEOXY * B0_TESLA * GEOMETRY_FACTOR *
        VASCULAR_FRAC * A_OXYGENATION * np.sin(cardiac_phase)
    )
    R2star   = 1.0 / T2star_s + GAMMA_RAD_PER_S_PER_T * np.abs(delta_B)
    T2star_t = 1.0 / R2star

    # Per-hemisphere signal at each step
    S_right = WATER_REF   * sin_a * np.exp(-TE_S / T2star_t) * T1_fac
    S_left  = water_left  * sin_a * np.exp(-TE_S / T2star_t) * T1_fac

    # Per-element signal [N_ELEMENTS, n_steps]
    S = np.zeros((N_ELEMENTS, n_steps), dtype=np.float64)
    for i in range(N_ELEMENTS):
        if i in TUMOR_SECTORS:
            S[i, :] = S_right
        else:
            S[i, :] = S_left

    S_baseline = np.zeros(N_ELEMENTS, dtype=np.float64)
    S_base_right = float(WATER_REF  * sin_a * np.exp(-TE_S / T2star_s) * T1_fac)
    S_base_left  = float(water_left * sin_a * np.exp(-TE_S / T2star_s) * T1_fac)
    for i in range(N_ELEMENTS):
        S_baseline[i] = S_base_right if i in TUMOR_SECTORS else S_base_left

    # Noise and SNR (same geometry at every sweep point)
    noise_rms = _element_noise()
    omega_0   = GAMMA_RAD_PER_S_PER_T * B0_TESLA
    M_0_water = (
        GAMMA_RAD_PER_S_PER_T**2 * HBAR**2 * B0_TESLA * PROTON_DENSITY_WATER
    ) / (4.0 * K_BOLTZMANN * TEMPERATURE_K)
    arc_rad      = np.radians(ELEMENT_ANGLE_DEG)
    r_phantom_m  = PHANTOM_RADIUS_MM * 1e-3
    arc_length_m = arc_rad * (r_phantom_m + 0.005)
    length_m     = PHANTOM_LENGTH_MM * 1e-3
    n_turns      = float(ELEMENT_TURNS)
    A_elem_m2    = arc_length_m * (length_m / n_turns)
    V_signal_peak = (
        omega_0 * MU_0 * n_turns * A_elem_m2 *
        M_0_water * ELEMENT_SENSITIVITY_FRACTION
    )
    SNR = np.array([
        V_signal_peak * float(S_baseline[i]) / noise_rms[i]
        for i in range(N_ELEMENTS)
    ])
    mod_depth = np.zeros(N_ELEMENTS)
    for i in range(N_ELEMENTS):
        if S_baseline[i] > 0:
            mod_depth[i] = float(
                np.abs(S[i, :] - S_baseline[i]).max() / S_baseline[i]
            )

    return SignalField0c(
        cardiac_phase=cardiac_phase,
        S=S,
        S_baseline=S_baseline,
        S_tissue={},
        noise_rms=noise_rms,
        SNR=SNR,
        modulation_depth=mod_depth,
    )


def _element_noise() -> np.ndarray:
    noise_rms    = np.zeros(N_ELEMENTS)
    arc_rad      = np.radians(ELEMENT_ANGLE_DEG)
    r_coil_m     = (PHANTOM_RADIUS_MM + 5.0) * 1e-3
    arc_len_m    = arc_rad * r_coil_m
    wire_diam_m  = WIRE_DIAMETER_MM * 1e-3
    wire_area_m2 = np.pi * (wire_diam_m / 2) ** 2
    for i in range(N_ELEMENTS):
        wire_len_m  = ELEMENT_TURNS * arc_len_m
        R_coil      = RHO_COPPER * wire_len_m / wire_area_m2
        noise_rms[i] = float(np.sqrt(
            4.0 * K_BOLTZMANN * TEMPERATURE_K *
            R_coil * RECEIVER_BW_HZ * PREAMP_NOISE_FACTOR
        ))
    return noise_rms


def _boundary_ratio(efield: EField0c) -> float:
    bnd  = float(efield.boundary_E_mean.mean())
    tum  = float(efield.interior_tumor_E_mean.mean())
    gm   = float(efield.interior_gm_E_mean.mean())
    interior = (tum + gm) / 2.0
    return bnd / interior if interior > 0 else 0.0


def main():
    print("=" * 65)
    print("abr-nmr-phase0c — Water Content Contrast Sweep")
    print("Metatron Dynamics, Inc.")
    print("=" * 65)
    print(f"\nReference tissue (right hemisphere): water={WATER_REF*100:.0f}%")
    print(f"T1={T1_REF_MS}ms  T2*={T2STAR_REF_MS}ms  vascular={VASCULAR_FRAC*100:.0f}%")
    print(f"Sweep: left hemisphere water content varies")
    print(f"Delta range: {DELTA_WATER_MIN*100:.0f}% – {DELTA_WATER_MAX*100:.0f}%")
    print(f"Steps: {DELTA_WATER_STEPS}")
    print(f"\nAll other parameters: Phase 0c declared values")
    print(f"  B0={B0_TESLA}T  TE={TE_S*1000:.0f}ms  TR={TR_S*1000:.0f}ms  32 elements")
    print("\nBounded over D. No claim beyond D.")
    print("-" * 65)

    domain = declare_domain()

    delta_water = np.linspace(DELTA_WATER_MIN, DELTA_WATER_MAX, DELTA_WATER_STEPS)
    ratios      = np.zeros(DELTA_WATER_STEPS)
    ratios_no_R = np.zeros(DELTA_WATER_STEPS)

    for k, dw in enumerate(delta_water):
        water_left = WATER_REF - dw
        sig = _build_sweep_signal(
            domain, water_left,
            domain.n_cardiac_steps,
            domain.cardiac_phase_advance_rad,
        )
        ef      = run_operators(domain, sig, rho_base=0.3)
        ef_no_R = run_operators(domain, sig, rho_base=0.0)
        ratios[k]      = _boundary_ratio(ef)
        ratios_no_R[k] = _boundary_ratio(ef_no_R)

    # ---- Operator separation threshold -----------------------------------
    # Minimum delta_water where boundary ratio > 1.
    # This is the operator separation threshold, not a detection criterion.
    # Detection depends on a declared decision rule (ratio > declared value,
    # SNR > declared value, etc.) — that declaration belongs to Origin.
    above_one      = delta_water[ratios      > 1.0]
    above_one_no_R = delta_water[ratios_no_R > 1.0]

    sep_threshold      = float(above_one.min())      if len(above_one)      > 0 else float('nan')
    sep_threshold_no_R = float(above_one_no_R.min()) if len(above_one_no_R) > 0 else float('nan')

    print(f"\n--- Sweep Results ---")
    print(f"  Phase 0c reference point (8% contrast): "
          f"ratio={ratios[np.argmin(np.abs(delta_water - 0.08))]:.2f}×")
    print(f"\n  Operator separation threshold (minimum Δwater for ratio > 1):")
    print(f"    rho_base=0.3: Δwater ≥ {sep_threshold*100:.2f}%")
    print(f"    rho_base=0.0: Δwater ≥ {sep_threshold_no_R*100:.2f}%")
    print(f"\n  Note: detection criterion (ratio > declared threshold) is")
    print(f"  Origin's declaration. This sweep reports operator separation only.")

    # ---- Figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'abr-nmr-phase0c — Water Content Contrast Sweep\n'
        '32-element array, 1.5T, TE=30ms, TR=800ms\n'
        'Metatron Dynamics, Inc.  |  Bounded over D',
        fontsize=12
    )

    pct = delta_water * 100

    # ---- Left: ratio vs delta_water ----
    ax = axes[0]
    ax.plot(pct, ratios,      color='#FF4400', lw=2.0,
            label='rho_base=0.3 (declared)')
    ax.plot(pct, ratios_no_R, color='#2244CC', lw=1.5, ls='--',
            label='rho_base=0.0 (R disabled)')
    ax.axhline(y=1.0, color='black', lw=1.2, ls=':', label='Operator separation threshold (ratio=1)')

    # Annotate Phase 0c reference point
    idx_ref = int(np.argmin(np.abs(delta_water - 0.08)))
    ax.scatter([pct[idx_ref]], [ratios[idx_ref]],
               color='#FF4400', s=80, zorder=5)
    ax.annotate(
        f'Phase 0c reference\n8% contrast → {ratios[idx_ref]:.1f}×',
        xy=(pct[idx_ref], ratios[idx_ref]),
        xytext=(pct[idx_ref] + 1.5, ratios[idx_ref] - 2.0),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='#FF4400'),
        color='#FF4400',
    )

    # Annotate operator separation threshold
    if not np.isnan(sep_threshold):
        ax.axvline(x=sep_threshold * 100, color='#FF4400', lw=1.0, ls='--', alpha=0.6)
        ax.text(sep_threshold * 100 + 0.1, 0.5,
                f'Sep. threshold: {sep_threshold*100:.2f}%', color='#FF4400', fontsize=8)
    if not np.isnan(sep_threshold_no_R):
        ax.axvline(x=sep_threshold_no_R * 100, color='#2244CC', lw=1.0, ls='--', alpha=0.6)
        ax.text(sep_threshold_no_R * 100 + 0.1, 0.2,
                f'Sep. threshold (no R): {sep_threshold_no_R*100:.2f}%', color='#2244CC', fontsize=8)

    ax.set_xlabel('Water content difference Δ [%]')
    ax.set_ylabel('Boundary E / interior E (mean ratio)')
    ax.set_title('Boundary E / interior E ratio vs water content contrast\n(ratio > 1 = operator separation; detection criterion declared by Origin)', fontsize=9)
    ax.legend(fontsize=9)
    ax.set_xlim(0, DELTA_WATER_MAX * 100)
    ax.set_ylim(bottom=0)

    # ---- Right: signal difference vs delta_water ----
    ax = axes[1]
    # S_right fixed; S_left = S_right * (water_left / WATER_REF) * correction
    T1_s     = T1_REF_MS * 1e-3
    T2star_s = T2STAR_REF_MS * 1e-3
    sin_a    = float(np.sin(FLIP_ANGLE_RAD))
    T1_fac   = 1.0 - np.exp(-TR_S / T1_s)
    S_right_base = WATER_REF * sin_a * np.exp(-TE_S / T2star_s) * T1_fac
    S_left_base  = (WATER_REF - delta_water) * sin_a * np.exp(-TE_S / T2star_s) * T1_fac
    delta_S_pct  = (S_right_base - S_left_base) / S_right_base * 100

    ax.plot(pct, delta_S_pct, color='#228833', lw=2.0)
    ax.scatter([8.0], [float((S_right_base - (WATER_REF - 0.08) *
                sin_a * np.exp(-TE_S / T2star_s) * T1_fac) / S_right_base * 100)],
               color='#228833', s=80, zorder=5)
    ax.set_xlabel('Water content difference Δ [%]')
    ax.set_ylabel('Signal contrast ΔS/S_ref [%]')
    ax.set_title('Signal contrast vs water content difference\n'
                 '(T1, T2*, vascular fraction held constant)', fontsize=10)
    ax.set_xlim(0, DELTA_WATER_MAX * 100)
    ax.set_ylim(bottom=0)

    # Annotate linearity
    ax.text(0.05, 0.92,
            'Linear relationship:\nΔS/S ∝ Δwater (at fixed T1, T2*)',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig_path = OUTPUT_DIR / 'phase0c_water_content_sweep.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure → {fig_path}")
    print("\nBounded over D. No claim beyond D.")
    print("=" * 65)


if __name__ == '__main__':
    main()
