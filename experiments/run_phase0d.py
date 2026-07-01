# experiments/run_phase0d.py — abr-nmr-phase0d
# Metatron Dynamics, Inc.
#
# Phase 0d: Low-field B0 viability sweep.
#
# For each declared B0 in the sweep:
#   1. Build signal at that B0
#   2. Run ABRCE operators
#   3. Record SNR and boundary ratio
#
# Reports:
#   - SNR survival threshold: lowest B0 with all elements detectable
#   - Operator separation threshold: lowest B0 with boundary ratio > 1
#   - Performance within declared permanent magnet range (0.2T–0.5T)
#
# Figures:
#   phase0d_snr_vs_b0.png       — per-element min SNR vs B0
#   phase0d_ratio_vs_b0.png     — boundary ratio vs B0
#   phase0d_summary.png         — combined: SNR + ratio on shared B0 axis
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataclasses import dataclass
from typing import List

from sim.declaration_0d import (
    declare_domain,
    B0_SWEEP,
    N_B0,
    PM_RANGE_LOW_T,
    PM_RANGE_HIGH_T,
    SNR_THRESHOLD,
    BOUNDARY_RATIO_THRESHOLD,
)
from sim.signal_0d    import build_signal
from sim.operators_0d import run_operators

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class SweepRecord:
    b0:                    float
    min_snr:               float
    all_detectable:        bool
    boundary_ratio:        float
    separation_survived:   bool
    boundary_E_mean:       float
    interior_tumor_E_mean: float
    interior_gm_E_mean:    float
    rho_mean:              float
    v_signal_peak:         float


def run_sweep(domain) -> List[SweepRecord]:
    records = []
    print(f"\nRunning Phase 0d sweep: {N_B0} B0 values "
          f"[{B0_SWEEP[0]:.4f}T – {B0_SWEEP[-1]:.4f}T]")
    print("-" * 65)

    for i, b0 in enumerate(B0_SWEEP):
        sig = build_signal(domain, b0)
        ops = run_operators(domain, sig)

        rec = SweepRecord(
            b0=b0,
            min_snr=sig.min_SNR,
            all_detectable=sig.all_detectable,
            boundary_ratio=ops.boundary_ratio,
            separation_survived=ops.separation_survived,
            boundary_E_mean=ops.boundary_E_mean,
            interior_tumor_E_mean=ops.interior_tumor_E_mean,
            interior_gm_E_mean=ops.interior_gm_E_mean,
            rho_mean=ops.rho_mean,
            v_signal_peak=sig.V_signal_peak,
        )
        records.append(rec)

        # Progress every 10 steps
        if (i + 1) % 10 == 0 or i == 0 or i == N_B0 - 1:
            det_str  = 'YES' if rec.all_detectable  else 'NO '
            sep_str  = 'YES' if rec.separation_survived else 'NO '
            print(f"  B0={b0:6.4f}T  "
                  f"min_SNR={rec.min_snr:10.1f}  detectable={det_str}  "
                  f"ratio={rec.boundary_ratio:6.3f}  sep={sep_str}")

    return records


def find_thresholds(records: List[SweepRecord]):
    """
    Find minimum B0 at which each condition is met.
    Scanning from high B0 downward — find lowest B0 where condition holds.
    """
    snr_threshold_b0   = None
    ratio_threshold_b0 = None

    for rec in records:
        if rec.all_detectable:
            snr_threshold_b0 = rec.b0
        if rec.separation_survived:
            ratio_threshold_b0 = rec.b0

    return snr_threshold_b0, ratio_threshold_b0


def print_report(records: List[SweepRecord],
                 snr_b0: float,
                 ratio_b0: float,
                 domain) -> None:
    print("\n" + "=" * 65)
    print("PHASE 0d SWEEP REPORT")
    print("=" * 65)

    print(f"\n--- SNR Survival Threshold ---")
    if snr_b0 is not None:
        print(f"  Minimum B0 for all-element detectability: {snr_b0:.4f}T")
        print(f"  (SNR ≥ {SNR_THRESHOLD} on all 32 elements)")
    else:
        print(f"  SNR threshold NOT met at any declared B0.")
        print(f"  HARD STOP: declared parameters insufficient.")

    print(f"\n--- Operator Separation Threshold ---")
    if ratio_b0 is not None:
        print(f"  Minimum B0 for boundary detection: {ratio_b0:.4f}T")
        print(f"  (boundary ratio > {BOUNDARY_RATIO_THRESHOLD})")
    else:
        print(f"  Operator separation NOT maintained at any declared B0.")
        print(f"  HARD STOP.")

    print(f"\n--- Permanent Magnet Range Assessment ---")
    pm_records = [r for r in records
                  if PM_RANGE_LOW_T <= r.b0 <= PM_RANGE_HIGH_T]
    if pm_records:
        r_low  = pm_records[0]
        r_high = pm_records[-1]
        print(f"  B0 range assessed: "
              f"{r_low.b0:.3f}T – {r_high.b0:.3f}T")
        print(f"  At {r_low.b0:.3f}T:  "
              f"min_SNR={r_low.min_snr:.1f}  "
              f"boundary_ratio={r_low.boundary_ratio:.3f}  "
              f"detectable={r_low.all_detectable}  "
              f"sep={r_low.separation_survived}")
        print(f"  At {r_high.b0:.3f}T: "
              f"min_SNR={r_high.min_snr:.1f}  "
              f"boundary_ratio={r_high.boundary_ratio:.3f}  "
              f"detectable={r_high.all_detectable}  "
              f"sep={r_high.separation_survived}")

    print(f"\n--- Key Values at Reference Points ---")
    ref_b0s = [0.1, 0.2, 0.3, 0.5, 1.0, 1.5]
    print(f"  {'B0 [T]':>8}  {'min_SNR':>12}  "
          f"{'Detectable':>12}  {'Ratio':>8}  {'Sep':>5}")
    for rb in ref_b0s:
        # Find closest record
        closest = min(records, key=lambda r: abs(r.b0 - rb))
        det = 'YES' if closest.all_detectable else 'NO'
        sep = 'YES' if closest.separation_survived else 'NO'
        print(f"  {closest.b0:>8.4f}  {closest.min_snr:>12.1f}  "
              f"{det:>12}  {closest.boundary_ratio:>8.3f}  {sep:>5}")

    print(f"\n--- Declared Open Conditions ---")
    print(f"  T1, T2* held at 1.5T values → sweep is conservative on SNR")
    print(f"  Perfect B0 homogeneity assumed → sweep overstates low-field perf")
    print(f"  Net conservatism direction: Phase 1 hardware required to resolve")
    print(f"\nBounded over D. No claim beyond D.")
    print("=" * 65)


def make_figures(records: List[SweepRecord],
                 snr_b0: float,
                 ratio_b0: float) -> None:

    b0_arr    = np.array([r.b0          for r in records])
    snr_arr   = np.array([r.min_snr     for r in records])
    ratio_arr = np.array([r.boundary_ratio for r in records])

    # ---- Figure 1: SNR vs B0 ----------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(b0_arr, snr_arr, 'b-', linewidth=2, label='Min SNR (32 elements)')
    ax.axhline(SNR_THRESHOLD, color='r', linestyle='--',
               linewidth=1.5, label=f'SNR threshold = {SNR_THRESHOLD}')
    if snr_b0 is not None:
        ax.axvline(snr_b0, color='orange', linestyle=':',
                   linewidth=1.5,
                   label=f'SNR survival threshold = {snr_b0:.3f}T')
    ax.axvspan(PM_RANGE_LOW_T, PM_RANGE_HIGH_T, alpha=0.12, color='green',
               label=f'Permanent magnet range ({PM_RANGE_LOW_T}–{PM_RANGE_HIGH_T}T)')

    # Reference B0 ∝ B0² line from 1.5T point
    snr_15 = snr_arr[-1]
    b0_ref  = np.array([b0_arr[0], b0_arr[-1]])
    snr_ref = snr_15 * (b0_ref / 1.5) ** 2
    ax.loglog(b0_ref, snr_ref, 'b--', linewidth=1, alpha=0.4,
              label='SNR ∝ B0² reference')

    ax.set_xlabel('B0 [T]', fontsize=12)
    ax.set_ylabel('Minimum SNR per element per step', fontsize=12)
    ax.set_title('Phase 0d — SNR vs B0\n'
                 '32-element array, 4-tissue phantom, 1200 cardiac steps',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([b0_arr[0] * 0.9, b0_arr[-1] * 1.1])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'phase0d_snr_vs_b0.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved: {path}")

    # ---- Figure 2: Boundary ratio vs B0 -----------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(b0_arr, ratio_arr, 'g-', linewidth=2,
                label='Boundary E ratio (boundary / max interior)')
    ax.axhline(BOUNDARY_RATIO_THRESHOLD, color='r', linestyle='--',
               linewidth=1.5,
               label=f'Separation threshold = {BOUNDARY_RATIO_THRESHOLD}')
    if ratio_b0 is not None:
        ax.axvline(ratio_b0, color='orange', linestyle=':',
                   linewidth=1.5,
                   label=f'Separation threshold B0 = {ratio_b0:.3f}T')
    ax.axvspan(PM_RANGE_LOW_T, PM_RANGE_HIGH_T, alpha=0.12, color='green',
               label=f'Permanent magnet range ({PM_RANGE_LOW_T}–{PM_RANGE_HIGH_T}T)')

    ax.set_xlabel('B0 [T]', fontsize=12)
    ax.set_ylabel('Boundary E ratio', fontsize=12)
    ax.set_title('Phase 0d — Operator Separation vs B0\n'
                 'Boundary E / max(tumor interior E, GM interior E)',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([b0_arr[0] * 0.9, b0_arr[-1] * 1.1])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'phase0d_ratio_vs_b0.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # ---- Figure 3: Summary (dual-panel) ------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: SNR
    ax1.loglog(b0_arr, snr_arr, 'b-', linewidth=2)
    ax1.axhline(SNR_THRESHOLD, color='r', linestyle='--', linewidth=1.5,
                label=f'SNR threshold = {SNR_THRESHOLD}')
    if snr_b0 is not None:
        ax1.axvline(snr_b0, color='orange', linestyle=':', linewidth=1.5,
                    label=f'SNR survival B0 = {snr_b0:.3f}T')
    ax1.axvspan(PM_RANGE_LOW_T, PM_RANGE_HIGH_T, alpha=0.12, color='green',
                label='Permanent magnet range')
    ax1.set_ylabel('Min SNR per element', fontsize=11)
    ax1.set_title('Phase 0d — Low-Field Viability Sweep\n'
                  '32-element ABR array, 4-tissue phantom, '
                  f'{len(b0_arr)}-step B0 sweep '
                  f'[{b0_arr[0]:.3f}T – {b0_arr[-1]:.3f}T]',
                  fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, which='both', alpha=0.3)

    # Bottom: ratio
    ax2.semilogx(b0_arr, ratio_arr, 'g-', linewidth=2)
    ax2.axhline(BOUNDARY_RATIO_THRESHOLD, color='r', linestyle='--',
                linewidth=1.5,
                label=f'Separation threshold = {BOUNDARY_RATIO_THRESHOLD}')
    if ratio_b0 is not None:
        ax2.axvline(ratio_b0, color='purple', linestyle=':', linewidth=1.5,
                    label=f'Separation B0 = {ratio_b0:.3f}T')
    ax2.axvspan(PM_RANGE_LOW_T, PM_RANGE_HIGH_T, alpha=0.12, color='green',
                label='Permanent magnet range')
    ax2.set_xlabel('B0 [T]', fontsize=11)
    ax2.set_ylabel('Boundary E ratio', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.set_xlim([b0_arr[0] * 0.9, b0_arr[-1] * 1.1])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'phase0d_summary.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("\n" + "=" * 65)
    print("abr-nmr-phase0d  |  Metatron Dynamics, Inc.")
    print("Phase 0d: Low-field B0 viability sweep")
    print("Bounded over D. No claim beyond D.")
    print("=" * 65)

    domain  = declare_domain()
    records = run_sweep(domain)

    snr_b0, ratio_b0 = find_thresholds(records)
    print_report(records, snr_b0, ratio_b0, domain)
    make_figures(records, snr_b0, ratio_b0)

    print("\nPhase 0d complete.")
    return records, snr_b0, ratio_b0


if __name__ == '__main__':
    main()
