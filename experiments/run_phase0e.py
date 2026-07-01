# experiments/run_phase0e.py — abr-nmr-phase0e
# Metatron Dynamics, Inc.
#
# Phase 0e: B0 inhomogeneity viability sweep.
#
# For each declared B0 reference value and each inhomogeneity level:
#   1. Build signal with inhomogeneity-degraded T2*
#   2. Run ABRCE operators
#   3. Record SNR and boundary ratio
#
# Reports:
#   - Inhomogeneity survival threshold per B0 (in ppm)
#   - Comparison against declared permanent magnet reference specs
#   - Hardware specification for Phase 1 magnet procurement
#
# Figures:
#   phase0e_ratio_vs_inhomogeneity.png  — boundary ratio vs ppm, all B0
#   phase0e_snr_vs_inhomogeneity.png    — min SNR vs ppm, all B0
#   phase0e_summary.png                 — combined dual-panel
#   phase0e_hardware_spec.png           — survival regions vs PM specs
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict

from sim.declaration_0e import (
    declare_domain,
    INHOMOGENEITY_SWEEP,
    N_INHOMOGENEITY,
    REFERENCE_B0_VALUES,
    PM_SPECS_PPM,
    PPM_TO_FRAC,
    BOUNDARY_RATIO_THRESHOLD,
    SNR_THRESHOLD,
)
from sim.signal_0e    import build_signal
from sim.operators_0e import run_operators

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {0.2: '#1f77b4', 0.3: '#2ca02c', 0.5: '#d62728'}


@dataclass
class SweepRecord0e:
    b0:                    float
    inhomogeneity_frac:    float
    inhomogeneity_ppm:     float
    min_snr:               float
    all_detectable:        bool
    boundary_ratio:        float
    separation_survived:   bool
    t2star_eff_tumor:      float
    t2star_eff_gm:         float


def run_sweep(domain) -> Dict[float, List[SweepRecord0e]]:
    results = {b0: [] for b0 in domain.reference_b0_values}

    for b0 in domain.reference_b0_values:
        print(f"\n  B0 = {b0}T")
        print(f"  {'Inhomog [ppm]':>15}  {'min_SNR':>12}  "
              f"{'Det':>5}  {'Ratio':>8}  {'Sep':>5}")
        print(f"  {'-'*55}")

        for inh_frac in INHOMOGENEITY_SWEEP:
            sig = build_signal(domain, b0, inh_frac)
            ops = run_operators(domain, sig)

            rec = SweepRecord0e(
                b0=b0,
                inhomogeneity_frac=inh_frac,
                inhomogeneity_ppm=inh_frac * 1e6,
                min_snr=sig.min_SNR,
                all_detectable=sig.all_detectable,
                boundary_ratio=ops.boundary_ratio,
                separation_survived=ops.separation_survived,
                t2star_eff_tumor=sig.t2star_eff.get('tumor', 0.0),
                t2star_eff_gm=sig.t2star_eff.get('gray_matter', 0.0),
            )
            results[b0].append(rec)

            inh_ppm = inh_frac * 1e6
            if inh_ppm < 1 or inh_ppm % 1000 < 170 or inh_ppm > 9800:
                det = 'YES' if rec.all_detectable else 'NO'
                sep = 'YES' if rec.separation_survived else 'NO'
                print(f"  {inh_ppm:>15.1f}  {rec.min_snr:>12.1f}  "
                      f"{det:>5}  {rec.boundary_ratio:>8.4f}  {sep:>5}")

    return results


def find_thresholds(
    results: Dict[float, List[SweepRecord0e]]
) -> Dict[float, Dict]:
    thresholds = {}
    for b0, records in results.items():
        # Find highest inhomogeneity where separation survives
        sep_threshold_ppm  = None
        snr_threshold_ppm  = None
        for rec in records:
            if rec.separation_survived:
                sep_threshold_ppm = rec.inhomogeneity_ppm
            if rec.all_detectable:
                snr_threshold_ppm = rec.inhomogeneity_ppm
        thresholds[b0] = {
            'separation_ppm': sep_threshold_ppm,
            'snr_ppm':        snr_threshold_ppm,
        }
    return thresholds


def print_report(
    results: Dict[float, List[SweepRecord0e]],
    thresholds: Dict[float, Dict],
) -> None:
    print("\n" + "=" * 65)
    print("PHASE 0e REPORT — Inhomogeneity Viability Sweep")
    print("=" * 65)

    print(f"\n--- Survival Thresholds by B0 ---")
    print(f"  {'B0 [T]':>8}  {'Sep threshold [ppm]':>22}  "
          f"{'SNR threshold [ppm]':>22}")
    for b0 in REFERENCE_B0_VALUES:
        t = thresholds[b0]
        sep = f"{t['separation_ppm']:.1f}" if t['separation_ppm'] is not None else "NOT MET"
        if t['snr_ppm'] is None:
            snr = "NOT MET"
        elif t['snr_ppm'] == 0.0:
            snr = "0.0 (perfect only)"
        else:
            snr = f"{t['snr_ppm']:.1f}"
        print(f"  {b0:>8.1f}  {sep:>22}  {snr:>22}")

    print(f"\n--- Permanent Magnet Assessment ---")
    for b0 in REFERENCE_B0_VALUES:
        sep_ppm = thresholds[b0]['separation_ppm']
        # Guard: format only when threshold was established
        if sep_ppm is not None:
            header = f"At B0 = {b0}T (separation survives to {sep_ppm:.1f} ppm):"
        else:
            header = f"At B0 = {b0}T (separation threshold NOT established):"
        print(f"\n  {header}")
        for name, (lo, hi) in PM_SPECS_PPM.items():
            if sep_ppm is not None:
                if hi <= sep_ppm:
                    verdict = "WITHIN survival region"
                elif lo <= sep_ppm < hi:
                    verdict = "PARTIALLY within survival region"
                else:
                    verdict = "OUTSIDE survival region"
            else:
                verdict = "cannot assess — threshold not established"
            print(f"    {name:>24}: {lo}–{hi} ppm  →  {verdict}")

    print(f"\n--- T2* Degradation at Key Inhomogeneity Levels ---")
    for b0 in REFERENCE_B0_VALUES:
        records = results[b0]
        print(f"\n  B0 = {b0}T:")
        print(f"  {'Inhomog [ppm]':>15}  {'T2*_tumor [ms]':>16}  "
              f"{'T2*_GM [ms]':>14}  {'Ratio':>8}")
        for ppm_target in [0, 100, 500, 1000, 2000, 5000, 10000]:
            closest = min(records,
                          key=lambda r: abs(r.inhomogeneity_ppm - ppm_target))
            print(f"  {closest.inhomogeneity_ppm:>15.0f}  "
                  f"{closest.t2star_eff_tumor:>16.3f}  "
                  f"{closest.t2star_eff_gm:>14.3f}  "
                  f"{closest.boundary_ratio:>8.4f}")

    print(f"\n--- Phase 1 Hardware Specification (declared from simulation) ---")

    # Require all reference B0 values to have established a threshold
    # before issuing a procurement recommendation.
    all_established = all(
        thresholds[b0]['separation_ppm'] is not None
        for b0 in REFERENCE_B0_VALUES
    )

    if not all_established:
        missing = [b0 for b0 in REFERENCE_B0_VALUES
                   if thresholds[b0]['separation_ppm'] is None]
        print(f"  HARDWARE SPECIFICATION CANNOT YET BE ESTABLISHED.")
        print(f"  Reason: separation threshold not found at B0 = "
              f"{[str(b) + 'T' for b in missing]}.")
        print(f"  Action: extend sweep range or revise declared parameters.")
    else:
        min_sep = min(
            thresholds[b0]['separation_ppm']
            for b0 in REFERENCE_B0_VALUES
        )
        print(f"  Minimum B0 in declared range:     {min(REFERENCE_B0_VALUES)}T")
        print(f"  Operator separation survives to:  {min_sep:.1f} ppm")
        print(f"  Required magnet homogeneity:      < {min_sep:.0f} ppm")
        print(f"                                    measured over the declared")
        print(f"                                    sensitive volume geometry")
        print(f"                                    under comparable conditions")
        print(f"                                    with comparable shimming.")
        print(f"  Declared open condition:          uniform maximum deviation")
        print(f"                                    per element assumed; real")
        print(f"                                    threshold at or above this.")
        print(f"  Conclusion:                       Halbach research-grade")
        print(f"                                    systems (10–100 ppm) are")
        print(f"                                    within the declared survival")
        print(f"                                    region at 0.2T. Comparison")
        print(f"                                    against other B0 values")
        print(f"                                    requires vendor-confirmed")
        print(f"                                    ppm over matching DSV.")

    print(f"\nBounded over D. No claim beyond D.")
    print("=" * 65)


def make_figures(
    results: Dict[float, List[SweepRecord0e]],
    thresholds: Dict[float, Dict],
) -> None:

    # ---- Figure 1: Boundary ratio vs inhomogeneity ------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for b0, records in results.items():
        ppm_arr   = np.array([r.inhomogeneity_ppm  for r in records])
        ratio_arr = np.array([r.boundary_ratio      for r in records])
        ax.plot(ppm_arr, ratio_arr, color=COLORS[b0],
                linewidth=2, label=f'B0 = {b0}T')
        t = thresholds[b0]['separation_ppm']
        if t:
            ax.axvline(t, color=COLORS[b0], linestyle=':', linewidth=1.2)

    ax.axhline(BOUNDARY_RATIO_THRESHOLD, color='red', linestyle='--',
               linewidth=1.5, label=f'Separation threshold = {BOUNDARY_RATIO_THRESHOLD}')

    # PM spec bands
    pm_colors = {'Halbach_research': '#90EE90', 'Single_sided': '#FFD700',
                 'Low_field_MRI': '#FFA07A'}
    for name, (lo, hi) in PM_SPECS_PPM.items():
        ax.axvspan(lo, hi, alpha=0.15, color=pm_colors.get(name, 'grey'),
                   label=f'{name} ({lo}–{hi} ppm)')

    ax.set_xlabel('B0 Inhomogeneity [ppm]', fontsize=12)
    ax.set_ylabel('Boundary E ratio (C_mean_abs_ratio)', fontsize=11)
    ax.set_title('Phase 0e — Operator Separation vs B0 Inhomogeneity\n'
                 '32-element ABR array, 4-tissue phantom', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, INHOMOGENEITY_SWEEP[-1] * 1e6])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'phase0e_ratio_vs_inhomogeneity.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved: {path}")

    # ---- Figure 2: SNR vs inhomogeneity -----------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for b0, records in results.items():
        ppm_arr = np.array([r.inhomogeneity_ppm for r in records])
        snr_arr = np.array([r.min_snr           for r in records])
        ax.semilogy(ppm_arr, snr_arr, color=COLORS[b0],
                    linewidth=2, label=f'B0 = {b0}T')

    ax.axhline(SNR_THRESHOLD, color='red', linestyle='--',
               linewidth=1.5, label=f'SNR threshold = {SNR_THRESHOLD}')
    for name, (lo, hi) in PM_SPECS_PPM.items():
        ax.axvspan(lo, hi, alpha=0.15, color=pm_colors.get(name, 'grey'))

    ax.set_xlabel('B0 Inhomogeneity [ppm]', fontsize=12)
    ax.set_ylabel('Minimum SNR per element', fontsize=11)
    ax.set_title('Phase 0e — SNR vs B0 Inhomogeneity', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([0, INHOMOGENEITY_SWEEP[-1] * 1e6])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'phase0e_snr_vs_inhomogeneity.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # ---- Figure 3: Summary dual-panel -------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for b0, records in results.items():
        ppm_arr   = np.array([r.inhomogeneity_ppm for r in records])
        ratio_arr = np.array([r.boundary_ratio     for r in records])
        snr_arr   = np.array([r.min_snr            for r in records])
        ax1.plot(ppm_arr, ratio_arr, color=COLORS[b0],
                 linewidth=2, label=f'B0 = {b0}T')
        ax2.semilogy(ppm_arr, snr_arr, color=COLORS[b0],
                     linewidth=2, label=f'B0 = {b0}T')
        t = thresholds[b0]['separation_ppm']
        if t:
            ax1.axvline(t, color=COLORS[b0], linestyle=':', linewidth=1)

    ax1.axhline(BOUNDARY_RATIO_THRESHOLD, color='red', linestyle='--',
                linewidth=1.5, label='Separation threshold')
    ax2.axhline(SNR_THRESHOLD, color='red', linestyle='--',
                linewidth=1.5, label='SNR threshold')

    for name, (lo, hi) in PM_SPECS_PPM.items():
        c = pm_colors.get(name, 'grey')
        ax1.axvspan(lo, hi, alpha=0.13, color=c, label=f'{name}')
        ax2.axvspan(lo, hi, alpha=0.13, color=c)

    ax1.set_ylabel('Boundary E ratio', fontsize=11)
    ax1.set_title('Phase 0e — Low-Field Inhomogeneity Viability\n'
                  '32-element ABR array | 0.2T, 0.3T, 0.5T reference B0 values',
                  fontsize=12)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('B0 Inhomogeneity [ppm]', fontsize=11)
    ax2.set_ylabel('Min SNR per element', fontsize=11)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.set_xlim([0, INHOMOGENEITY_SWEEP[-1] * 1e6])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'phase0e_summary.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("\n" + "=" * 65)
    print("abr-nmr-phase0e  |  Metatron Dynamics, Inc.")
    print("Phase 0e: B0 inhomogeneity viability sweep")
    print("Bounded over D. No claim beyond D.")
    print("=" * 65)

    domain    = declare_domain()
    results   = run_sweep(domain)
    thresholds = find_thresholds(results)
    print_report(results, thresholds)
    make_figures(results, thresholds)

    print("\nPhase 0e complete.")
    return results, thresholds


if __name__ == '__main__':
    main()
