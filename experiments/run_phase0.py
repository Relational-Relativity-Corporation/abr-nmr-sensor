# experiments/run_phase0.py — abr-nmr-phase0
# Metatron Dynamics, Inc.
#
# Phase 0 simulation entry point.
#
# Runs the full declared chain:
#   declare_domain()
#     → build_signal()      [cardiac-modulated spin echo]
#       → build_noise_model() [SNR and detectability]
#         → parameter_sweep() [vary B0, coil geometry, BW]
#
# The parameter sweep examines detectability across the declared
# design space — identifying the minimum viable configuration
# for the Phase 0 bench prototype.
#
# Outputs:
#   experiments/outputs/phase0_report.txt   — full console report
#   experiments/outputs/phase0_snr.png      — SNR vs parameter plots
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.declaration import declare_domain
from sim.signal      import build_signal
from sim.noise       import build_noise_model, NoiseModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_PATH = os.path.join(OUTPUT_DIR, 'phase0_report.txt')
FIGURE_PATH = os.path.join(OUTPUT_DIR, 'phase0_snr.png')


class Tee:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log      = open(path, 'w', encoding='utf-8')
    def write(self, m):
        self.terminal.write(m); self.log.write(m)
    def flush(self):
        self.terminal.flush(); self.log.flush()
    def close(self):
        self.log.close()


def run():
    tee = Tee(REPORT_PATH)
    sys.stdout = tee
    t0 = time.time()

    print("=" * 65)
    print("abr-nmr-phase0 — Phase 0 Bench Prototype Simulation")
    print("Single-segment solenoid, pulsatile flow phantom")
    print("Metatron Dynamics, Inc.")
    print("=" * 65)

    # ---- Baseline run --------------------------------------------------
    print("\n[1/3] Declaring domain and computing baseline signal...")
    domain = declare_domain()
    signal = build_signal(domain)

    print("\n[2/3] Computing noise model and detectability...")
    max_frac_mod = float(np.abs(
        (np.real(signal.S) - signal.S_baseline) / signal.S_baseline
    ).max())

    noise = build_noise_model(domain, signal.S_baseline, max_frac_mod)

    # ---- Parameter sweep -----------------------------------------------
    print("\n[3/3] Parameter sweep — minimum viable configuration...")
    results = _parameter_sweep(signal, max_frac_mod)
    _render_sweep_figure(results)

    elapsed = time.time() - t0
    print(f"\nTotal run time: {elapsed:.1f}s")
    print(f"Report → {REPORT_PATH}")
    print(f"Figure → {FIGURE_PATH}")
    print("Bounded over D. No claim beyond D.")
    print("=" * 65)

    sys.stdout = tee.terminal
    tee.close()


def _parameter_sweep(signal, max_frac_mod):
    """
    Sweep key parameters and compute modulation SNR for each.
    All sweeps use the declared formula — no statistical operations.

    Swept parameters:
        B0 field strength [T]
        Coil turns (affects signal and resistance)
        Phantom diameter (affects filling factor and signal volume)
        Receiver bandwidth (affects noise floor)
    """
    from sim.declaration import (
        K_BOLTZMANN, TEMPERATURE_K, PREAMP_NOISE_FACTOR,
        GAMMA_RAD_PER_S_PER_T, PROTON_DENSITY_M3, PROTON_DENSITY_REL,
        COIL_INNER_DIAMETER_MM, COIL_LENGTH_MM, COIL_WIRE_DIAMETER_MM,
        RHO_COPPER, T2STAR_BASELINE_MS, TE_S,
    )

    HBAR   = 1.054571817e-34
    MU_0   = 4.0 * np.pi * 1e-7
    GAMMA  = GAMMA_RAD_PER_S_PER_T
    SNR_TARGET = 5.0

    def compute_mod_snr(B0, n_turns, phantom_diam_mm, bw_hz):
        """Compute modulation SNR for given parameters."""
        # Coil geometry
        coil_diam_m    = (phantom_diam_mm + 1.0) * 1e-3
        coil_len_m     = COIL_LENGTH_MM * 1e-3
        wire_diam_m    = COIL_WIRE_DIAMETER_MM * 1e-3
        wire_len_m     = n_turns * np.pi * coil_diam_m
        wire_area_m2   = np.pi * (wire_diam_m / 2)**2
        R_coil         = RHO_COPPER * wire_len_m / wire_area_m2

        phantom_r_m    = phantom_diam_mm * 1e-3 / 2
        V_phantom_m3   = np.pi * phantom_r_m**2 * coil_len_m
        A_coil_m2      = np.pi * (coil_diam_m / 2)**2
        V_coil_m3      = A_coil_m2 * coil_len_m
        eta            = V_phantom_m3 / V_coil_m3

        # Noise
        V_noise_rms    = np.sqrt(
            4 * K_BOLTZMANN * TEMPERATURE_K * R_coil * bw_hz
            * PREAMP_NOISE_FACTOR
        )

        # Signal
        omega_0        = GAMMA * B0
        N_proton       = PROTON_DENSITY_M3 * PROTON_DENSITY_REL
        M_0            = (GAMMA**2 * HBAR**2 * B0 * N_proton
                          ) / (4 * K_BOLTZMANN * TEMPERATURE_K)

        T2star_s       = T2STAR_BASELINE_MS * 1e-3
        S_base         = float(np.exp(-TE_S / T2star_s))

        V_signal       = omega_0 * MU_0 * n_turns * A_coil_m2 * M_0 * eta
        V_signal_at_TE = V_signal * S_base

        mod_snr        = max_frac_mod * V_signal_at_TE / V_noise_rms
        return float(mod_snr)

    results = {}

    # Sweep B0 [0.5T to 7T]
    B0_vals = np.linspace(0.5, 7.0, 50)
    snr_B0  = [compute_mod_snr(b, 20, 8.0, 10000) for b in B0_vals]
    results['B0'] = (B0_vals, np.array(snr_B0), 'B0 [T]',
                     'Declared B0 [T]')

    # Sweep coil turns [5 to 100]
    turns_vals = np.arange(5, 101, 5)
    snr_turns  = [compute_mod_snr(1.5, t, 8.0, 10000) for t in turns_vals]
    results['turns'] = (turns_vals.astype(float), np.array(snr_turns),
                        'Coil turns', 'Number of turns')

    # Sweep phantom diameter [3mm to 30mm]
    diam_vals  = np.linspace(3.0, 30.0, 50)
    snr_diam   = [compute_mod_snr(1.5, 20, d, 10000) for d in diam_vals]
    results['diameter'] = (diam_vals, np.array(snr_diam),
                           'Phantom diameter [mm]',
                           'Phantom inner diameter [mm]')

    # Sweep receiver bandwidth [1kHz to 100kHz]
    bw_vals    = np.linspace(1000, 100000, 50)
    snr_bw     = [compute_mod_snr(1.5, 20, 8.0, b) for b in bw_vals]
    results['bandwidth'] = (bw_vals/1000, np.array(snr_bw),
                            'Receiver BW [kHz]',
                            'Receiver bandwidth [kHz]')

    # Print sweep summary
    print(f"\n  Parameter sweep — modulation SNR (target = {SNR_TARGET})")
    print(f"\n  {'Parameter':>20}  {'Declared':>12}  {'Mod SNR':>10}  "
          f"{'Min for SNR=5':>16}")

    from sim.declaration import (
        B0_TESLA, COIL_TURNS, PHANTOM_DIAMETER_MM, RECEIVER_BW_HZ
    )

    for name, (xvals, snrs, xlabel, _) in results.items():
        declared_idx = 0
        if name == 'B0':
            declared_x = B0_TESLA
            declared_idx = int(np.argmin(np.abs(xvals - declared_x)))
        elif name == 'turns':
            declared_x = float(COIL_TURNS)
            declared_idx = int(np.argmin(np.abs(xvals - declared_x)))
        elif name == 'diameter':
            declared_x = float(PHANTOM_DIAMETER_MM)
            declared_idx = int(np.argmin(np.abs(xvals - declared_x)))
        elif name == 'bandwidth':
            declared_x = RECEIVER_BW_HZ / 1000.0
            declared_idx = int(np.argmin(np.abs(xvals - declared_x)))

        declared_snr = snrs[declared_idx]

        # Find minimum x for SNR=5
        above = xvals[snrs >= SNR_TARGET]
        min_for_target = float(above[0]) if len(above) > 0 else float('inf')

        print(f"  {xlabel:>20}  {float(xvals[declared_idx]):>12.2f}  "
              f"{declared_snr:>10.4f}  {min_for_target:>16.2f}")

    return results


def _render_sweep_figure(results):
    """
    Render parameter sweep results.
    Each panel: modulation SNR vs one parameter.
    Declared value marked. SNR=5 threshold line shown.
    """
    SNR_TARGET = 5.0
    from sim.declaration import (
        B0_TESLA, COIL_TURNS, PHANTOM_DIAMETER_MM, RECEIVER_BW_HZ
    )
    declared_vals = {
        'B0': B0_TESLA,
        'turns': float(COIL_TURNS),
        'diameter': float(PHANTOM_DIAMETER_MM),
        'bandwidth': RECEIVER_BW_HZ / 1000.0,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, (name, (xvals, snrs, xlabel, title)) in zip(axes, results.items()):
        ax.plot(xvals, snrs, color='#00B4D8', linewidth=2.0)
        ax.axhline(y=SNR_TARGET, color='#FF6B35', linewidth=1.5,
                   linestyle='--', label=f'Detection threshold (SNR={SNR_TARGET})')
        ax.axvline(x=declared_vals[name], color='#FFD700', linewidth=1.5,
                   linestyle=':', label=f'Declared value ({declared_vals[name]:.1f})')

        # Shade detectable region
        detectable = snrs >= SNR_TARGET
        if np.any(detectable):
            ax.fill_between(xvals, 0, snrs, where=detectable,
                           alpha=0.15, color='#00B4D8',
                           label='Detectable region')

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Modulation SNR per step', fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        'abr-nmr-phase0 — Modulation SNR Parameter Sweep\n'
        'Single-segment solenoid, pulsatile phantom\n'
        'Metatron Dynamics, Inc.  |  Bounded over D',
        fontsize=11, y=1.02
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved → {FIGURE_PATH}")
    plt.close(fig)


if __name__ == '__main__':
    run()
