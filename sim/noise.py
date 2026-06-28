# sim/noise.py — abr-nmr-phase0
# Metatron Dynamics, Inc.
#
# Thermal noise model for the Phase 0 solenoid receiver.
#
# The noise floor determines whether the declared cardiac pulsatility
# signal is detectable. All noise sources are derived from declared
# physical constants — no empirical fitting, no statistical proxies.
#
# Noise sources declared:
#
#   1. Johnson-Nyquist thermal noise from coil resistance
#      V_noise_coil^2 = 4 * k_B * T * R_coil * BW
#
#   2. Preamplifier noise contribution
#      V_noise_total^2 = V_noise_coil^2 * F_preamp
#      where F_preamp is the noise factor (linear, from declared NF)
#
#   3. Sample noise (radiation resistance from lossy tissue)
#      For a bench phantom at room temperature, sample noise is
#      comparable to coil noise. Declared as a multiplier on coil noise.
#      At clinical field strengths (1.5T), sample noise dominates
#      for large samples; for the 8mm tube phantom, coil noise dominates.
#
# SNR per cardiac step:
#   SNR = S(t) / V_noise_rms
#
# SNR after N steps of coherent averaging:
#   SNR_N = SNR_1 * sqrt(N)
#   (valid only if the signal is phase-coherent across steps)
#
# The cardiac-gated acquisition ensures phase coherence across steps
# at the same cardiac phase. The operators do not average — they act
# on the full per-step signal. SNR_1 is the admissible quantity.
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass
from sim.declaration import (
    DeclaredDomain,
    K_BOLTZMANN,
    TEMPERATURE_K,
    COIL_RESISTANCE_OHM,
    RECEIVER_BW_HZ,
    PREAMP_NOISE_FACTOR,
    B0_TESLA,
    GAMMA_RAD_PER_S_PER_T,
    PHANTOM_VOLUME_M3,
    PROTON_DENSITY_M3,
    PROTON_DENSITY_REL,
    FILLING_FACTOR,
    COIL_TURNS,
    COIL_INNER_DIAMETER_MM,
    COIL_LENGTH_MM,
    LARMOR_FREQ_MHZ,
)


@dataclass
class NoiseModel:
    """
    Declared noise model for the Phase 0 solenoid receiver.

    Attributes
    ----------
    V_noise_coil_rms : float
        RMS thermal noise voltage from coil resistance [V].
        V_noise_coil = sqrt(4 * k_B * T * R_coil * BW)

    V_noise_total_rms : float
        RMS total noise voltage including preamplifier [V].
        V_noise_total = V_noise_coil * sqrt(F_preamp)

    V_signal_peak : float
        Peak signal voltage at the receiver input [V].
        Derived from the equilibrium magnetization and coil geometry.

    SNR_per_step : float
        Signal-to-noise ratio per single cardiac evolution step.
        SNR = V_signal_peak / V_noise_total_rms

    SNR_after_N : callable
        SNR after N coherent averages: SNR_per_step * sqrt(N).
        Declared for reference — operators do not average.

    n_steps_for_snr : int
        Number of cardiac steps required to reach SNR=5 by averaging.
        Declared detection threshold: SNR=5 (standard NMR convention).

    modulation_depth : float
        Fractional signal change from cardiac pulsatility.
        delta_S / S_baseline.

    modulation_snr_per_step : float
        SNR of the cardiac modulation signal per step.
        SNR_mod = modulation_depth * V_signal_peak / V_noise_total_rms

    detectable_in_N_steps : int
        Steps required for modulation SNR to reach 5 by averaging.
    """
    V_noise_coil_rms:       float
    V_noise_total_rms:      float
    V_signal_peak:          float
    SNR_per_step:           float
    n_steps_for_snr:        int
    modulation_depth:       float
    modulation_snr_per_step: float
    detectable_in_N_steps:  int


def build_noise_model(
    domain: DeclaredDomain,
    S_baseline: float,
    max_frac_mod: float,
) -> NoiseModel:
    """
    Build the declared noise model for the Phase 0 solenoid.

    Parameters
    ----------
    domain : DeclaredDomain
    S_baseline : float
        Baseline signal amplitude (dimensionless, relative to PD).
    max_frac_mod : float
        Maximum fractional cardiac modulation |delta_S / S_baseline|.

    Returns
    -------
    NoiseModel
    """

    # ---- Johnson-Nyquist thermal noise [V_rms] -----------------------
    # V_noise^2 = 4 * k_B * T * R * BW
    V_noise_coil_var = (
        4.0 * K_BOLTZMANN *
        domain.temperature_k *
        domain.coil_resistance_ohm *
        domain.receiver_bw_hz
    )
    V_noise_coil_rms = float(np.sqrt(V_noise_coil_var))

    # ---- Total noise including preamplifier --------------------------
    # Noise factor multiplies the noise power
    V_noise_total_rms = float(
        V_noise_coil_rms * np.sqrt(domain.preamp_noise_factor)
    )

    # ---- Signal voltage at receiver input [V] ------------------------
    # The NMR signal voltage induced in a solenoid is:
    # V_signal = omega_0 * M_0 * eta * mu_0 * n_turns * A_coil / L_coil
    #
    # where:
    #   omega_0  = Larmor angular frequency [rad/s]
    #   M_0      = equilibrium magnetization [A/m]
    #   eta      = filling factor
    #   n_turns  = number of coil turns
    #   A_coil   = coil cross-section area [m^2]
    #   L_coil   = coil length [m]
    #
    # Equilibrium magnetization M_0:
    # M_0 = (gamma^2 * hbar^2 * B0 * N_protons) / (4 * k_B * T)
    # where N_protons = proton density * volume
    #
    # Using the high-temperature approximation (valid at clinical fields)
    # and simplified NMR signal voltage formula:
    # V_signal ~ omega_0 * mu_0 * M_0 * eta * V_coil / (something)
    #
    # More directly, use the declared SNR formula from Hoult & Richards (1976):
    # SNR = (omega_0^(7/4) * V_sample * C * PD) / (F * T * R * BW)^(1/2)
    # where C contains physical constants.
    #
    # Implement the full chain:

    # Physical constants
    HBAR      = 1.054571817e-34   # [J*s]
    MU_0      = 4.0 * np.pi * 1e-7  # [H/m]
    GAMMA_RAD = GAMMA_RAD_PER_S_PER_T

    omega_0   = GAMMA_RAD * B0_TESLA   # Larmor [rad/s]

    # Equilibrium magnetization [A/m]
    # M_0 = (gamma^2 * hbar^2 * B0 * rho_proton * PD) / (4 * k_B * T)
    N_proton_per_m3 = PROTON_DENSITY_M3 * PROTON_DENSITY_REL
    M_0 = (
        GAMMA_RAD**2 * HBAR**2 * B0_TESLA *
        N_proton_per_m3
    ) / (4.0 * K_BOLTZMANN * TEMPERATURE_K)

    # Coil geometry
    A_coil_m2 = np.pi * (COIL_INNER_DIAMETER_MM * 1e-3 / 2)**2
    L_coil_m  = COIL_LENGTH_MM * 1e-3
    n_turns   = float(COIL_TURNS)

    # Induced EMF in solenoid from precessing magnetization:
    # V_emf = omega_0 * mu_0 * n_turns * A_coil * M_0 * eta
    # (from Faraday's law applied to precessing M in solenoid)
    V_signal_peak = float(
        omega_0 * MU_0 * n_turns * A_coil_m2 * M_0 * domain.filling_factor
    )

    # Apply S_baseline as the fractional signal at declared TE and flip angle
    # (the above gives the peak FID; spin echo at TE reduces by exp(-TE/T2*)
    # which is already captured in S_baseline)
    V_signal_at_TE = V_signal_peak * S_baseline

    # ---- SNR per cardiac step ----------------------------------------
    SNR_per_step = float(V_signal_at_TE / V_noise_total_rms)

    # Steps to reach SNR=5 by coherent averaging
    SNR_TARGET = 5.0
    if SNR_per_step >= SNR_TARGET:
        n_steps_for_snr = 1
    else:
        n_steps_for_snr = int(np.ceil((SNR_TARGET / SNR_per_step)**2))

    # ---- Modulation SNR ----------------------------------------------
    # The cardiac modulation signal is:
    # delta_V = max_frac_mod * V_signal_at_TE
    delta_V = max_frac_mod * V_signal_at_TE
    modulation_snr_per_step = float(delta_V / V_noise_total_rms)

    if modulation_snr_per_step >= SNR_TARGET:
        detectable_in_N_steps = 1
    else:
        detectable_in_N_steps = int(
            np.ceil((SNR_TARGET / modulation_snr_per_step)**2)
        )

    noise = NoiseModel(
        V_noise_coil_rms=V_noise_coil_rms,
        V_noise_total_rms=V_noise_total_rms,
        V_signal_peak=V_signal_peak,
        SNR_per_step=SNR_per_step,
        n_steps_for_snr=n_steps_for_snr,
        modulation_depth=float(max_frac_mod),
        modulation_snr_per_step=modulation_snr_per_step,
        detectable_in_N_steps=detectable_in_N_steps,
    )

    _print_noise_report(noise)
    return noise


def _print_noise_report(noise: NoiseModel) -> None:
    print("\n--- Noise Model Summary ---")
    print(f"  V_noise_coil_rms:     {noise.V_noise_coil_rms:.4e} V")
    print(f"  V_noise_total_rms:    {noise.V_noise_total_rms:.4e} V")
    print(f"  V_signal_peak (FID):  {noise.V_signal_peak:.4e} V")
    print(f"\n  SNR per cardiac step: {noise.SNR_per_step:.2f}")
    if noise.SNR_per_step >= 5.0:
        print(f"  Steps for SNR=5:      1  (single step sufficient)")
    else:
        print(f"  Steps for SNR=5:      {noise.n_steps_for_snr:,}")

    print(f"\n  Cardiac modulation depth:    "
          f"{noise.modulation_depth:.6f}  "
          f"({noise.modulation_depth*100:.4f}%)")
    print(f"  Modulation SNR per step:     "
          f"{noise.modulation_snr_per_step:.4f}")
    if noise.detectable_in_N_steps == 1:
        print(f"  Steps to detect modulation:  1  (single step sufficient)")
    else:
        print(f"  Steps to detect modulation:  "
              f"{noise.detectable_in_N_steps:,}")

    print(f"\n  --- DETECTABILITY VERDICT ---")
    if noise.modulation_snr_per_step >= 5.0:
        print(f"  DETECTABLE in single cardiac step.")
        print(f"  Operators act on unaveraged per-step signal. ADMISSIBLE.")
    elif noise.detectable_in_N_steps <= 1200:
        print(f"  Detectable in {noise.detectable_in_N_steps} cardiac steps "
              f"(within 1200-step declared window).")
        print(f"  Note: averaging reduces relational resolution.")
        print(f"  Declare averaging as an open condition.")
    else:
        print(f"  NOT detectable within declared 1200-step window.")
        print(f"  Requires {noise.detectable_in_N_steps:,} steps.")
        print(f"  HARD STOP: declared parameters need revision.")
        print(f"  Candidate revisions: higher B0, larger phantom,")
        print(f"  longer coil, more turns, lower noise figure.")
