# sim/signal.py — abr-nmr-phase0
# Metatron Dynamics, Inc.
#
# Spin echo signal model for the Phase 0 solenoid receiver.
#
# Produces:
#   - cardiac_phase[n_steps]: declared phase at each evolution step
#   - delta_B[n_steps]:       susceptibility field modulated by cardiac phase
#   - T2star[n_steps]:        effective T2* at each cardiac step
#   - S[n_steps]:             spin echo signal amplitude (complex)
#
# Signal model (spin echo, steady state):
#   S(t) = PD * sin(alpha) * exp(-TE / T2*(t)) * (1 - exp(-TR / T1))
#          * exp(i * phi_0)
#
# where:
#   T2*(t) is modulated by cardiac pulsatility through delta_B(t)
#   phi_0 is the receiver reference phase (declared zero)
#
# The cardiac pulsatility modulates the oxygenation state of blood
# in the phantom. Higher oxygenation → lower deoxy-Hb fraction →
# lower delta_chi → lower delta_B → longer T2* → higher signal.
#
# This is the same physical chain as the fMRI simulation, now
# applied to a single solenoid element rather than a spatial lattice.
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass
from sim.declaration import (
    DeclaredDomain,
    CARDIAC_PHASE_ADVANCE_RAD,
    A_OXYGENATION,
    DELTA_CHI_DEOXY,
    GEOMETRY_FACTOR,
    B0_TESLA,
    BLOOD_VOLUME_FRACTION,
    GAMMA_RAD_PER_S_PER_T,
    T1_MS,
    T2STAR_BASELINE_MS,
    PROTON_DENSITY_REL,
    TR_S,
    TE_S,
    FLIP_ANGLE_RAD,
)


@dataclass
class SignalField:
    """
    Declared signal outputs for the Phase 0 solenoid receiver.

    Attributes
    ----------
    cardiac_phase : float64 array [n_steps]
        Declared cardiac phase at each evolution step [rad].
        phi(t) = (t * CARDIAC_PHASE_ADVANCE_RAD) mod 2pi

    oxygenation : float64 array [n_steps]
        Fractional oxygenation state at each step.
        Modulated sinusoidally by cardiac phase.
        oxygenation(t) = A_OXYGENATION * sin(phi(t))
        Peak at systole (phi = pi/2), minimum at diastole.

    delta_chi : float64 array [n_steps]
        Effective susceptibility difference at each step [dimensionless].
        delta_chi(t) = DELTA_CHI_DEOXY * (1 - oxygenation(t))
                       * BLOOD_VOLUME_FRACTION
        Reduced when oxygenation is high (systole).

    delta_B : float64 array [n_steps]
        Susceptibility-driven field perturbation [T].
        delta_B(t) = delta_chi(t) * B0 * GEOMETRY_FACTOR

    T2star : float64 array [n_steps]
        Effective T2* at each cardiac step [ms].
        1/T2*(t) = 1/T2*_baseline + gamma * |delta_B(t)|

    S : complex128 array [n_steps]
        Spin echo signal amplitude at each cardiac step.
        S(t) = PD * sin(alpha) * exp(-TE/T2*(t)) * (1-exp(-TR/T1))
        Phase declared zero (receiver reference phase).

    S_baseline : float64 (scalar)
        Signal amplitude at zero susceptibility perturbation.
        Reference for measuring cardiac modulation depth.
    """
    cardiac_phase:  np.ndarray   # [n_steps] float64
    oxygenation:    np.ndarray   # [n_steps] float64
    delta_chi:      np.ndarray   # [n_steps] float64
    delta_B:        np.ndarray   # [n_steps] float64
    T2star:         np.ndarray   # [n_steps] float64 [ms]
    S:              np.ndarray   # [n_steps] complex128
    S_baseline:     float


def build_signal(domain: DeclaredDomain) -> SignalField:
    """
    Compute the spin echo signal at each cardiac evolution step.

    The declared physical chain:
        cardiac phase phi(t)
            → oxygenation state
            → effective delta_chi(t)
            → delta_B(t)
            → T2*(t)
            → S(t)

    All operations are exact formula evaluations.
    No statistical operations. No approximations.

    Parameters
    ----------
    domain : DeclaredDomain

    Returns
    -------
    SignalField
    """
    n  = domain.n_cardiac_steps
    advance = float(CARDIAC_PHASE_ADVANCE_RAD)

    # ---- Cardiac phase sequence [n_steps] ----------------------------
    # phi(t) = (t * advance) mod 2pi
    # Computed through float32 intermediate to match declared precision
    t_idx         = np.arange(n, dtype=np.float32)
    cardiac_phase = (
        (t_idx * advance).astype(np.float64) % (2.0 * np.pi)
    )

    # ---- Oxygenation state [n_steps] ---------------------------------
    # Sinusoidal modulation by cardiac phase.
    # Maximum oxygenation at systole onset (phi = 0 → sin = 0,
    # then rising to peak at phi = pi/2).
    # Declared: oxygenation = A_OXYGENATION * sin(phi(t))
    # Range: [-A, +A] — signed, positive = net oxygenation above baseline
    oxygenation = A_OXYGENATION * np.sin(cardiac_phase)

    # ---- Effective delta_chi [n_steps] --------------------------------
    # delta_chi(t) = DELTA_CHI_DEOXY * (1 - oxygenation(t))
    #                * BLOOD_VOLUME_FRACTION
    # When oxygenation is high (systole), deoxy-Hb fraction falls,
    # reducing the effective susceptibility difference.
    delta_chi = (
        DELTA_CHI_DEOXY *
        (1.0 - oxygenation) *
        BLOOD_VOLUME_FRACTION
    )

    # ---- Susceptibility field perturbation [n_steps] -----------------
    # delta_B(t) = delta_chi(t) * B0 * GEOMETRY_FACTOR
    delta_B = delta_chi * B0_TESLA * GEOMETRY_FACTOR

    # ---- T2* at each cardiac step [n_steps, ms] ----------------------
    # 1/T2*(t) = 1/T2*_baseline + gamma * |delta_B(t)|
    T2star_baseline_s = T2STAR_BASELINE_MS * 1e-3
    R2star_baseline   = 1.0 / T2star_baseline_s
    R2star            = R2star_baseline + GAMMA_RAD_PER_S_PER_T * np.abs(delta_B)
    T2star_s          = 1.0 / R2star
    T2star_ms         = T2star_s * 1e3

    # ---- Static decay factors ----------------------------------------
    T1_s      = T1_MS * 1e-3
    T1_factor = 1.0 - np.exp(-TR_S / T1_s)   # T1 recovery (scalar)
    PD        = float(PROTON_DENSITY_REL)
    sin_alpha = float(np.sin(FLIP_ANGLE_RAD))

    # ---- Baseline signal (zero susceptibility) -----------------------
    T2decay_baseline = float(np.exp(-TE_S / T2star_baseline_s))
    S_baseline       = PD * sin_alpha * T2decay_baseline * T1_factor

    # ---- Spin echo signal [n_steps] ----------------------------------
    # S(t) = PD * sin(alpha) * exp(-TE/T2*(t)) * (1 - exp(-TR/T1))
    # Phase = 0 (declared receiver reference phase)
    T2decay = np.exp(-TE_S / T2star_s)   # [n_steps]
    S_real  = PD * sin_alpha * T2decay * T1_factor
    S       = S_real.astype(np.complex128)   # imaginary part = 0 at phi_0 = 0

    signal = SignalField(
        cardiac_phase=cardiac_phase,
        oxygenation=oxygenation,
        delta_chi=delta_chi,
        delta_B=delta_B,
        T2star=T2star_ms,
        S=S,
        S_baseline=float(S_baseline),
    )

    _print_signal_report(signal)
    return signal


def _print_signal_report(sig: SignalField) -> None:
    print("\n--- Signal Field Summary ---")
    print(f"  S_baseline:        {sig.S_baseline:.6f}  (zero susceptibility)")
    print(f"  S range:           "
          f"{float(np.real(sig.S).min()):.6f} – "
          f"{float(np.real(sig.S).max()):.6f}")

    frac_mod = (np.real(sig.S) - sig.S_baseline) / sig.S_baseline
    print(f"  Fractional ΔS/S:   "
          f"{float(frac_mod.min()):.6f} – "
          f"{float(frac_mod.max()):.6f}")
    print(f"  Max |ΔS/S|:        {float(np.abs(frac_mod).max()):.6f}  "
          f"({float(np.abs(frac_mod).max())*100:.4f}%)")

    print(f"\n  delta_B range:     "
          f"{float(sig.delta_B.min()):.4e} – "
          f"{float(sig.delta_B.max()):.4e} T")
    print(f"  T2* range:         "
          f"{float(sig.T2star.min()):.4f} – "
          f"{float(sig.T2star.max()):.4f} ms")
    print(f"  T2* baseline:      {T2STAR_BASELINE_MS} ms")
    print(f"  Max ΔT2*:          "
          f"{float(sig.T2star.max() - T2STAR_BASELINE_MS):.4f} ms")

    print("\n  Signal declared. Noise analysis follows.")
