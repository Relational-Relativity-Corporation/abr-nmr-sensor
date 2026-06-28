# sim/declaration.py — abr-nmr-phase0
# Metatron Dynamics, Inc.
#
# Domain declaration for the Phase 0 NMR bench prototype simulation.
#
# Declares:
#   - Phantom geometry: cylindrical tube of deoxygenated saline
#   - Solenoid coil geometry: single segment surrounding phantom
#   - Physical constants: B0, Larmor frequency, T1, T2*, proton density
#   - Cardiac parameters: heart rate, phase advance, pulsatile flow
#   - Susceptibility parameters: delta_chi_deoxy, geometry factor
#   - Thermal noise: Johnson-Nyquist from coil resistance and
#     preamplifier noise figure
#   - Signal chain: spin echo amplitude per cardiac step
#
# The declared domain D is the spin field within the solenoid
# sensitivity volume at each cardiac evolution step.
#
# The question this simulation answers:
#   Is the declared cardiac pulsatility-driven T2* modulation
#   detectable above the thermal noise floor of a practical
#   single-segment solenoid receiver?
#
# If yes: Phase 1 multi-segment array is viable.
# If no: declared parameters need revision before hardware build.
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass


# ---- Physical constants [declared] ------------------------------------

B0_TESLA              = 1.5          # Static field strength [T]
GAMMA_MHZ_PER_T       = 42.577       # Proton gyromagnetic ratio [MHz/T]
LARMOR_FREQ_MHZ       = GAMMA_MHZ_PER_T * B0_TESLA   # ~63.87 MHz
GAMMA_RAD_PER_S_PER_T = GAMMA_MHZ_PER_T * 2.0 * np.pi * 1e6

# Boltzmann constant [J/K]
K_BOLTZMANN           = 1.380649e-23

# Temperature [K] — room temperature phantom
TEMPERATURE_K         = 310.0        # 37°C — body temperature


# ---- Phantom geometry [declared] --------------------------------------
#
# Cylindrical tube of deoxygenated saline solution.
# Deoxygenated saline approximates deoxygenated blood susceptibility.
# Pulsatile pump varies oxygenation state across the cardiac cycle,
# modulating delta_chi and therefore T2*.
#
# Tube inner diameter: 8mm — representative of a medium-sized artery
# (e.g. middle cerebral artery diameter range 2-5mm, but using 8mm
# for bench SNR optimization; Phase 1 will test smaller diameters).

PHANTOM_DIAMETER_MM   = 8.0          # Tube inner diameter [mm]
PHANTOM_RADIUS_MM     = PHANTOM_DIAMETER_MM / 2.0
PHANTOM_LENGTH_MM     = 50.0         # Active length within coil [mm]

# Phantom volume [mm^3] = pi * r^2 * L
PHANTOM_VOLUME_MM3    = np.pi * PHANTOM_RADIUS_MM**2 * PHANTOM_LENGTH_MM
PHANTOM_VOLUME_M3     = PHANTOM_VOLUME_MM3 * 1e-9   # convert to m^3

# Proton number density in water [protons/m^3]
# Pure water: ~6.7e28 protons/m^3
PROTON_DENSITY_M3     = 6.7e28


# ---- Solenoid coil geometry [declared] --------------------------------
#
# A single solenoid segment wound directly around the phantom tube.
# This maximizes filling factor — the fraction of coil field volume
# occupied by the sample. Filling factor approaches 1.0 for a
# solenoid wound tightly around its sample.
#
# This is the bench NMR geometry. It is also Phase 0 of the
# segmented cylindrical solenoid — one segment at tube scale,
# validating the physics before scaling to head geometry.

COIL_INNER_DIAMETER_MM = PHANTOM_DIAMETER_MM + 1.0  # 1mm clearance [mm]
COIL_LENGTH_MM         = PHANTOM_LENGTH_MM           # Coil covers full phantom
COIL_TURNS            = 20           # Number of turns
COIL_WIRE_DIAMETER_MM  = 0.5         # Wire diameter [mm]

# Coil resistance [Ohm] — copper wire
# R = rho_copper * L_wire / A_wire
# L_wire = n_turns * pi * coil_diameter
# A_wire = pi * (wire_diameter/2)^2
RHO_COPPER            = 1.68e-8      # Copper resistivity [Ohm*m]
COIL_WIRE_LENGTH_M    = (COIL_TURNS * np.pi *
                          COIL_INNER_DIAMETER_MM * 1e-3)
COIL_WIRE_AREA_M2     = np.pi * (COIL_WIRE_DIAMETER_MM * 1e-3 / 2)**2
COIL_RESISTANCE_OHM   = RHO_COPPER * COIL_WIRE_LENGTH_M / COIL_WIRE_AREA_M2

# Filling factor [dimensionless]
# For a solenoid wound around its sample:
# eta = V_sample / V_coil
V_COIL_M3             = (np.pi * (COIL_INNER_DIAMETER_MM * 1e-3 / 2)**2
                          * COIL_LENGTH_MM * 1e-3)
FILLING_FACTOR        = PHANTOM_VOLUME_M3 / V_COIL_M3


# ---- Preamplifier [declared] -----------------------------------------
#
# Low-noise preamplifier immediately at the coil.
# Noise figure: declared from commercially available LNA specifications
# (e.g. Mini-Circuits ZFL-500LN+: NF ~ 2.9 dB at 64 MHz).

PREAMP_NOISE_FIGURE_DB = 2.9         # [dB]
PREAMP_NOISE_FACTOR    = 10**(PREAMP_NOISE_FIGURE_DB / 10.0)


# ---- Acquisition parameters [declared] --------------------------------

TR_S                  = 0.800        # Repetition time [s] — ~75 bpm
TE_S                  = 0.030        # Echo time [s] — 30ms at 1.5T
FLIP_ANGLE_DEG        = 90.0         # Spin echo uses 90° excitation
FLIP_ANGLE_RAD        = np.deg2rad(FLIP_ANGLE_DEG)

# Receiver bandwidth [Hz]
# Determined by T2* — capture full decay with margin
# BW = 1 / (2 * T2*_min) approximately
RECEIVER_BW_HZ        = 10000.0      # 10 kHz bandwidth

# Number of cardiac evolution steps
N_CARDIAC_STEPS       = 1200         # Same as fMRI simulation


# ---- Tissue/phantom NMR parameters [declared] ------------------------
#
# Deoxygenated saline approximating blood T1, T2*, proton density.
# Values from published MRI physics for blood at 1.5T.

T1_MS                 = 1200.0       # Blood T1 at 1.5T [ms]
T2STAR_BASELINE_MS    = 50.0         # T2* of deoxygenated blood at 1.5T [ms]
PROTON_DENSITY_REL    = 0.85         # Relative proton density (blood vs water)


# ---- Cardiac parameters [declared] -----------------------------------

HEART_RATE_BPM        = 60.0
T_CARDIAC_S           = 60.0 / HEART_RATE_BPM   # 1.0s

# Phase advance per TR
CARDIAC_PHASE_ADVANCE_RAD = 2.0 * np.pi * (TR_S / T_CARDIAC_S)

# Pulsatile oxygenation modulation
# Cardiac systole delivers oxygenated blood, reducing deoxy-Hb fraction.
# Diastole allows re-deoxygenation. Peak-to-peak oxygenation variation
# declared as 10% of total hemoglobin (conservative estimate).
A_OXYGENATION         = 0.10         # Fractional oxygenation modulation


# ---- Susceptibility parameters [declared] ----------------------------

DELTA_CHI_DEOXY_PPM   = 0.264        # Deoxygenated vs oxygenated Hb [ppm]
DELTA_CHI_DEOXY       = DELTA_CHI_DEOXY_PPM * 1e-6

# Geometry factor for cylindrical vessel in B0 field
# For vessel perpendicular to B0: geometry_factor = 1/2
# For vessel parallel to B0: geometry_factor = 0
# Declared: perpendicular (worst case, maximum susceptibility effect)
GEOMETRY_FACTOR       = 1.0 / 2.0

# Fractional blood volume in phantom
# For a tube phantom: essentially 1.0 (pure blood substitute)
BLOOD_VOLUME_FRACTION = 0.95         # 95% blood substitute, 5% vessel wall


# ---- DeclaredDomain ---------------------------------------------------

@dataclass
class DeclaredDomain:
    """
    Fully declared physical domain for the Phase 0 NMR simulation.

    Attributes
    ----------
    n_cardiac_steps : int
        Number of cardiac evolution steps (1200).
    cardiac_phase_advance_rad : float
        Phase advance per TR [rad].
    coil_resistance_ohm : float
        Solenoid coil resistance [Ohm].
    filling_factor : float
        Coil filling factor (sample volume / coil volume).
    preamp_noise_factor : float
        Preamplifier noise factor (linear).
    receiver_bw_hz : float
        Receiver bandwidth [Hz].
    temperature_k : float
        System temperature [K].
    """
    n_cardiac_steps:          int
    cardiac_phase_advance_rad: float
    coil_resistance_ohm:      float
    filling_factor:           float
    preamp_noise_factor:      float
    receiver_bw_hz:           float
    temperature_k:            float


def declare_domain() -> DeclaredDomain:
    """
    Declare and return the Phase 0 simulation domain.
    Prints full declaration report for Origin review.
    """
    domain = DeclaredDomain(
        n_cardiac_steps=N_CARDIAC_STEPS,
        cardiac_phase_advance_rad=float(CARDIAC_PHASE_ADVANCE_RAD),
        coil_resistance_ohm=float(COIL_RESISTANCE_OHM),
        filling_factor=float(FILLING_FACTOR),
        preamp_noise_factor=float(PREAMP_NOISE_FACTOR),
        receiver_bw_hz=float(RECEIVER_BW_HZ),
        temperature_k=float(TEMPERATURE_K),
    )
    _print_declaration(domain)
    return domain


def _print_declaration(domain: DeclaredDomain) -> None:
    print("=" * 60)
    print("DOMAIN DECLARATION — abr-nmr-phase0")
    print("Phase 0: Single-segment solenoid, pulsatile flow phantom")
    print("=" * 60)

    print("\n--- Static Field ---")
    print(f"  B0 = {B0_TESLA}T  "
          f"Larmor = {LARMOR_FREQ_MHZ:.2f} MHz")

    print("\n--- Phantom Geometry ---")
    print(f"  Tube diameter:  {PHANTOM_DIAMETER_MM}mm")
    print(f"  Tube length:    {PHANTOM_LENGTH_MM}mm")
    print(f"  Volume:         {PHANTOM_VOLUME_MM3:.1f} mm³  "
          f"({PHANTOM_VOLUME_M3*1e6:.4f} mL)")

    print("\n--- Solenoid Coil ---")
    print(f"  Inner diameter: {COIL_INNER_DIAMETER_MM}mm")
    print(f"  Length:         {COIL_LENGTH_MM}mm")
    print(f"  Turns:          {COIL_TURNS}")
    print(f"  Wire diameter:  {COIL_WIRE_DIAMETER_MM}mm")
    print(f"  Resistance:     {domain.coil_resistance_ohm*1000:.3f} mOhm")
    print(f"  Filling factor: {domain.filling_factor:.4f}")

    print("\n--- Preamplifier ---")
    print(f"  Noise figure:   {PREAMP_NOISE_FIGURE_DB} dB")
    print(f"  Noise factor:   {domain.preamp_noise_factor:.4f}")

    print("\n--- Acquisition ---")
    print(f"  TR={TR_S*1000:.0f}ms  TE={TE_S*1000:.0f}ms  "
          f"flip={FLIP_ANGLE_DEG}°  BW={RECEIVER_BW_HZ:.0f}Hz")
    print(f"  Cardiac steps:  {domain.n_cardiac_steps}")
    print(f"  Phase advance/TR: "
          f"{np.degrees(domain.cardiac_phase_advance_rad):.1f}°")

    print("\n--- Phantom NMR Parameters ---")
    print(f"  T1={T1_MS}ms  T2*_baseline={T2STAR_BASELINE_MS}ms")
    print(f"  Proton density (rel): {PROTON_DENSITY_REL}")
    print(f"  Blood volume fraction: {BLOOD_VOLUME_FRACTION}")

    print("\n--- Susceptibility ---")
    print(f"  Δχ_deoxy = {DELTA_CHI_DEOXY_PPM} ppm")
    print(f"  Geometry factor = {GEOMETRY_FACTOR}")
    print(f"  Oxygenation modulation A = {A_OXYGENATION}")

    print("\n--- Thermal Noise ---")
    print(f"  Temperature: {domain.temperature_k}K")
    print(f"  Johnson-Nyquist noise power: "
          f"{K_BOLTZMANN * domain.temperature_k * domain.receiver_bw_hz:.4e} W")

    print("\nBounded over D. No claim beyond D.")
    print("=" * 60)
