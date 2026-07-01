# sim/declaration_0e.py — abr-nmr-phase0e
# Metatron Dynamics, Inc.
#
# Domain declaration for Phase 0e: B0 inhomogeneity viability sweep.
#
# Phase 0d established that SNR and operator separation survive to below
# 0.05T under perfect B0 homogeneity. The binding practical constraint
# at low field is B0 inhomogeneity — permanent magnets are not perfectly
# homogeneous, and field variation across the sample degrades effective
# T2*, reducing signal and eventually destroying operator separation.
#
# Phase 0e declares inhomogeneity as a sweep parameter and finds:
#   (1) The inhomogeneity level at which operator separation fails
#       at each declared reference B0 (0.2T, 0.3T, 0.5T)
#   (2) Whether commercial permanent magnet systems fall within the
#       survival region
#
# Physical model:
#   B0 inhomogeneity adds an additional R2* contribution per element.
#   Each element's sensitivity volume spans a declared arc of the phantom.
#   The field variation across that arc is:
#       delta_B0_abs = inhomogeneity_frac * B0
#   This contributes to effective T2* as:
#       1/T2*_eff(tissue) = 1/T2*_tissue + gamma * delta_B0_abs
#
#   This is the same R2* shortening mechanism as susceptibility-driven
#   modulation in signal_0d.py — same formula, different source.
#   The inhomogeneity contribution is B0-dependent (scales with B0);
#   the tissue T2* contribution is held at declared 1.5T values.
#
# Declared sweep:
#   INHOMOGENEITY_MIN = 0.0      (perfect homogeneity — Phase 0d baseline)
#   INHOMOGENEITY_MAX = 0.01     (1% = 10,000 ppm — far beyond any magnet)
#
#   Two-zone sweep (piecewise linear, not uniform):
#     Zone 1: 0 – 200 ppm, 40 steps (~5 ppm resolution)
#             Fine resolution where the survival threshold is expected to live.
#     Zone 2: 200 – 10,000 ppm, 19 steps (coarse)
#             Confirms total signal destruction above the threshold.
#   Total: 59 steps. Zone boundary at 200 ppm, not duplicated.
#
#   Reference B0 values: 0.2T, 0.3T, 0.5T
#   (Spanning the declared permanent magnet range from Phase 0d)
#
# Declared permanent magnet reference specs (from published literature):
#   Halbach array (research grade):   10–100 ppm homogeneity over DSV
#   Single-sided permanent magnet:    100–1000 ppm over sensitive volume
#   Low-field MRI systems (e.g. 0.5T Hyperfine): ~1000 ppm corrected
#
#   1 ppm = 1e-6 fractional inhomogeneity
#   100 ppm = 1e-4 fractional inhomogeneity
#   1000 ppm = 1e-3 fractional inhomogeneity
#
# The survival threshold is reported in ppm for direct comparison
# against magnet procurement specifications.
#
# Declared open condition:
#   The model applies the declared fractional inhomogeneity as a uniform
#   field deviation across each element's entire sensitivity volume —
#   equivalent to treating the maximum field variation as if it were
#   present everywhere within that volume simultaneously. This is a
#   worst-case approximation: real B0 inhomogeneity has spatial structure,
#   and the signal-weighted average deviation across the sensitivity volume
#   is smaller than the peak deviation. Shimming reduces the peak further.
#   Because this implementation applies the maximum deviation uniformly,
#   it overstates the R2* contribution from inhomogeneity relative to
#   what a real element would experience. The survival threshold reported
#   is therefore a lower bound — real hardware will survive to equal or
#   higher inhomogeneity levels than the simulation predicts.
#
#   Comparison caveat: whether a commercial magnet system falls within
#   the declared survival region depends on whether its published ppm
#   specification is measured over the same sensitive volume geometry,
#   under comparable conditions, and with comparable shimming assumptions.
#   Those comparison assumptions must be declared explicitly in Phase 1.
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from sim.declaration_0c import (
    GAMMA_RAD_PER_S_PER_T,
    K_BOLTZMANN,
    TEMPERATURE_K,
    HBAR,
    MU_0,
    PROTON_DENSITY_WATER,
    TR_S,
    TE_S,
    FLIP_ANGLE_RAD,
    N_CARDIAC_STEPS,
    CARDIAC_PHASE_ADVANCE_RAD,
    RECEIVER_BW_HZ,
    PREAMP_NOISE_FACTOR,
    PHANTOM_DIAMETER_MM,
    PHANTOM_RADIUS_MM,
    PHANTOM_LENGTH_MM,
    TISSUE_TYPES,
    N_ELEMENTS,
    ELEMENT_ANGLE_DEG,
    ELEMENT_TURNS,
    WIRE_DIAMETER_MM,
    RHO_COPPER,
    ELEMENT_SENSITIVITY_DEPTH_MM,
    ELEMENT_SENSITIVITY_FRACTION,
    TUMOR_SECTORS,
    GRAY_MAT_SECTORS,
    ElementGeometry,
    _build_elements,
    _build_adjacency,
)
from sim.declaration_0d import C_PROJECTION_NAME, SNR_THRESHOLD, BOUNDARY_RATIO_THRESHOLD

# ---- Sweep parameters [declared by Origin] ---------------------------

INHOMOGENEITY_MIN = 0.0       # Fractional (0 = perfect homogeneity)
INHOMOGENEITY_MAX = 0.01      # Fractional (1% = 10,000 ppm)

# Two-zone sweep: fine resolution at low ppm where the threshold lives,
# coarse at high ppm to confirm total signal destruction.
# Zone 1: 0 – 200 ppm (40 steps, 5 ppm resolution)
# Zone 2: 200 – 10000 ppm (20 steps, coarse)
_ZONE1 = np.linspace(0.0,       200e-6,  40)
_ZONE2 = np.linspace(200e-6,    0.01,    20)[1:]   # exclude overlap at 200 ppm
INHOMOGENEITY_SWEEP = np.concatenate([_ZONE1, _ZONE2])
N_INHOMOGENEITY     = len(INHOMOGENEITY_SWEEP)

# Reference B0 values for the sweep [T]
REFERENCE_B0_VALUES = [0.2, 0.3, 0.5]

# ---- Permanent magnet reference specs [declared, ppm] ---------------
# Source: published literature on low-field NMR/MRI permanent magnets.
# These are declared reference ranges for interpretation only.
# No claim is made about specific hardware products.

PM_SPECS_PPM = {
    'Halbach_research':    (10,   100),    # Research-grade Halbach array
    'Single_sided':        (100,  1000),   # Single-sided permanent magnet
    'Low_field_MRI':       (500,  2000),   # Low-field clinical systems (shimmed)
}

# Conversion: 1 ppm = 1e-6 fractional
PPM_TO_FRAC = 1e-6


@dataclass
class DeclaredDomain0e:
    """
    Declared sweep domain for Phase 0e inhomogeneity analysis.

    Attributes
    ----------
    inhomogeneity_sweep : float64 [N_INHOMOGENEITY]
        Declared fractional B0 inhomogeneity values. Piecewise linear
        (two-zone): fine resolution (Zone 1, 0–200 ppm) concatenated
        with coarse resolution (Zone 2, 200–10,000 ppm).
        delta_B0_abs = inhomogeneity * B0 [T]

    reference_b0_values : list[float]
        B0 values at which the sweep is run [T].

    n_elements, n_cardiac_steps, cardiac_phase_advance_rad,
    elements, adjacency, reverse_edge, tissue_types,
    receiver_bw_hz, preamp_noise_factor, temperature_k :
        Unchanged from Phase 0c/0d.

    snr_threshold : float
    boundary_ratio_threshold : float
    projection_name : str
    """
    inhomogeneity_sweep:       np.ndarray
    reference_b0_values:       List[float]
    n_elements:                int
    n_cardiac_steps:           int
    cardiac_phase_advance_rad: float
    elements:                  List[ElementGeometry]
    adjacency:                 List[Tuple[int, int]]
    reverse_edge:              List[int]
    tissue_types:              dict
    receiver_bw_hz:            float
    preamp_noise_factor:       float
    temperature_k:             float
    snr_threshold:             float
    boundary_ratio_threshold:  float
    projection_name:           str


def declare_domain() -> DeclaredDomain0e:
    elements                = _build_elements()
    adjacency, reverse_edge = _build_adjacency()

    domain = DeclaredDomain0e(
        inhomogeneity_sweep=INHOMOGENEITY_SWEEP,
        reference_b0_values=REFERENCE_B0_VALUES,
        n_elements=N_ELEMENTS,
        n_cardiac_steps=N_CARDIAC_STEPS,
        cardiac_phase_advance_rad=float(CARDIAC_PHASE_ADVANCE_RAD),
        elements=elements,
        adjacency=adjacency,
        reverse_edge=reverse_edge,
        tissue_types=TISSUE_TYPES,
        receiver_bw_hz=RECEIVER_BW_HZ,
        preamp_noise_factor=PREAMP_NOISE_FACTOR,
        temperature_k=TEMPERATURE_K,
        snr_threshold=SNR_THRESHOLD,
        boundary_ratio_threshold=BOUNDARY_RATIO_THRESHOLD,
        projection_name=C_PROJECTION_NAME,
    )

    _print_declaration(domain)
    return domain


def _print_declaration(domain: DeclaredDomain0e) -> None:
    print("=" * 65)
    print("DOMAIN DECLARATION — abr-nmr-phase0e")
    print("Phase 0e: B0 inhomogeneity viability sweep")
    print("=" * 65)

    print(f"\n--- Inhomogeneity Sweep ---")
    print(f"  Range:  {INHOMOGENEITY_MIN} → {INHOMOGENEITY_MAX} (fractional)")
    print(f"  Range:  {INHOMOGENEITY_MIN*1e6:.0f} → "
          f"{INHOMOGENEITY_MAX*1e6:.0f} ppm")
    print(f"  Steps:  {N_INHOMOGENEITY} (piecewise linear — two zones)")
    print(f"  Zone 1: 0 – 200 ppm, 40 steps (~5 ppm resolution)")
    print(f"  Zone 2: 200 – 10,000 ppm, 19 steps (coarse)")

    print(f"\n--- Reference B0 Values ---")
    for b0 in domain.reference_b0_values:
        print(f"  {b0}T")

    print(f"\n--- Physical Model ---")
    print(f"  1/T2*_eff = 1/T2*_tissue + gamma * (inhomogeneity * B0)")
    print(f"  Inhomogeneity applied as uniform maximum deviation per element.")
    print(f"  Worst-case: real signal-weighted average deviation is smaller.")
    print(f"  Survival threshold reported is a lower bound on real hardware.")

    print(f"\n--- Permanent Magnet Reference Specs ---")
    print(f"  {'System':>24}  {'Low [ppm]':>10}  {'High [ppm]':>10}  "
          f"{'Frac low':>12}  {'Frac high':>12}")
    for name, (lo, hi) in PM_SPECS_PPM.items():
        print(f"  {name:>24}  {lo:>10}  {hi:>10}  "
              f"{lo*PPM_TO_FRAC:>12.2e}  {hi*PPM_TO_FRAC:>12.2e}")

    print(f"\n--- Survival Thresholds (inherited from Phase 0d) ---")
    print(f"  SNR threshold:            {domain.snr_threshold}")
    print(f"  Boundary ratio threshold: {domain.boundary_ratio_threshold}")
    print(f"  C projection:             {domain.projection_name}")

    print(f"\n--- Declared Open Condition ---")
    print(f"  Uniform maximum deviation applied per element (worst-case).")
    print(f"  Real signal-weighted average deviation is smaller.")
    print(f"  Real shimming reduces peak deviation further.")
    print(f"  Reported threshold is a lower bound on real hardware survival.")
    print(f"  PM comparison requires matching sensitive volume, conditions,")
    print(f"  and shimming assumptions — declared explicitly in Phase 1.")

    print(f"\nBounded over D. No claim beyond D.")
    print("=" * 65)
