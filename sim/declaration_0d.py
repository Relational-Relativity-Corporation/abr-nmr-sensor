# sim/declaration_0d.py — abr-nmr-phase0d
# Metatron Dynamics, Inc.
#
# Domain declaration for Phase 0d: low-field B0 viability sweep.
#
# Extends Phase 0c (32-element array, 4-tissue phantom) by making B0
# a declared sweep parameter rather than a fixed constant.
#
# Purpose:
#   Determine the minimum B0 at which:
#     (1) All 32 elements remain detectable above the noise floor
#         (SNR threshold: SNR >= SNR_THRESHOLD per element)
#     (2) Operator separation survives: boundary E ratio > 1.0
#
#   These are two distinct thresholds. They may not coincide.
#   Both are reported. The simulation finds them empirically from
#   declared first-principles formulas — no fitting, no proxies.
#
# Declared sweep:
#   B0_MIN_T = 0.05T   (below permanent magnet territory)
#   B0_MAX_T = 1.5T    (Phase 0c reference)
#   N_B0     = 50      (log-spaced)
#
# B0-dependent quantities (recomputed per sweep step):
#   omega_0    ∝ B0             (Larmor frequency)
#   M_0        ∝ B0             (equilibrium magnetization)
#   V_signal   ∝ omega_0 * M_0  ∝ B0²
#   V_noise    — independent of B0 (Johnson-Nyquist, coil resistance)
#   SNR        ∝ B0²
#
# B0-independent quantities (held constant — declared conservative):
#   T1, T2*, water_fraction     (held at 1.5T published values)
#   Coil geometry, resistance   (unchanged)
#   Tissue contrast             (water content difference — intrinsic)
#
# Declared open condition:
#   T1 shortens and T2* lengthens at lower B0. Both effects improve
#   SNR margin relative to this simulation. This sweep is therefore
#   conservative — the real survival threshold is at or below what
#   the simulation reports.
#
#   B0 field homogeneity is assumed perfect. Real permanent magnets
#   have inhomogeneity that degrades effective T2* and broadens
#   linewidth. This simulation overstates performance at low B0.
#   The real survival threshold for a permanent magnet system is
#   at or above what the simulation reports.
#
#   The two effects partially cancel. The net direction of the
#   conservatism must be validated in Phase 1 hardware.
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

# ---- Import all unchanged parameters from Phase 0c ------------------
from sim.declaration_0c import (
    GAMMA_MHZ_PER_T,
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
    COMPARTMENT_RADII,
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
    DeclaredDomain0c,
    _build_elements,
    _build_adjacency,
)

# ---- Sweep parameters [declared by Origin] ---------------------------

B0_MIN_T   = 0.05     # Lower bound [T] — below permanent magnet territory
B0_MAX_T   = 1.5      # Upper bound [T] — Phase 0c reference
N_B0       = 50       # Number of sweep steps (log-spaced)

# Survival thresholds [declared]
SNR_THRESHOLD          = 5.0    # Minimum SNR per element per step
BOUNDARY_RATIO_THRESHOLD = 1.0  # Minimum boundary E / interior E

# ---- C projection declaration [required before threshold use] --------
#
# The survival criterion "boundary E / interior E" is a declared
# projection (C layer). It must be stated here before use.
#
# Declared projection: C_mean_abs_ratio
#
#   C_mean_abs_ratio(E_field, edge_class_A, edge_class_B) =
#       mean( |E_field[e, :]| for e in edge_class_A )
#     / mean( |E_field[e, :]| for e in edge_class_B )
#
#   where mean is taken over all edges in the class AND all cardiac steps.
#
# Preserves:
#   - Average relational contrast magnitude across all boundary edges
#     and all cardiac steps
#   - Directional symmetry (absolute value removes sign before mean)
#
# Discards:
#   - Per-edge variation within each class (collapsed to class mean)
#   - Per-step variation across cardiac steps (collapsed to step mean)
#   - Sign of E_field at each edge-step
#   - Distribution shape within each edge class
#
# Denominator:
#   max( C_mean_abs_ratio(E, interior_tumor_edges),
#        C_mean_abs_ratio(E, interior_gm_edges) )
#   The larger interior mean is used as denominator. Conservative:
#   detection is harder when interior contrast is already elevated.
#
# Admissibility:
#   - Edge classes are declared in domain topology, not inferred from E
#   - Absolute value and mean are declared operations with known effect
#   - Preserved and discarded quantities are stated above
#
# Inadmissible alternatives (not declared, therefore not used):
#   - max(|E|) ratio: sensitive to single-edge outliers, not declared
#   - Frobenius norm ratio: equivalent to RMS, different from mean|E|
#   - Per-step ratio: requires a declared per-step decision rule
#
# C_PROJECTION_NAME is the canonical string identifier for this
# projection, used in operator output, test assertions, and reports.

C_PROJECTION_NAME = "C_mean_abs_ratio"

# Declared B0 sweep array
B0_SWEEP = np.logspace(
    np.log10(B0_MIN_T),
    np.log10(B0_MAX_T),
    N_B0,
)

# ---- Permanent magnet reference range [declared] --------------------
# Halbach array systems: typically 0.2T–0.5T
# Single-sided permanent magnet: 0.1T–0.3T
# These are declared reference ranges for interpretation only.
# The simulation makes no claim about specific hardware designs.
PM_RANGE_LOW_T  = 0.2
PM_RANGE_HIGH_T = 0.5


@dataclass
class DeclaredDomain0d:
    """
    Declared sweep domain for Phase 0d low-field viability analysis.

    Inherits all geometry from Phase 0c. Adds B0 as a sweep parameter.

    Attributes
    ----------
    b0_sweep : float64 [N_B0]
        Declared B0 values for the sweep [T]. Log-spaced.

    n_elements : int
    n_cardiac_steps : int
    cardiac_phase_advance_rad : float
    elements : list[ElementGeometry]
    adjacency : list[tuple[int,int]]
    reverse_edge : list[int]
    tissue_types : dict
    receiver_bw_hz : float
    preamp_noise_factor : float
    temperature_k : float

    snr_threshold : float
        Declared minimum SNR per element. Below this: not detectable.

    boundary_ratio_threshold : float
        Declared minimum boundary E ratio. Below this: separation lost.
    """
    b0_sweep:                  np.ndarray
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


def declare_domain() -> DeclaredDomain0d:
    elements                = _build_elements()
    adjacency, reverse_edge = _build_adjacency()

    domain = DeclaredDomain0d(
        b0_sweep=B0_SWEEP,
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
    )

    _print_declaration(domain)
    return domain


def _print_declaration(domain: DeclaredDomain0d) -> None:
    print("=" * 65)
    print("DOMAIN DECLARATION — abr-nmr-phase0d")
    print("Phase 0d: Low-field B0 viability sweep, 32-element array")
    print("=" * 65)

    print(f"\n--- Sweep Parameters ---")
    print(f"  B0 range:    {B0_MIN_T}T  →  {B0_MAX_T}T  (log-spaced)")
    print(f"  Steps:       {N_B0}")
    print(f"  B0[0]:       {domain.b0_sweep[0]:.4f}T")
    print(f"  B0[-1]:      {domain.b0_sweep[-1]:.4f}T")
    print(f"  B0[25]:      {domain.b0_sweep[25]:.4f}T  (midpoint)")

    print(f"\n--- Survival Thresholds ---")
    print(f"  SNR threshold:           {domain.snr_threshold:.1f}  "
          f"(per element, per cardiac step)")
    print(f"  Boundary ratio threshold: {domain.boundary_ratio_threshold:.1f}  "
          f"(boundary E / interior E)")

    print(f"\n--- Geometry (unchanged from Phase 0c) ---")
    print(f"  Phantom:     {PHANTOM_DIAMETER_MM}mm diameter, "
          f"{PHANTOM_LENGTH_MM}mm length")
    print(f"  Array:       {N_ELEMENTS} elements, "
          f"{ELEMENT_ANGLE_DEG:.3f}° spacing, clockwise ring")
    print(f"  Tissue:      CSF / white matter / gray matter / tumor")
    print(f"  Boundary:    elements 15→16 and 31→0 "
          f"(tumor↔GM declared crossing)")

    print(f"\n--- B0-Dependent Scaling (declared) ---")
    print(f"  omega_0  ∝ B0        (Larmor angular frequency)")
    print(f"  M_0      ∝ B0        (equilibrium magnetization)")
    print(f"  V_signal ∝ B0²       (induced EMF)")
    print(f"  V_noise  — constant  (Johnson-Nyquist, coil only)")
    print(f"  SNR      ∝ B0²")

    print(f"\n--- Declared Conservative Assumptions ---")
    print(f"  T1, T2* held at 1.5T published values (conservative)")
    print(f"  Perfect B0 homogeneity (optimistic at low field)")
    print(f"  Net conservatism direction: to be determined by Phase 1")

    print(f"\n--- Permanent Magnet Reference Range ---")
    print(f"  Halbach / single-sided: {PM_RANGE_LOW_T}T – {PM_RANGE_HIGH_T}T")
    print(f"  (declared for interpretation; no hardware claim)")

    print(f"\nBounded over D. No claim beyond D.")
    print("=" * 65)
