# sim/declaration_0c.py — abr-nmr-phase0c
# Metatron Dynamics, Inc.
#
# Domain declaration for Phase 0c: 32-element array, 4-tissue phantom.
#
# Scaled from Phase 0b (8 elements, 45° spacing) to:
#   32 elements, 11.25° spacing
#
# All tissue types, compartment radii, and NMR parameters are unchanged
# from Phase 0b. Only the array geometry changes.
#
# Declared element count rationale (from Origin):
#   32 elements preferred over 64.
#   At 64 elements, per-element sensitivity volume halves relative to 32.
#   Signal voltage drops proportionally; noise reduction from shorter
#   wire is insufficient to compensate. Net SNR per element is lower.
#   For tissue differentiation by water content contrast, the boundary
#   detection ratio is set by tissue contrast, not element count.
#   32 elements demonstrates the principle at higher per-element SNR.
#
# Phantom cross-section (8cm diameter, axial view):
#   Same as Phase 0b — 4 tissue compartments, same radial structure.
#   Tumor declared in right hemisphere (elements 0–15).
#   Gray matter declared in left hemisphere (elements 16–31).
#
# Adjacency: clockwise directed ring.
#   Edge e = i → (i+1) % 32 for i in 0..31  (clockwise)
#   Edge e+32 = (i+1)%32 → i for i in 0..31 (counterclockwise)
#   Total: 64 directed edges.
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ---- Physical constants [declared, unchanged from Phase 0b] -----------

B0_TESLA              = 1.5
GAMMA_MHZ_PER_T       = 42.577
LARMOR_FREQ_MHZ       = GAMMA_MHZ_PER_T * B0_TESLA
GAMMA_RAD_PER_S_PER_T = GAMMA_MHZ_PER_T * 2.0 * np.pi * 1e6
K_BOLTZMANN           = 1.380649e-23
TEMPERATURE_K         = 310.0
HBAR                  = 1.054571817e-34
MU_0                  = 4.0 * np.pi * 1e-7
PROTON_DENSITY_WATER  = 6.7e28   # protons/m^3 in pure water


# ---- Acquisition parameters [declared, unchanged] ---------------------

TR_S                  = 0.800
TE_S                  = 0.030
FLIP_ANGLE_RAD        = np.radians(90.0)
N_CARDIAC_STEPS       = 1200
CARDIAC_PHASE_ADVANCE_RAD = 2.0 * np.pi * (TR_S / 1.0)   # 60 bpm
RECEIVER_BW_HZ        = 10000.0
PREAMP_NOISE_FIGURE_DB = 2.9
PREAMP_NOISE_FACTOR   = 10 ** (PREAMP_NOISE_FIGURE_DB / 10.0)


# ---- Phantom geometry [declared, unchanged] ---------------------------

PHANTOM_DIAMETER_MM   = 80.0
PHANTOM_RADIUS_MM     = PHANTOM_DIAMETER_MM / 2.0
PHANTOM_LENGTH_MM     = 50.0


# ---- Declared tissue compartments [unchanged from Phase 0b] ----------

TISSUE_TYPES = {
    'CSF': {
        'water_fraction': 0.99,
        'T1_ms':          4000.0,
        'T2star_ms':      500.0,
        'vascular':       False,
        'description':    'Cerebrospinal fluid — outer rim',
    },
    'white_matter': {
        'water_fraction': 0.72,
        'T1_ms':          700.0,
        'T2star_ms':      45.0,
        'vascular':       True,
        'vascular_fraction': 0.02,
        'description':    'White matter — inner ring',
    },
    'gray_matter': {
        'water_fraction': 0.84,
        'T1_ms':          1100.0,
        'T2star_ms':      55.0,
        'vascular':       True,
        'vascular_fraction': 0.04,
        'description':    'Gray matter — inner core, left hemisphere',
    },
    'tumor': {
        'water_fraction': 0.92,
        'T1_ms':          1400.0,
        'T2star_ms':      80.0,
        'vascular':       True,
        'vascular_fraction': 0.08,
        'description':    'Glioma — focal right hemisphere',
    },
}

COMPARTMENT_RADII = {
    'CSF':          (35.0, 40.0),
    'white_matter': (15.0, 35.0),
    'gray_matter':  (0.0,  15.0),
    'tumor':        (0.0,  15.0),
}


# ---- 32-element array geometry [declared] ----------------------------

N_ELEMENTS        = 32
ELEMENT_ANGLE_DEG = 360.0 / N_ELEMENTS    # 11.25° per element

# Right hemisphere: elements 0–15 (0° to 180°) — tumor declared here
# Left hemisphere:  elements 16–31 (180° to 360°) — gray matter declared here
TUMOR_SECTORS    = set(range(0, 16))
GRAY_MAT_SECTORS = set(range(16, 32))

# Element dimensions
ELEMENT_ARC_MM    = np.radians(ELEMENT_ANGLE_DEG) * (PHANTOM_RADIUS_MM + 5.0)
ELEMENT_LENGTH_MM = PHANTOM_LENGTH_MM
ELEMENT_WIDTH_MM  = ELEMENT_ARC_MM
ELEMENT_TURNS     = 5

# Wire geometry [unchanged]
WIRE_DIAMETER_MM  = 0.5
RHO_COPPER        = 1.68e-8

ELEMENT_SENSITIVITY_DEPTH_MM  = 10.0
ELEMENT_SENSITIVITY_FRACTION  = 0.6


# ---- DeclaredDomain ---------------------------------------------------

@dataclass
class ElementGeometry:
    element_id:       int
    angle_center_deg: float
    angle_start_deg:  float
    angle_end_deg:    float
    sectors_covered:  List[int]
    tissue_mix:       dict


@dataclass
class DeclaredDomain0c:
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


def declare_domain() -> 'DeclaredDomain0c':
    elements     = _build_elements()
    adjacency, reverse_edge = _build_adjacency()

    domain = DeclaredDomain0c(
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
    )

    _print_declaration(domain)
    return domain


def _build_elements() -> List[ElementGeometry]:
    elements = []
    r_csf_in,  r_csf_out  = COMPARTMENT_RADII['CSF']
    r_wm_in,   r_wm_out   = COMPARTMENT_RADII['white_matter']
    r_core_in, r_core_out = COMPARTMENT_RADII['gray_matter']

    total_area = np.pi * r_csf_out ** 2 / N_ELEMENTS
    csf_area   = np.pi * (r_csf_out ** 2 - r_csf_in ** 2) / N_ELEMENTS
    wm_area    = np.pi * (r_wm_out  ** 2 - r_wm_in  ** 2) / N_ELEMENTS
    core_area  = np.pi * r_core_out ** 2 / N_ELEMENTS

    for i in range(N_ELEMENTS):
        angle_center = i * ELEMENT_ANGLE_DEG
        angle_start  = angle_center - ELEMENT_ANGLE_DEG / 2.0
        angle_end    = angle_center + ELEMENT_ANGLE_DEG / 2.0

        tissue_mix = {
            'CSF':         csf_area  / total_area,
            'white_matter': wm_area  / total_area,
        }
        if i in TUMOR_SECTORS:
            tissue_mix['tumor']       = core_area / total_area
            tissue_mix['gray_matter'] = 0.0
        else:
            tissue_mix['gray_matter'] = core_area / total_area
            tissue_mix['tumor']       = 0.0

        elements.append(ElementGeometry(
            element_id=i,
            angle_center_deg=angle_center,
            angle_start_deg=angle_start,
            angle_end_deg=angle_end,
            sectors_covered=[i],
            tissue_mix=tissue_mix,
        ))
    return elements


def _build_adjacency() -> Tuple[List[Tuple[int, int]], List[int]]:
    N = N_ELEMENTS
    adjacency = []
    for i in range(N):
        adjacency.append((i, (i + 1) % N))        # clockwise edges 0..N-1
    for i in range(N):
        adjacency.append(((i + 1) % N, i))        # counterclockwise edges N..2N-1
    reverse_edge = list(range(N, 2 * N)) + list(range(N))
    return adjacency, reverse_edge


def _print_declaration(domain: 'DeclaredDomain0c') -> None:
    N = domain.n_elements
    print("=" * 65)
    print("DOMAIN DECLARATION — abr-nmr-phase0c")
    print(f"Phase 0c: Multi-tissue phantom, {N}-element solenoid array")
    print("=" * 65)

    print(f"\n--- Phantom ---")
    print(f"  Diameter: {PHANTOM_DIAMETER_MM}mm  Length: {PHANTOM_LENGTH_MM}mm")
    print(f"  Tissue compartments: {len(TISSUE_TYPES)}")

    print(f"\n--- Tissue Types ---")
    print(f"  {'Name':>16}  {'Water%':>7}  {'T1 ms':>7}  "
          f"{'T2* ms':>7}  {'Vascular':>9}")
    for name, t in TISSUE_TYPES.items():
        print(f"  {name:>16}  {t['water_fraction']*100:>6.1f}%  "
              f"{t['T1_ms']:>7.0f}  {t['T2star_ms']:>7.1f}  "
              f"{'Yes' if t['vascular'] else 'No':>9}")

    print(f"\n--- Coil Array ---")
    print(f"  Elements: {N}  Spacing: {ELEMENT_ANGLE_DEG:.3f}°  "
          f"Topology: clockwise directed ring")
    print(f"  Directed edges: {len(domain.adjacency)} "
          f"({N} clockwise + {N} counterclockwise)")
    print(f"  Right hemisphere (tumor):      elements 0–15")
    print(f"  Left hemisphere (gray matter): elements 16–31")

    print(f"\n--- First and last 4 adjacency edges (clockwise) ---")
    for e in list(range(4)) + list(range(N-2, N)):
        src, tgt = domain.adjacency[e]
        rev = domain.reverse_edge[e]
        rsrc, rtgt = domain.adjacency[rev]
        print(f"  Edge {e:>2}: {src:>2}→{tgt:>2}  "
              f"(reverse: edge {rev:>2} = {rsrc:>2}→{rtgt:>2})")
    print(f"  ... ({N - 6} edges omitted) ...")

    print(f"\n--- Element tissue mix (sample: boundary region) ---")
    print(f"  {'Elem':>4}  {'Angle':>8}  {'Hemisphere':>16}  "
          f"{'CSF%':>5}  {'WM%':>5}  {'GM%':>5}  {'Tumor%':>7}")
    # Print elements near the declared boundary (elements 13–18)
    for el in domain.elements[13:19]:
        hemi = 'Right (tumor)' if el.element_id in TUMOR_SECTORS else 'Left (GM)'
        print(f"  {el.element_id:>4}  {el.angle_center_deg:>6.2f}°  "
              f"{hemi:>16}  "
              f"{el.tissue_mix['CSF']*100:>5.1f}  "
              f"{el.tissue_mix['white_matter']*100:>5.1f}  "
              f"{el.tissue_mix.get('gray_matter', 0)*100:>5.1f}  "
              f"{el.tissue_mix.get('tumor', 0)*100:>7.1f}")

    print(f"\n  Cardiac steps: {domain.n_cardiac_steps}")
    print(f"  Phase advance/TR: "
          f"{np.degrees(domain.cardiac_phase_advance_rad):.1f}°")
    print("\nBounded over D. No claim beyond D.")
    print("=" * 65)
