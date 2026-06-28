# sim/declaration_0b.py — abr-nmr-phase0b
# Metatron Dynamics, Inc.
#
# Domain declaration for Phase 0b: multi-tissue phantom simulation.
#
# Declares:
#   - A cylindrical phantom with 4 declared tissue compartments
#   - An 8-element segmented solenoid array surrounding the phantom
#   - Per-element sensitivity volumes and declared adjacency topology
#   - NMR parameters (PD, T1, T2*) for each tissue type
#   - Thermal noise model per element
#   - Cardiac modulation per tissue (vascular tissues only)
#
# Phantom cross-section (8cm diameter cylinder, axial view):
#
#         CSF rim (99% water)
#       ___________________
#      /                   \
#     |   White matter      |
#     |   (72% water)       |
#     |    ___________      |
#     |   /           \     |
#     |  | Gray matter |    |
#     |  | (84% water) |    |
#     |  |   _______   |    |
#     |  |  / Tumor  \ |    |
#     |  | | (92% H2O)| |   |
#     |  |  \_______/ |    |
#     |  |            |    |
#     |   \___________/    |
#     |                    |
#      \___________________/
#
# 8 coil elements spaced 45° apart around the cylinder.
# Each element covers a 45° arc segment — one sector.
# Elements are numbered 0–7 clockwise from the top.
# Declared adjacency: each element adjacent to its two neighbors.
# Directed: clockwise direction declared for edge orientation.
#
# The tumor is declared in the right hemisphere (elements 1-3).
# The operators should find elevated relational contrast at the
# declared tumor boundary — between the tumor sector elements
# and the surrounding gray/white matter sector elements.
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ---- Physical constants [declared] ------------------------------------

B0_TESLA              = 1.5
GAMMA_MHZ_PER_T       = 42.577
LARMOR_FREQ_MHZ       = GAMMA_MHZ_PER_T * B0_TESLA
GAMMA_RAD_PER_S_PER_T = GAMMA_MHZ_PER_T * 2.0 * np.pi * 1e6
K_BOLTZMANN           = 1.380649e-23
TEMPERATURE_K         = 310.0
HBAR                  = 1.054571817e-34
MU_0                  = 4.0 * np.pi * 1e-7
PROTON_DENSITY_WATER  = 6.7e28   # protons/m^3 in pure water


# ---- Acquisition parameters [declared] --------------------------------

TR_S                  = 0.800    # [s]
TE_S                  = 0.030    # [s]
FLIP_ANGLE_RAD        = np.radians(90.0)
N_CARDIAC_STEPS       = 1200
CARDIAC_PHASE_ADVANCE_RAD = 2.0 * np.pi * (TR_S / 1.0)  # 60 bpm
RECEIVER_BW_HZ        = 10000.0
PREAMP_NOISE_FIGURE_DB = 2.9
PREAMP_NOISE_FACTOR   = 10**(PREAMP_NOISE_FIGURE_DB / 10.0)


# ---- Phantom geometry [declared] --------------------------------------

PHANTOM_DIAMETER_MM   = 80.0     # 8cm — head-scale phantom
PHANTOM_RADIUS_MM     = PHANTOM_DIAMETER_MM / 2.0
PHANTOM_LENGTH_MM     = 50.0     # Axial coverage [mm]


# ---- Declared tissue compartments ------------------------------------
#
# Each compartment is declared with:
#   name, water_fraction, T1_ms, T2star_ms, vascular (bool),
#   inner_radius_mm, outer_radius_mm, angular_extent (deg)
#
# Radii are from phantom center.
# Angular extent: which sectors of the phantom contain this tissue.
# 'all' means full 360° ring. Tuple means arc from angle_start to angle_end.
#
# NMR parameters from published values at 1.5T.

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
        'vascular_fraction': 0.02,   # 2% blood volume
        'description':    'White matter — inner ring',
    },
    'gray_matter': {
        'water_fraction': 0.84,
        'T1_ms':          1100.0,
        'T2star_ms':      55.0,
        'vascular':       True,
        'vascular_fraction': 0.04,   # 4% blood volume
        'description':    'Gray matter — inner core',
    },
    'tumor': {
        'water_fraction': 0.92,
        'T1_ms':          1400.0,
        'T2star_ms':      80.0,
        'vascular':       True,
        'vascular_fraction': 0.08,   # Elevated — neovascularization
        'description':    'Glioma — focal right hemisphere',
    },
}

# Phantom radial structure [mm from center]
# CSF:          35–40mm (outer 5mm rim)
# White matter: 15–35mm (ring)
# Gray matter:  0–15mm  (core, left hemisphere sectors 4-7)
# Tumor:        0–15mm  (core, right hemisphere sectors 0-3)

COMPARTMENT_RADII = {
    'CSF':          (35.0, 40.0),    # inner, outer radius [mm]
    'white_matter': (15.0, 35.0),
    'gray_matter':  (0.0,  15.0),    # sectors 4-7 (left)
    'tumor':        (0.0,  15.0),    # sectors 0-3 (right)
}

# Angular sectors containing each tissue core type
# Sectors 0-3: right hemisphere (tumor declared here)
# Sectors 4-7: left hemisphere (gray matter declared here)
TUMOR_SECTORS    = {0, 1, 2, 3}
GRAY_MAT_SECTORS = {4, 5, 6, 7}


# ---- Coil array geometry [declared] ----------------------------------

N_ELEMENTS        = 8            # 8-element segmented solenoid
ELEMENT_ANGLE_DEG = 360.0 / N_ELEMENTS   # 45° per element

# Element dimensions
ELEMENT_ARC_MM    = np.radians(ELEMENT_ANGLE_DEG) * (PHANTOM_RADIUS_MM + 5.0)
ELEMENT_LENGTH_MM = PHANTOM_LENGTH_MM
ELEMENT_WIDTH_MM  = ELEMENT_ARC_MM
ELEMENT_TURNS     = 5

# Wire geometry
WIRE_DIAMETER_MM  = 0.5
RHO_COPPER        = 1.68e-8

# Each element's sensitivity volume: the wedge-shaped tissue sector
# within ~10mm of the element arc (approximately element arc length / 2)
ELEMENT_SENSITIVITY_DEPTH_MM = 10.0

# Filling factor per element
# Each element sees its wedge sector of the phantom
# Approximated as element arc area / element coil area
ELEMENT_SENSITIVITY_FRACTION = 0.6   # Declared approximation

# Declared adjacency: clockwise ring
# Element i is adjacent to element (i+1) % N_ELEMENTS
# Direction: clockwise (i → i+1)
# This encodes the declared signal propagation direction around the array


# ---- DeclaredDomain ---------------------------------------------------

@dataclass
class ElementGeometry:
    """Declared geometry for one coil element."""
    element_id:      int
    angle_center_deg: float       # Center angle [degrees]
    angle_start_deg: float        # Arc start [degrees]
    angle_end_deg:   float        # Arc end [degrees]
    sectors_covered: List[int]    # Which 45° sectors this element covers
    tissue_mix:      dict         # Fraction of each tissue type in sensitivity vol


@dataclass
class DeclaredDomain0b:
    """
    Fully declared domain for Phase 0b multi-tissue simulation.
    """
    n_elements:               int
    n_cardiac_steps:          int
    cardiac_phase_advance_rad: float
    elements:                 List[ElementGeometry]
    adjacency:                List[Tuple[int,int]]   # directed edges (src, tgt)
    reverse_edge:             List[int]              # reverse_edge[e] index
    tissue_types:             dict
    receiver_bw_hz:           float
    preamp_noise_factor:      float
    temperature_k:            float


def declare_domain() -> DeclaredDomain0b:
    """
    Declare the Phase 0b multi-tissue domain.
    Build element geometry and adjacency topology.
    """
    elements = _build_elements()
    adjacency, reverse_edge = _build_adjacency()

    domain = DeclaredDomain0b(
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
    """
    Declare 8 elements evenly spaced around the phantom circumference.
    Each element's sensitivity volume contains a declared tissue mix
    based on the compartment radii and angular extent.
    """
    elements = []
    for i in range(N_ELEMENTS):
        angle_center = i * ELEMENT_ANGLE_DEG
        angle_start  = angle_center - ELEMENT_ANGLE_DEG / 2
        angle_end    = angle_center + ELEMENT_ANGLE_DEG / 2

        # This element covers sector i
        sectors = [i]

        # Declare tissue mix in sensitivity volume
        # The element sees through the CSF rim into white matter
        # and into the core (gray matter or tumor depending on sector)
        tissue_mix = {}

        # CSF fraction: outer rim always present
        # Volume fraction = rim area / total sensitivity area
        r_csf_in, r_csf_out = COMPARTMENT_RADII['CSF']
        r_wm_in,  r_wm_out  = COMPARTMENT_RADII['white_matter']
        r_core_in, r_core_out = COMPARTMENT_RADII['gray_matter']

        # Approximate as radial fractions (2D cross-section)
        total_area = np.pi * r_csf_out**2 / N_ELEMENTS
        csf_area   = np.pi * (r_csf_out**2 - r_csf_in**2) / N_ELEMENTS
        wm_area    = np.pi * (r_wm_out**2  - r_wm_in**2)  / N_ELEMENTS
        core_area  = np.pi * r_core_out**2 / N_ELEMENTS

        tissue_mix['CSF']         = csf_area  / total_area
        tissue_mix['white_matter'] = wm_area  / total_area

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
            sectors_covered=sectors,
            tissue_mix=tissue_mix,
        ))

    return elements


def _build_adjacency() -> Tuple[List[Tuple[int,int]], List[int]]:
    """
    Declare clockwise ring adjacency.
    Edge e = (i → (i+1) % N) for i in 0..N-1.
    Reverse edge of e_i is e_{(i + N/2) % N} ... actually
    the reverse of (i → i+1) is (i+1 → i) which is
    edge going counterclockwise = edge N + i in our layout.

    Edges 0..N-1: clockwise  (i → i+1 mod N)
    Edges N..2N-1: counterclockwise (i+1 → i mod N)
    reverse_edge[e] = e + N  (and vice versa)
    """
    N = N_ELEMENTS
    adjacency = []

    # Clockwise edges: 0..N-1
    for i in range(N):
        adjacency.append((i, (i + 1) % N))

    # Counterclockwise edges: N..2N-1
    for i in range(N):
        adjacency.append(((i + 1) % N, i))

    # reverse_edge: clockwise edge i ↔ counterclockwise edge i+N
    reverse_edge = list(range(N, 2*N)) + list(range(N))

    return adjacency, reverse_edge


def _print_declaration(domain: DeclaredDomain0b) -> None:
    print("=" * 65)
    print("DOMAIN DECLARATION — abr-nmr-phase0b")
    print("Phase 0b: Multi-tissue phantom, 8-element solenoid array")
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
    print(f"  Elements: {domain.n_elements}  "
          f"Spacing: {ELEMENT_ANGLE_DEG}°  "
          f"Topology: clockwise ring")
    print(f"  Directed edges: {len(domain.adjacency)} "
          f"({N_ELEMENTS} clockwise + {N_ELEMENTS} counterclockwise)")

    print(f"\n--- Element Tissue Mix ---")
    print(f"  {'Elem':>4}  {'Angle':>7}  {'Hemisphere':>12}  "
          f"{'CSF%':>5}  {'WM%':>5}  {'GM%':>5}  {'Tumor%':>7}")
    for el in domain.elements:
        hemi = 'Right (tumor)' if el.element_id in TUMOR_SECTORS else 'Left (GM)'
        print(f"  {el.element_id:>4}  {el.angle_center_deg:>5.0f}°  "
              f"{hemi:>12}  "
              f"{el.tissue_mix['CSF']*100:>5.1f}  "
              f"{el.tissue_mix['white_matter']*100:>5.1f}  "
              f"{el.tissue_mix.get('gray_matter',0)*100:>5.1f}  "
              f"{el.tissue_mix.get('tumor',0)*100:>7.1f}")

    print(f"\n--- Adjacency ---")
    for e, (src, tgt) in enumerate(domain.adjacency[:N_ELEMENTS]):
        rev = domain.reverse_edge[e]
        print(f"  Edge {e}: {src}→{tgt}  "
              f"(reverse: edge {rev} = "
              f"{domain.adjacency[rev][0]}→{domain.adjacency[rev][1]})")

    print(f"\n  Cardiac steps: {domain.n_cardiac_steps}")
    print(f"  Phase advance/TR: "
          f"{np.degrees(domain.cardiac_phase_advance_rad):.1f}°")
    print("\nBounded over D. No claim beyond D.")
    print("=" * 65)
