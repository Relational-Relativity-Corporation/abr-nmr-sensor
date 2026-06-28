# sim/operators_0b.py — abr-nmr-phase0b
# Metatron Dynamics, Inc.
#
# ABRCE operators for the 8-element segmented solenoid array.
#
# Domain: 8 coil elements arranged in a ring.
# Topology: clockwise directed ring (8 forward + 8 reverse edges).
# Field: S[n_elements, n_cardiac_steps]
#
# The operators act on the per-element signal exactly as they act
# on the spin field lattice in the fMRI simulation — same kernel,
# different declared domain. This is the declared invariance of the
# ABRCE framework: the operators are domain-independent.
#
# E[e, t] = R(B(A(S)), rho(A(S)))[e, t]
#
# The declared test:
#   Edges crossing the tumor boundary (edges between tumor-sector
#   elements and gray-matter-sector elements) should show elevated
#   E field magnitude relative to edges within homogeneous tissue.
#
#   Declared boundary edges:
#     Edge 3 → 4  (element 3: tumor sector → element 4: GM sector)
#     Edge 7 → 0  (element 7: GM sector → element 0: tumor sector)
#     (and their reverses)
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass
from sim.declaration_0b import (
    DeclaredDomain0b,
    N_ELEMENTS,
    TUMOR_SECTORS,
    GRAY_MAT_SECTORS,
)
from sim.signal_0b import SignalField0b


@dataclass
class EField0b:
    """
    ABRCE operator outputs for the 8-element array.

    Attributes
    ----------
    A_field : float64 [n_edges, n_steps]
        Operator A: directed signal difference per edge per step.

    rho : float64 [n_elements]
        Per-element circulation strength from full timeseries.

    B_field : float64 [n_edges, n_steps]
        Operator B: accumulated gradient per edge per step.

    E_field : float64 [n_edges, n_steps]
        Kernel output: relational circulation field.

    boundary_edges : list of int
        Edge indices crossing the declared tumor boundary.

    interior_edges_tumor : list of int
        Edge indices within the tumor sector.

    interior_edges_gm : list of int
        Edge indices within the gray matter sector.

    boundary_E_mean : float64 [n_steps]
        Mean |E| at boundary edges per step.

    interior_tumor_E_mean : float64 [n_steps]
        Mean |E| at tumor interior edges per step.

    interior_gm_E_mean : float64 [n_steps]
        Mean |E| at GM interior edges per step.
    """
    A_field:               np.ndarray
    rho:                   np.ndarray
    B_field:               np.ndarray
    E_field:               np.ndarray
    boundary_edges:        list
    interior_edges_tumor:  list
    interior_edges_gm:     list
    boundary_E_mean:       np.ndarray
    interior_tumor_E_mean: np.ndarray
    interior_gm_E_mean:    np.ndarray


def run_operators(
    domain: DeclaredDomain0b,
    signal: SignalField0b,
    rho_base: float = 0.3,
) -> EField0b:
    """
    Apply ABRCE operators to the 8-element array signal.

    Parameters
    ----------
    domain : DeclaredDomain0b
    signal : SignalField0b
    rho_base : float
        Declared open parameter.

    Returns
    -------
    EField0b
    """
    n_edges  = len(domain.adjacency)
    n_steps  = domain.n_cardiac_steps
    S        = signal.S   # [n_elements, n_steps]

    src = np.array([e[0] for e in domain.adjacency], dtype=np.int32)
    tgt = np.array([e[1] for e in domain.adjacency], dtype=np.int32)
    rev = np.array(domain.reverse_edge, dtype=np.int32)

    # ---- Operator A --------------------------------------------------
    # A[e, t] = S[src(e), t] - S[tgt(e), t]
    A_field = (S[src, :] - S[tgt, :]).astype(np.float64)

    # ---- Operator ρ --------------------------------------------------
    # rho[v] = rho_base * m[v] / (1 + m[v])
    # m[v] = max |A[e]| over all edges incident to v, all steps
    edge_max = np.abs(A_field).max(axis=1)   # [n_edges]
    m = np.zeros(N_ELEMENTS, dtype=np.float64)
    np.maximum.at(m, src, edge_max)
    np.maximum.at(m, tgt, edge_max)
    rho = (rho_base * m / (1.0 + m)).astype(np.float64)

    # ---- Operator B --------------------------------------------------
    # B(g)[e] = g[e] + Σ_{f ∈ succ(e), f ≠ rev(e)} g[f]
    # For a ring topology with 2 directed edges per position:
    # succ(e=(i→j)) = edges leaving j, excluding reverse edge (j→i)
    # On a ring, each node has exactly 2 outgoing edges (CW and CCW).
    # The reverse of CW edge i→i+1 is CCW edge i+1→i.
    # succ(CW edge i) = {CW edge i+1} (excluding the CCW reverse)
    # succ(CCW edge i) = {CCW edge i-1} (excluding the CW reverse)

    B_field = A_field.copy()
    for e in range(n_edges):
        j   = int(tgt[e])
        rev_e = int(rev[e])
        # Add all edges leaving j, except the reverse of e
        for f in range(n_edges):
            if int(src[f]) == j and f != rev_e:
                B_field[e, :] += A_field[f, :]

    # ---- Operator R --------------------------------------------------
    # R(g)[e] = g[e] + rho[src(e)] * (Σ_{succ(e)} g - Σ_{pred(e)} g)
    # pred(e=(i→j)) = edges entering i, excluding reverse edge
    E_field = B_field.copy()
    for e in range(n_edges):
        i     = int(src[e])
        j     = int(tgt[e])
        rev_e = int(rev[e])

        # Forward sum: edges leaving j, excluding reverse
        fwd = 0.0
        for f in range(n_edges):
            if int(src[f]) == j and f != rev_e:
                fwd += B_field[f, :]

        # Backward sum: edges entering i, excluding reverse
        bwd = 0.0
        for f in range(n_edges):
            if int(tgt[f]) == i and f != rev_e:
                bwd += B_field[f, :]

        E_field[e, :] += rho[i] * (fwd - bwd)

    # ---- Classify edges ----------------------------------------------
    boundary_edges       = []
    interior_edges_tumor = []
    interior_edges_gm    = []

    for e in range(n_edges):
        s = int(src[e])
        t = int(tgt[e])
        s_tumor = s in TUMOR_SECTORS
        t_tumor = t in TUMOR_SECTORS

        if s_tumor != t_tumor:
            boundary_edges.append(e)
        elif s_tumor and t_tumor:
            interior_edges_tumor.append(e)
        else:
            interior_edges_gm.append(e)

    # ---- Per-step mean |E| per edge class ----------------------------
    def mean_abs_E(edge_list):
        if not edge_list:
            return np.zeros(n_steps)
        return np.abs(E_field[edge_list, :]).mean(axis=0)

    boundary_E_mean       = mean_abs_E(boundary_edges)
    interior_tumor_E_mean = mean_abs_E(interior_edges_tumor)
    interior_gm_E_mean    = mean_abs_E(interior_edges_gm)

    result = EField0b(
        A_field=A_field,
        rho=rho,
        B_field=B_field,
        E_field=E_field,
        boundary_edges=boundary_edges,
        interior_edges_tumor=interior_edges_tumor,
        interior_edges_gm=interior_edges_gm,
        boundary_E_mean=boundary_E_mean,
        interior_tumor_E_mean=interior_tumor_E_mean,
        interior_gm_E_mean=interior_gm_E_mean,
    )

    _print_operator_report(result, domain, signal)
    return result


def _print_operator_report(
    result: EField0b,
    domain: DeclaredDomain0b,
    signal: SignalField0b,
) -> None:
    print("\n--- ABRCE Operator Report (Phase 0b) ---")

    print(f"\n  ρ per element:")
    for i in range(N_ELEMENTS):
        hemi = 'R(tumor)' if i in TUMOR_SECTORS else 'L(GM)'
        print(f"    Element {i} [{hemi}]: rho={result.rho[i]:.6f}")

    print(f"\n  Edge classification:")
    print(f"    Boundary edges (tumor↔GM): {result.boundary_edges}")
    print(f"    Tumor interior edges:      {result.interior_edges_tumor}")
    print(f"    GM interior edges:         {result.interior_edges_gm}")

    print(f"\n  E field range: "
          f"{result.E_field.min():.4e} – {result.E_field.max():.4e}")

    # At t=0
    t0 = 0
    print(f"\n  |E| at t=0 by edge class:")
    if result.boundary_edges:
        bnd_t0 = float(np.abs(result.E_field[result.boundary_edges, t0]).mean())
        print(f"    Boundary edges:       {bnd_t0:.6e}")
    if result.interior_edges_tumor:
        tum_t0 = float(np.abs(result.E_field[result.interior_edges_tumor, t0]).mean())
        print(f"    Tumor interior edges: {tum_t0:.6e}")
    if result.interior_edges_gm:
        gm_t0  = float(np.abs(result.E_field[result.interior_edges_gm, t0]).mean())
        print(f"    GM interior edges:    {gm_t0:.6e}")

    # Mean over all steps
    bnd_mean  = float(result.boundary_E_mean.mean())
    tum_mean  = float(result.interior_tumor_E_mean.mean()) if result.interior_edges_tumor else 0
    gm_mean   = float(result.interior_gm_E_mean.mean()) if result.interior_edges_gm else 0

    print(f"\n  Mean |E| over all steps:")
    print(f"    Boundary:      {bnd_mean:.6e}")
    print(f"    Tumor interior:{tum_mean:.6e}")
    print(f"    GM interior:   {gm_mean:.6e}")

    print(f"\n  --- BOUNDARY DETECTION VERDICT ---")
    if bnd_mean > tum_mean and bnd_mean > gm_mean:
        ratio_t = bnd_mean / tum_mean if tum_mean > 0 else float('inf')
        ratio_g = bnd_mean / gm_mean  if gm_mean  > 0 else float('inf')
        print(f"  BOUNDARY DETECTED.")
        print(f"  Boundary E / tumor interior: {ratio_t:.2f}×")
        print(f"  Boundary E / GM interior:    {ratio_g:.2f}×")
        print(f"  The operators find elevated relational contrast")
        print(f"  at the declared tumor boundary.")
    else:
        print(f"  BOUNDARY NOT DETECTED at declared parameters.")
        print(f"  Boundary E does not exceed interior E.")
        print(f"  Declare revised parameters before hardware build.")
