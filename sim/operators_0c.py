# sim/operators_0c.py — abr-nmr-phase0c
# Metatron Dynamics, Inc.
#
# ABRCE operators for the 32-element segmented solenoid array.
#
# Same kernel as Phase 0b (operators_0b.py), scaled to 32 elements.
# The element loop in Phase 0b was O(n_edges^2) — acceptable at 8
# elements (64 edges), inadmissible at 32 elements (1024 pairs).
# This file replaces the Python edge loops with vectorized NumPy
# operations. The declared operator definitions are unchanged.
#
# Declared boundary:
#   Elements 0–15:  right hemisphere (tumor sector)
#   Elements 16–31: left hemisphere (gray matter sector)
#   Boundary edges: those crossing element 15 → 16 or 31 → 0
#     Clockwise:        edge 15 (15→16)  and edge 31 (31→0)
#     Counterclockwise: edge 47 (16→15)  and edge 63 (0→31)
#
# The declared test:
#   Mean |E| at boundary edges > mean |E| at interior edges.
#   Ratio reported as boundary / tumor-interior and boundary / GM-interior.
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass
from sim.declaration_0c import (
    DeclaredDomain0c,
    N_ELEMENTS,
    TUMOR_SECTORS,
    GRAY_MAT_SECTORS,
)
from sim.signal_0c import SignalField0c


@dataclass
class EField0c:
    """
    ABRCE operator outputs for the 32-element array.

    Attributes
    ----------
    A_field : float64 [n_edges, n_steps]
    rho     : float64 [n_elements]
    B_field : float64 [n_edges, n_steps]
    E_field : float64 [n_edges, n_steps]
    boundary_edges        : list[int]
    interior_edges_tumor  : list[int]
    interior_edges_gm     : list[int]
    boundary_E_mean       : float64 [n_steps]
    interior_tumor_E_mean : float64 [n_steps]
    interior_gm_E_mean    : float64 [n_steps]
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
    domain: DeclaredDomain0c,
    signal: SignalField0c,
    rho_base: float = 0.3,
) -> EField0c:
    """
    Apply ABRCE operators to the 32-element array signal.
    Vectorized over edges and steps.
    """
    N       = N_ELEMENTS
    n_edges = len(domain.adjacency)      # 64
    n_steps = domain.n_cardiac_steps
    S       = signal.S                   # [N, n_steps]

    src = np.array([e[0] for e in domain.adjacency], dtype=np.int32)
    tgt = np.array([e[1] for e in domain.adjacency], dtype=np.int32)
    rev = np.array(domain.reverse_edge,              dtype=np.int32)

    # ---- Operator A --------------------------------------------------
    # A[e, t] = S[src(e), t] - S[tgt(e), t]
    A_field = (S[src, :] - S[tgt, :]).astype(np.float64)   # [n_edges, n_steps]

    # ---- Operator ρ --------------------------------------------------
    # rho[v] = rho_base * m[v] / (1 + m[v])
    # m[v] = max |A[e]| over all edges incident to v, all steps
    edge_max = np.abs(A_field).max(axis=1)   # [n_edges]
    m = np.zeros(N, dtype=np.float64)
    np.maximum.at(m, src, edge_max)
    np.maximum.at(m, tgt, edge_max)
    rho = (rho_base * m / (1.0 + m)).astype(np.float64)    # [N]

    # ---- Operator B (vectorized) -------------------------------------
    # B(g)[e] = g[e] + Σ_{f ∈ succ(e), f ≠ rev(e)} g[f]
    #
    # For the declared clockwise ring with paired reverse edges:
    #   succ(e) = all edges leaving tgt(e), excluding rev(e)
    #
    # Build succ_map: succ_map[e] = list of successor edge indices
    # On this ring topology each node has exactly 2 outgoing edges
    # (one CW, one CCW). succ(e) excludes the reverse — leaves 1 edge.
    # Vectorized: for each edge e, the single successor is determined.

    # For clockwise edge e (0..N-1): e = i→i+1
    #   tgt = i+1, leaving edges from i+1: CW=(i+1) and CCW=(i+1+N)
    #   rev(e) = e+N (the CCW edge i+1→i)
    #   succ = the OTHER outgoing edge from i+1 = CW edge (i+1)%N
    # For CCW edge e (N..2N-1): e = i+1→i  (where e-N = CW edge i)
    #   tgt = i, leaving edges from i: CW=i and CCW=i+N
    #   rev(e) = e-N (the CW edge i→i+1)
    #   succ = the OTHER outgoing edge from i = CCW edge i+N

    # Build successor index array: succ_idx[e] = single successor edge
    succ_idx = np.empty(n_edges, dtype=np.int32)
    for e in range(n_edges):
        j     = int(tgt[e])
        rev_e = int(rev[e])
        # Find edges leaving j, excluding rev_e
        candidates = [f for f in range(n_edges)
                      if int(src[f]) == j and f != rev_e]
        # On this topology there is exactly one successor
        assert len(candidates) == 1, (
            f"Edge {e}: expected 1 successor, got {candidates}"
        )
        succ_idx[e] = candidates[0]

    # B[e] = A[e] + A[succ(e)]
    B_field = A_field + A_field[succ_idx, :]    # [n_edges, n_steps]

    # ---- Operator R (vectorized) -------------------------------------
    # R(g)[e] = g[e] + rho[src(e)] * (Σ_{succ(e)} g - Σ_{pred(e)} g)
    #
    # pred(e) = edges entering src(e), excluding rev(e)
    # By the same ring-topology argument, each node has exactly 2
    # incoming edges; pred(e) excludes rev(e) → 1 predecessor edge.

    pred_idx = np.empty(n_edges, dtype=np.int32)
    for e in range(n_edges):
        i     = int(src[e])
        rev_e = int(rev[e])
        candidates = [f for f in range(n_edges)
                      if int(tgt[f]) == i and f != rev_e]
        assert len(candidates) == 1, (
            f"Edge {e}: expected 1 predecessor, got {candidates}"
        )
        pred_idx[e] = candidates[0]

    rho_e   = rho[src]                          # [n_edges]
    fwd     = B_field[succ_idx, :]              # [n_edges, n_steps]
    bwd     = B_field[pred_idx, :]              # [n_edges, n_steps]
    E_field = (B_field
               + rho_e[:, None] * (fwd - bwd)) # [n_edges, n_steps]

    # ---- Classify edges ----------------------------------------------
    boundary_edges       = []
    interior_edges_tumor = []
    interior_edges_gm    = []

    for e in range(n_edges):
        s       = int(src[e])
        t       = int(tgt[e])
        s_tumor = s in TUMOR_SECTORS
        t_tumor = t in TUMOR_SECTORS
        if s_tumor != t_tumor:
            boundary_edges.append(e)
        elif s_tumor:
            interior_edges_tumor.append(e)
        else:
            interior_edges_gm.append(e)

    # ---- Per-step mean |E| per edge class ----------------------------
    def mean_abs_E(edge_list: list) -> np.ndarray:
        if not edge_list:
            return np.zeros(n_steps)
        return np.abs(E_field[edge_list, :]).mean(axis=0)

    boundary_E_mean       = mean_abs_E(boundary_edges)
    interior_tumor_E_mean = mean_abs_E(interior_edges_tumor)
    interior_gm_E_mean    = mean_abs_E(interior_edges_gm)

    result = EField0c(
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

    _print_operator_report(result)
    return result


def _print_operator_report(result: EField0c) -> None:
    print("\n--- ABRCE Operator Report (Phase 0c, 32-element) ---")

    # ρ summary (not printing all 32 — print hemisphere means)
    rho_tumor = result.rho[list(TUMOR_SECTORS)]
    rho_gm    = result.rho[list(GRAY_MAT_SECTORS)]
    print(f"\n  ρ summary:")
    print(f"    Tumor hemisphere (elements  0–15): "
          f"mean={rho_tumor.mean():.6f}  "
          f"min={rho_tumor.min():.6f}  max={rho_tumor.max():.6f}")
    print(f"    GM hemisphere   (elements 16–31): "
          f"mean={rho_gm.mean():.6f}  "
          f"min={rho_gm.min():.6f}  max={rho_gm.max():.6f}")

    print(f"\n  Edge classification:")
    print(f"    Boundary edges (tumor↔GM): {result.boundary_edges}")
    print(f"    Tumor interior edges:      "
          f"{len(result.interior_edges_tumor)} edges")
    print(f"    GM interior edges:         "
          f"{len(result.interior_edges_gm)} edges")

    print(f"\n  E field range: "
          f"{result.E_field.min():.4e} – {result.E_field.max():.4e}")

    # At t=0
    t0 = 0
    print(f"\n  |E| at t=0 by edge class:")
    if result.boundary_edges:
        bnd_t0 = float(np.abs(result.E_field[result.boundary_edges, t0]).mean())
        print(f"    Boundary edges:       {bnd_t0:.6e}")
    if result.interior_edges_tumor:
        tum_t0 = float(
            np.abs(result.E_field[result.interior_edges_tumor, t0]).mean()
        )
        print(f"    Tumor interior edges: {tum_t0:.6e}")
    if result.interior_edges_gm:
        gm_t0 = float(
            np.abs(result.E_field[result.interior_edges_gm, t0]).mean()
        )
        print(f"    GM interior edges:    {gm_t0:.6e}")

    # Mean over all steps
    bnd_mean = float(result.boundary_E_mean.mean())
    tum_mean = (float(result.interior_tumor_E_mean.mean())
                if result.interior_edges_tumor else 0.0)
    gm_mean  = (float(result.interior_gm_E_mean.mean())
                if result.interior_edges_gm else 0.0)

    print(f"\n  Mean |E| over all steps:")
    print(f"    Boundary:       {bnd_mean:.6e}")
    print(f"    Tumor interior: {tum_mean:.6e}")
    print(f"    GM interior:    {gm_mean:.6e}")

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
