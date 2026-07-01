# sim/operators_0e.py — abr-nmr-phase0e
# Metatron Dynamics, Inc.
#
# ABRCE operators for Phase 0e inhomogeneity sweep.
#
# Kernel identical to Phase 0d (operators_0d.py).
# Accepts SignalResult0e; returns OperatorResult0e with
# inhomogeneity_frac and inhomogeneity_ppm carried through.
#
# C projection: C_mean_abs_ratio — declared in declaration_0d.py,
# inherited here. Mean |E| over edge class and all cardiac steps.
# Preserves average magnitude. Discards per-edge/step variation and sign.
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass

from sim.declaration_0e import (
    DeclaredDomain0e,
    N_ELEMENTS,
    TUMOR_SECTORS,
    GRAY_MAT_SECTORS,
)
from sim.signal_0e import SignalResult0e

RHO_BASE_DEFAULT = 0.3


@dataclass
class OperatorResult0e:
    """
    Operator output for one (B0, inhomogeneity) sweep step.

    Attributes
    ----------
    b0 : float
    inhomogeneity_frac : float
    inhomogeneity_ppm : float
    projection_name : str
    boundary_ratio : float
        C_mean_abs_ratio: mean|E|(boundary) / max(mean|E|(tumor), mean|E|(gm))
    boundary_E_mean : float
    interior_tumor_E_mean : float
    interior_gm_E_mean : float
    separation_survived : bool
    rho_mean : float
    """
    b0:                    float
    inhomogeneity_frac:    float
    inhomogeneity_ppm:     float
    projection_name:       str
    boundary_ratio:        float
    boundary_E_mean:       float
    interior_tumor_E_mean: float
    interior_gm_E_mean:    float
    separation_survived:   bool
    rho_mean:              float


def run_operators(
    domain: DeclaredDomain0e,
    signal: SignalResult0e,
    rho_base: float = RHO_BASE_DEFAULT,
) -> OperatorResult0e:

    n_edges = len(domain.adjacency)
    S       = signal.S_field

    src = np.array([e[0] for e in domain.adjacency], dtype=np.int32)
    tgt = np.array([e[1] for e in domain.adjacency], dtype=np.int32)
    rev = np.array(domain.reverse_edge, dtype=np.int32)

    # ---- Operator A --------------------------------------------------
    A_field = (S[src, :] - S[tgt, :]).astype(np.float64)

    # ---- Operator rho ------------------------------------------------
    edge_max = np.abs(A_field).max(axis=1)
    m        = np.zeros(N_ELEMENTS, dtype=np.float64)
    np.maximum.at(m, src, edge_max)
    np.maximum.at(m, tgt, edge_max)
    rho = (rho_base * m / (1.0 + m)).astype(np.float64)

    # ---- Operator B --------------------------------------------------
    succ_idx = _build_succ(src, tgt, rev, n_edges)
    B_field  = A_field + A_field[succ_idx, :]

    # ---- Operator R --------------------------------------------------
    pred_idx = _build_pred(src, tgt, rev, n_edges)
    rho_e    = rho[src]
    E_field  = B_field + rho_e[:, None] * (B_field[succ_idx, :] - B_field[pred_idx, :])

    # ---- Classify edges ----------------------------------------------
    boundary_edges, interior_edges_tumor, interior_edges_gm = \
        _classify_edges(src, tgt, n_edges)

    # ---- C_mean_abs_ratio projection ---------------------------------
    def C_mean_abs_ratio(edge_list):
        if not edge_list:
            return 0.0
        return float(np.abs(E_field[edge_list, :]).mean())

    boundary_E_mean       = C_mean_abs_ratio(boundary_edges)
    interior_tumor_E_mean = C_mean_abs_ratio(interior_edges_tumor)
    interior_gm_E_mean    = C_mean_abs_ratio(interior_edges_gm)

    max_interior = max(interior_tumor_E_mean, interior_gm_E_mean)

    # Numerical floor: if all E values are below declared floor,
    # signal has underflowed — T2*_eff << TE, exp(-TE/T2*) → 0.
    # 0/0 is not admissible. Declare as separation failed.
    # Floor: 1e-100 (well above float64 min ~5e-324).
    E_FLOOR = 1e-100
    if boundary_E_mean < E_FLOOR and max_interior < E_FLOOR:
        boundary_ratio      = 0.0
        separation_survived = False
    elif max_interior <= 0:
        boundary_ratio      = 0.0
        separation_survived = False
    else:
        boundary_ratio      = boundary_E_mean / max_interior
        separation_survived = boundary_ratio > domain.boundary_ratio_threshold

    return OperatorResult0e(
        b0=signal.b0,
        inhomogeneity_frac=signal.inhomogeneity_frac,
        inhomogeneity_ppm=signal.inhomogeneity_ppm,
        projection_name=domain.projection_name,
        boundary_ratio=float(boundary_ratio),
        boundary_E_mean=boundary_E_mean,
        interior_tumor_E_mean=interior_tumor_E_mean,
        interior_gm_E_mean=interior_gm_E_mean,
        separation_survived=separation_survived,
        rho_mean=float(rho.mean()),
    )


def _build_succ(src, tgt, rev, n_edges):
    succ_idx = np.empty(n_edges, dtype=np.int32)
    for e in range(n_edges):
        j, rev_e = int(tgt[e]), int(rev[e])
        candidates = [f for f in range(n_edges)
                      if int(src[f]) == j and f != rev_e]
        assert len(candidates) == 1
        succ_idx[e] = candidates[0]
    return succ_idx


def _build_pred(src, tgt, rev, n_edges):
    pred_idx = np.empty(n_edges, dtype=np.int32)
    for e in range(n_edges):
        i, rev_e = int(src[e]), int(rev[e])
        candidates = [f for f in range(n_edges)
                      if int(tgt[f]) == i and f != rev_e]
        assert len(candidates) == 1
        pred_idx[e] = candidates[0]
    return pred_idx


def _classify_edges(src, tgt, n_edges):
    boundary, tumor_int, gm_int = [], [], []
    for e in range(n_edges):
        s_tumor = int(src[e]) in TUMOR_SECTORS
        t_tumor = int(tgt[e]) in TUMOR_SECTORS
        if s_tumor != t_tumor:
            boundary.append(e)
        elif s_tumor:
            tumor_int.append(e)
        else:
            gm_int.append(e)
    return boundary, tumor_int, gm_int
