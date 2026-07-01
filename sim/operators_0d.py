# sim/operators_0d.py — abr-nmr-phase0d
# Metatron Dynamics, Inc.
#
# ABRCE operators for Phase 0d sweep.
#
# The operator kernel is identical to Phase 0c (operators_0c.py).
# This file wraps it to accept a SignalResult0d and return the
# quantities the sweep needs: boundary ratio and operator separation.
#
# For the sweep, we do not need the full E_field timeseries.
# We need:
#   boundary_ratio : float
#       mean |E| at boundary edges / mean |E| at interior edges
#       (maximum of tumor-interior and GM-interior denominators)
#   separation_survived : bool
#       boundary_ratio > declared threshold
#   boundary_E_mean : float
#   interior_tumor_E_mean : float
#   interior_gm_E_mean : float
#
# The operator formulas are unchanged:
#   A[e,t] = S[src(e),t] - S[tgt(e),t]
#   B[e]   = A[e] + A[succ(e)]
#   E[e]   = B[e] + rho[src(e)] * (B[succ(e)] - B[pred(e)])
#
# rho_base is held at 0.3 (Phase 0c declared value).
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass

from sim.declaration_0d import (
    DeclaredDomain0d,
    N_ELEMENTS,
    TUMOR_SECTORS,
    GRAY_MAT_SECTORS,
    C_PROJECTION_NAME,
)
from sim.signal_0d import SignalResult0d

RHO_BASE_DEFAULT = 0.3


@dataclass
class OperatorResult0d:
    """
    Operator output quantities for one B0 sweep step.

    Attributes
    ----------
    b0 : float
    projection_name : str
        Canonical name of the declared C projection used to compute
        boundary_ratio. Must match C_PROJECTION_NAME in declaration_0d.
    boundary_ratio : float
        C_mean_abs_ratio: mean|E| at boundary edges divided by
        max(mean|E| at tumor interior, mean|E| at GM interior).
        Mean taken over all edges in class AND all cardiac steps.
        See declaration_0d.py for full provenance statement.
    boundary_E_mean : float
        mean( |E_field[e,t]| ) over boundary edges and all steps.
    interior_tumor_E_mean : float
        mean( |E_field[e,t]| ) over tumor interior edges and all steps.
    interior_gm_E_mean : float
        mean( |E_field[e,t]| ) over GM interior edges and all steps.
    separation_survived : bool
        boundary_ratio > declared BOUNDARY_RATIO_THRESHOLD.
    rho_mean : float
        Mean rho across all elements (for reporting).
    """
    b0:                    float
    projection_name:       str
    boundary_ratio:        float
    boundary_E_mean:       float
    interior_tumor_E_mean: float
    interior_gm_E_mean:    float
    separation_survived:   bool
    rho_mean:              float


def run_operators(
    domain: DeclaredDomain0d,
    signal: SignalResult0d,
    rho_base: float = RHO_BASE_DEFAULT,
) -> OperatorResult0d:
    """
    Apply ABRCE operators to one sweep step.

    Parameters
    ----------
    domain : DeclaredDomain0d
    signal : SignalResult0d
    rho_base : float

    Returns
    -------
    OperatorResult0d
    """
    n_edges = len(domain.adjacency)
    S       = signal.S_field   # [N_ELEMENTS, n_steps]

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

    # ---- Operator B (vectorized ring) --------------------------------
    succ_idx = _build_succ(domain, src, tgt, rev, n_edges)
    B_field  = A_field + A_field[succ_idx, :]

    # ---- Operator R (vectorized ring) --------------------------------
    pred_idx = _build_pred(domain, src, tgt, rev, n_edges)
    rho_e    = rho[src]
    fwd      = B_field[succ_idx, :]
    bwd      = B_field[pred_idx, :]
    E_field  = B_field + rho_e[:, None] * (fwd - bwd)

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

    # ---- C_mean_abs_ratio projection ---------------------------------
    # Declared projection: mean( |E_field[e,:]| ) over edge class and
    # all cardiac steps. See declaration_0d.C_PROJECTION_NAME.
    # Preserves: average magnitude. Discards: per-edge and per-step
    # variation, sign, distribution shape.
    def C_mean_abs_ratio(edge_list):
        """
        Declared C projection: mean absolute E over edge class and steps.
        Preserves average magnitude. Discards per-edge/step variation and sign.
        """
        if not edge_list:
            return 0.0
        return float(np.abs(E_field[edge_list, :]).mean())

    boundary_E_mean       = C_mean_abs_ratio(boundary_edges)
    interior_tumor_E_mean = C_mean_abs_ratio(interior_edges_tumor)
    interior_gm_E_mean    = C_mean_abs_ratio(interior_edges_gm)

    # Denominator: max of the two interior means (conservative)
    max_interior = max(interior_tumor_E_mean, interior_gm_E_mean)
    if max_interior > 0:
        boundary_ratio = boundary_E_mean / max_interior
    else:
        boundary_ratio = float('inf')

    separation_survived = boundary_ratio > domain.boundary_ratio_threshold

    return OperatorResult0d(
        b0=signal.b0,
        projection_name=C_PROJECTION_NAME,
        boundary_ratio=float(boundary_ratio),
        boundary_E_mean=boundary_E_mean,
        interior_tumor_E_mean=interior_tumor_E_mean,
        interior_gm_E_mean=interior_gm_E_mean,
        separation_survived=separation_survived,
        rho_mean=float(rho.mean()),
    )


def _build_succ(domain, src, tgt, rev, n_edges):
    """Build successor index array for vectorized B operator."""
    succ_idx = np.empty(n_edges, dtype=np.int32)
    for e in range(n_edges):
        j     = int(tgt[e])
        rev_e = int(rev[e])
        candidates = [f for f in range(n_edges)
                      if int(src[f]) == j and f != rev_e]
        assert len(candidates) == 1
        succ_idx[e] = candidates[0]
    return succ_idx


def _build_pred(domain, src, tgt, rev, n_edges):
    """Build predecessor index array for vectorized R operator."""
    pred_idx = np.empty(n_edges, dtype=np.int32)
    for e in range(n_edges):
        i     = int(src[e])
        rev_e = int(rev[e])
        candidates = [f for f in range(n_edges)
                      if int(tgt[f]) == i and f != rev_e]
        assert len(candidates) == 1
        pred_idx[e] = candidates[0]
    return pred_idx
