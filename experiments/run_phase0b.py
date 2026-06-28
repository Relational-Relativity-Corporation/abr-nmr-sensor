# experiments/run_phase0b.py — abr-nmr-phase0b
# Metatron Dynamics, Inc.
#
# Phase 0b: Multi-tissue boundary detection simulation.
#
# Runs the full declared chain:
#   declare_domain()        — 8-element array, 4-tissue phantom
#     → build_signal()      — per-element NMR signal with tissue mix
#       → run_operators()   — ABRCE kernel, boundary detection
#         → visualize()     — polar array map + E field plots
#
# The declared test: do the operators find elevated E field magnitude
# at edges crossing the declared tumor boundary, relative to edges
# within homogeneous tissue?
#
# Bounded over D. No claim beyond D.
# -----------------------------------------------------------------------

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.declaration_0b import declare_domain, N_ELEMENTS, TUMOR_SECTORS
from sim.signal_0b      import build_signal
from sim.operators_0b   import run_operators

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_PATH = os.path.join(OUTPUT_DIR, 'phase0b_report.txt')
FIG1_PATH   = os.path.join(OUTPUT_DIR, 'phase0b_array_map.png')
FIG2_PATH   = os.path.join(OUTPUT_DIR, 'phase0b_E_field.png')
FIG3_PATH   = os.path.join(OUTPUT_DIR, 'phase0b_boundary_detection.png')


class Tee:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log      = open(path, 'w', encoding='utf-8')
    def write(self, m):
        self.terminal.write(m); self.log.write(m)
    def flush(self):
        self.terminal.flush(); self.log.flush()
    def close(self):
        self.log.close()


def run():
    tee = Tee(REPORT_PATH)
    sys.stdout = tee
    t0 = time.time()

    print("=" * 65)
    print("abr-nmr-phase0b — Multi-tissue Boundary Detection")
    print("8-element solenoid array, 4-compartment phantom")
    print("Metatron Dynamics, Inc.")
    print("=" * 65)

    print("\n[1/4] Declaring domain...")
    domain = declare_domain()

    print("\n[2/4] Computing per-element signal...")
    signal = build_signal(domain)

    print("\n[3/4] Running ABRCE operators...")
    result = run_operators(domain, signal, rho_base=0.3)

    print("\n[4/4] Rendering figures...")
    _render_array_map(domain, signal, result)
    _render_E_field(domain, result)
    _render_boundary_detection(domain, result, signal)

    elapsed = time.time() - t0
    print(f"\nTotal run time: {elapsed:.1f}s")
    print(f"Figures → {OUTPUT_DIR}")
    print("Bounded over D. No claim beyond D.")
    print("=" * 65)

    sys.stdout = tee.terminal
    tee.close()


def _render_array_map(domain, signal, result):
    """
    Polar map of the 8-element array showing:
    - Element positions and tissue type (color)
    - Baseline signal amplitude (element size)
    - E field magnitude at each edge (edge color/width)
    - Declared tumor boundary (highlighted)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ---- Left panel: tissue map + signal amplitude -------------------
    ax1.set_aspect('equal')
    ax1.set_xlim(-55, 55); ax1.set_ylim(-55, 55)
    ax1.set_title('Array topology — tissue type and baseline signal\n'
                  '(element size ∝ S_baseline, color = tissue type)',
                  fontsize=10)

    # Draw phantom cross-section
    from matplotlib.patches import Circle, Wedge
    phantom = Circle((0, 0), 40, fill=False, color='gray',
                     linewidth=1, linestyle='--')
    ax1.add_patch(phantom)

    # Color scheme by dominant tissue
    tissue_colors = {
        'tumor':       '#FF4444',
        'gray_matter': '#4444FF',
        'white_matter': '#888888',
        'CSF':         '#44AAFF',
    }

    # Draw tissue wedges
    for el in domain.elements:
        a_start = el.angle_start_deg
        a_end   = el.angle_end_deg
        # Dominant core tissue
        if el.element_id in TUMOR_SECTORS:
            core_color = '#FF4444'
            core_label = 'Tumor'
        else:
            core_color = '#4444FF'
            core_label = 'Gray matter'

        # Draw core wedge
        wedge = Wedge((0,0), 15, a_start-90, a_end-90,
                      color=core_color, alpha=0.3)
        ax1.add_patch(wedge)
        # White matter ring
        wm_wedge = Wedge((0,0), 35, a_start-90, a_end-90,
                         width=20, color='#888888', alpha=0.2)
        ax1.add_patch(wm_wedge)
        # CSF rim
        csf_wedge = Wedge((0,0), 40, a_start-90, a_end-90,
                          width=5, color='#44AAFF', alpha=0.3)
        ax1.add_patch(csf_wedge)

    # Draw elements
    for el in domain.elements:
        angle_rad = np.radians(el.angle_center_deg)
        x = 45 * np.cos(angle_rad)
        y = 45 * np.sin(angle_rad)

        s_norm = float(signal.S_baseline[el.element_id])
        size   = 200 + 3000 * s_norm

        color = '#FF4444' if el.element_id in TUMOR_SECTORS else '#4444FF'
        ax1.scatter(x, y, s=size, c=color, zorder=5,
                    edgecolors='white', linewidths=1.5)
        ax1.text(x*1.15, y*1.15, str(el.element_id),
                 ha='center', va='center', fontsize=9, fontweight='bold')

        # S_baseline label
        ax1.text(x*0.85, y*0.85,
                 f'{signal.S_baseline[el.element_id]:.4f}',
                 ha='center', va='center', fontsize=6, color='white')

    # Legend
    patches = [
        mpatches.Patch(color='#FF4444', alpha=0.6, label='Tumor (92% H₂O)'),
        mpatches.Patch(color='#4444FF', alpha=0.6, label='Gray matter (84% H₂O)'),
        mpatches.Patch(color='#888888', alpha=0.4, label='White matter (72% H₂O)'),
        mpatches.Patch(color='#44AAFF', alpha=0.4, label='CSF (99% H₂O)'),
    ]
    ax1.legend(handles=patches, loc='lower left', fontsize=8)
    ax1.set_xlabel('Lateral [mm]', fontsize=9)
    ax1.set_ylabel('Depth [mm]', fontsize=9)

    # ---- Right panel: E field at edges -------------------------------
    ax2.set_aspect('equal')
    ax2.set_xlim(-55, 55); ax2.set_ylim(-55, 55)
    ax2.set_title('E field at array edges — mean |E| over all steps\n'
                  '(edge width and color ∝ |E|, red = boundary)',
                  fontsize=10)

    # Redraw phantom outline
    ax2.add_patch(Circle((0, 0), 40, fill=False, color='gray',
                         linewidth=1, linestyle='--'))

    # Draw edges with E field magnitude
    E_mean_per_edge = np.abs(result.E_field).mean(axis=1)  # [n_edges]
    E_max = float(E_mean_per_edge.max()) + 1e-20

    for e, (src, tgt) in enumerate(domain.adjacency):
        # Only draw clockwise edges (first N_ELEMENTS)
        if e >= N_ELEMENTS:
            continue
        a_src = np.radians(domain.elements[src].angle_center_deg)
        a_tgt = np.radians(domain.elements[tgt].angle_center_deg)
        x1, y1 = 45*np.cos(a_src), 45*np.sin(a_src)
        x2, y2 = 45*np.cos(a_tgt), 45*np.sin(a_tgt)

        e_val   = float(E_mean_per_edge[e])
        e_norm  = e_val / E_max

        is_boundary = e in result.boundary_edges
        color  = '#FF6B35' if is_boundary else '#00B4D8'
        lw     = 1.0 + 8.0 * e_norm
        alpha  = 0.4 + 0.6 * e_norm

        ax2.plot([x1, x2], [y1, y2], color=color,
                 linewidth=lw, alpha=alpha, zorder=3)
        # Midpoint label
        xm, ym = (x1+x2)/2, (y1+y2)/2
        ax2.text(xm*1.05, ym*1.05, f'{e_val:.2e}',
                 fontsize=6, ha='center', color=color)

    # Draw elements
    for el in domain.elements:
        angle_rad = np.radians(el.angle_center_deg)
        x = 45*np.cos(angle_rad)
        y = 45*np.sin(angle_rad)
        color = '#FF4444' if el.element_id in TUMOR_SECTORS else '#4444FF'
        ax2.scatter(x, y, s=150, c=color, zorder=5,
                    edgecolors='white', linewidths=1.5)
        ax2.text(x*1.15, y*1.15, str(el.element_id),
                 ha='center', va='center', fontsize=9, fontweight='bold')

    edge_patches = [
        mpatches.Patch(color='#FF6B35', label='Boundary edge (tumor↔GM)'),
        mpatches.Patch(color='#00B4D8', label='Interior edge'),
    ]
    ax2.legend(handles=edge_patches, loc='lower left', fontsize=8)
    ax2.set_xlabel('Lateral [mm]', fontsize=9)
    ax2.set_ylabel('Depth [mm]', fontsize=9)

    fig.suptitle(
        'abr-nmr-phase0b — 8-Element Array: Tissue Map and Relational Field\n'
        'Metatron Dynamics, Inc.  |  Bounded over D',
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    fig.savefig(FIG1_PATH, dpi=150, bbox_inches='tight')
    print(f"  Figure 1 → {FIG1_PATH}")
    plt.close(fig)


def _render_E_field(domain, result):
    """
    E field at each edge across all cardiac evolution steps.
    Boundary edges highlighted vs interior edges.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                              gridspec_kw={'height_ratios': [2, 2, 1]})

    n_steps = result.E_field.shape[1]
    steps   = np.arange(n_steps)

    # Panel 1: E field at boundary edges
    ax1 = axes[0]
    for e in result.boundary_edges:
        src, tgt = domain.adjacency[e]
        ax1.plot(steps, np.abs(result.E_field[e, :]),
                 linewidth=1.0, alpha=0.9,
                 label=f'Edge {e}: {src}→{tgt} (boundary)')
    ax1.set_ylabel('|E|', fontsize=10)
    ax1.set_title('|E| at declared tumor boundary edges', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, n_steps)

    # Panel 2: E field at interior edges
    ax2 = axes[1]
    for e in result.interior_edges_tumor:
        src, tgt = domain.adjacency[e]
        ax2.plot(steps, np.abs(result.E_field[e, :]),
                 color='#FF4444', linewidth=0.8, alpha=0.7,
                 label=f'Edge {e}: {src}→{tgt} (tumor interior)')
    for e in result.interior_edges_gm:
        src, tgt = domain.adjacency[e]
        ax2.plot(steps, np.abs(result.E_field[e, :]),
                 color='#4444FF', linewidth=0.8, alpha=0.7,
                 label=f'Edge {e}: {src}→{tgt} (GM interior)')
    ax2.set_ylabel('|E|', fontsize=10)
    ax2.set_title('|E| at interior edges (red=tumor, blue=GM)', fontsize=10)
    ax2.legend(fontsize=7)
    ax2.set_xlim(0, n_steps)

    # Panel 3: cardiac phase driver
    ax3 = axes[2]
    phase_deg = np.degrees(result.A_field)   # use cardiac phase from signal
    # Recompute phase for display
    advance = float(domain.cardiac_phase_advance_rad)
    t_idx   = np.arange(n_steps, dtype=np.float32)
    phase   = np.degrees(
        (t_idx * advance).astype(np.float64) % (2.0 * np.pi)
    )
    ax3.plot(steps, phase, color='#00B4D8', linewidth=0.6)
    ax3.set_ylabel('Phase [°]', fontsize=9)
    ax3.set_xlabel('Cardiac evolution index', fontsize=10)
    ax3.set_title('Declared cardiac phase — evolution driver', fontsize=9)
    ax3.set_xlim(0, n_steps)
    ax3.set_ylim(0, 360)

    fig.suptitle(
        'abr-nmr-phase0b — E Field at Declared Edge Classes\n'
        'Metatron Dynamics, Inc.  |  Bounded over D',
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    fig.savefig(FIG2_PATH, dpi=150, bbox_inches='tight')
    print(f"  Figure 2 → {FIG2_PATH}")
    plt.close(fig)


def _render_boundary_detection(domain, result, signal):
    """
    Direct comparison of mean |E| at boundary vs interior edges.
    This is the primary boundary detection result.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    n_steps = result.E_field.shape[1]
    steps   = np.arange(n_steps)

    # Panel 1: mean |E| per edge class over evolution
    ax1 = axes[0]
    ax1.plot(steps, result.boundary_E_mean,
             color='#FF6B35', linewidth=1.5,
             label='Boundary edges (tumor↔GM)')
    ax1.plot(steps, result.interior_tumor_E_mean,
             color='#FF4444', linewidth=1.0, linestyle='--',
             label='Tumor interior edges')
    ax1.plot(steps, result.interior_gm_E_mean,
             color='#4444FF', linewidth=1.0, linestyle='--',
             label='GM interior edges')
    ax1.set_xlabel('Cardiac evolution index', fontsize=10)
    ax1.set_ylabel('Mean |E|', fontsize=10)
    ax1.set_title('Mean |E| per edge class across evolution\n'
                  'Boundary should exceed interior if detected',
                  fontsize=10)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, n_steps)

    # Panel 2: per-element baseline signal — tissue differentiation
    ax2 = axes[1]
    elements = list(range(N_ELEMENTS))
    colors   = ['#FF4444' if i in TUMOR_SECTORS else '#4444FF'
                for i in elements]
    bars = ax2.bar(elements, signal.S_baseline,
                   color=colors, alpha=0.8, edgecolor='white')

    ax2.set_xlabel('Element index', fontsize=10)
    ax2.set_ylabel('S_baseline (signal amplitude)', fontsize=10)
    ax2.set_title('Per-element baseline signal\n'
                  'Red=tumor sector, Blue=GM sector\n'
                  'Difference = water content contrast',
                  fontsize=10)
    ax2.set_xticks(elements)

    # Add modulation depth labels
    for i, (bar, mod) in enumerate(zip(bars, signal.modulation_depth)):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.001,
                 f'mod={mod*100:.3f}%',
                 ha='center', va='bottom', fontsize=7, rotation=45)

    # Declared tissue labels
    ax2.text(1.5, max(signal.S_baseline)*0.5,
             'TUMOR\n(92% H₂O)',
             ha='center', color='#FF4444', fontsize=10, fontweight='bold')
    ax2.text(5.5, max(signal.S_baseline)*0.5,
             'GRAY MATTER\n(84% H₂O)',
             ha='center', color='#4444FF', fontsize=10, fontweight='bold')

    # Annotate signal difference
    tumor_mean = float(np.array([signal.S_baseline[i]
                                  for i in TUMOR_SECTORS]).mean())
    gm_mean    = float(np.array([signal.S_baseline[i]
                                  for i in {4,5,6,7}]).mean())
    diff_pct   = (tumor_mean - gm_mean) / gm_mean * 100
    ax2.annotate(
        f'Δ = {diff_pct:.1f}%\n(water content contrast)',
        xy=(3.5, (tumor_mean + gm_mean)/2),
        xytext=(3.5, max(signal.S_baseline)*0.85),
        ha='center', fontsize=9, color='black',
        arrowprops=dict(arrowstyle='->', color='black'),
    )

    fig.suptitle(
        'abr-nmr-phase0b — Boundary Detection and Tissue Differentiation\n'
        'Metatron Dynamics, Inc.  |  Bounded over D',
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    fig.savefig(FIG3_PATH, dpi=150, bbox_inches='tight')
    print(f"  Figure 3 → {FIG3_PATH}")
    plt.close(fig)


if __name__ == '__main__':
    run()
