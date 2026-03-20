"""
Generate paper figures.

Figure 1: Canonical high score vs NPN collapse
  - Left: MLP predictions vs true on canonical test set (scatter, r_s=0.79)
  - Right: Same model, NPN-transformed inputs (scatter, r_s=0.25)
  - Uses real per-sample predictions from fig1_predictions.npz

Figure 2: Contamination source decomposition
  - Bar chart: NPN prediction std by feature group
  - Annotated with canonical r_s and NPN-averaged r_s

Figure 3: Augmentation curve
  - x-axis: augmentation level (0×, 1×, 3×, 7×)
  - y-axis: NPN-averaged r_s (the genuine invariant performance)
  - Horizontal lines: linear baseline (0.56), invariant-measure ceiling (0.635)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")
fig_dir = os.path.join(script_dir, "..", "figures")
os.makedirs(fig_dir, exist_ok=True)

# Shared style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

COLORS = {
    'clean': '#2ecc71',
    'low_bias': '#f1c40f',
    'severe_bits': '#e74c3c',
    'severe_fourier': '#e67e22',
    'noninv': '#f39c12',
    'augmented': '#3498db',
    'canonical': '#95a5a6',
    'baseline': '#bdc3c7',
}


def fig1_canonical_vs_npn():
    """Figure 1: Canonical high score vs NPN collapse (real predictions)."""
    pred_path = os.path.join(data_dir, "fig1_predictions.npz")
    if not os.path.exists(pred_path):
        print("  [SKIP] fig1_predictions.npz not found — run save_fig1_predictions.py first")
        return

    d = np.load(pred_path)
    true_targets = d["true_targets"]
    canonical_preds = d["canonical_preds"]
    transform_preds = d["transform_preds"]  # (N, K)
    rs_canonical = float(d["rs_canonical"])
    rs_npn_avg = float(d["rs_npn_avg"])
    mae_canonical = float(d["mae_canonical"])
    mae_npn_avg = float(d["mae_npn_avg"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rng = np.random.RandomState(42)

    # Left panel: canonical predictions (real)
    ax = axes[0]
    jitter = rng.uniform(-0.2, 0.2, len(true_targets))
    ax.scatter(true_targets + jitter, canonical_preds,
               alpha=0.25, s=10, c='#3498db', edgecolors='none')
    ax.plot([0, 12], [0, 12], 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel('True circuit size (gates)')
    ax.set_ylabel('Predicted circuit size')
    ax.set_title(f'(a) Canonical test set\n$r_s$ = {rs_canonical:.2f}, MAE = {mae_canonical:.2f}')
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 12.5)
    ax.set_aspect('equal')

    # Right panel: orbit-averaged predictions
    ax = axes[1]
    npn_averaged_preds = d["npn_averaged_preds"]

    jitter2 = rng.uniform(-0.2, 0.2, len(true_targets))
    ax.scatter(true_targets + jitter2, npn_averaged_preds,
               alpha=0.25, s=10, c='#e74c3c', edgecolors='none')
    ax.plot([0, 12], [0, 12], 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel('True circuit size (gates)')
    ax.set_ylabel('Predicted circuit size')
    ax.set_title(f'(b) NPN-averaged predictions\n$r_s$ = {rs_npn_avg:.2f}, MAE = {mae_npn_avg:.2f}')
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 12.5)
    ax.set_aspect('equal')

    plt.tight_layout()
    out = os.path.join(fig_dir, "fig1_canonical_vs_npn.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def fig2_contamination_hierarchy():
    """Figure 2: Contamination source decomposition."""
    decomp_path = os.path.join(data_dir, "npn_contamination_decomp.json")
    if not os.path.exists(decomp_path):
        print("  [SKIP] npn_contamination_decomp.json not found")
        return

    with open(decomp_path) as f:
        decomp = json.load(f)

    # Order: most clean to most contaminated
    order = ["C_invariant_measures", "C_all_measures", "C_noninvariant_measures",
             "B_fourier", "A_bits"]
    labels = ["Invariant\nmeasures (7)", "All\nmeasures (10)",
              "Non-invariant\nmeasures (3)", "Fourier\nspectrum (32)",
              "Truth table\nbits (32)"]
    colors = [COLORS['clean'], COLORS['low_bias'], COLORS['noninv'],
              COLORS['severe_fourier'], COLORS['severe_bits']]

    stds = [decomp[k]["mean_std"] for k in order]
    can_rs = [decomp[k]["canonical_rs"] for k in order]
    npn_rs = [decomp[k]["npn_averaged_rs"] for k in order]
    consist = [decomp[k]["consistency_rate"] for k in order]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(order))
    width = 0.5

    bars = ax1.bar(x, stds, width, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('NPN prediction std (gates)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.axhline(y=0, color='black', linewidth=0.5)

    # Annotate bars
    for i, (bar, s, cr, nr, con) in enumerate(zip(bars, stds, can_rs, npn_rs, consist)):
        # std value on bar
        if s > 0.05:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                     f'{s:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax1.text(bar.get_x() + bar.get_width()/2, 0.05,
                     f'{s:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # r_s values below
        ax1.text(bar.get_x() + bar.get_width()/2, -0.25,
                 f'can: {cr:.3f}\nnpn: {nr:.3f}', ha='center', va='top',
                 fontsize=8, color='#555555')

    ax1.set_ylim(-0.6, 2.0)
    ax1.set_title('Contamination Source Decomposition:\nNPN prediction std by feature group',
                  fontsize=13, pad=10)

    # Legend
    clean_patch = mpatches.Patch(color=COLORS['clean'], label='NPN-invariant (clean)')
    bias_patch = mpatches.Patch(color=COLORS['severe_bits'], label='Canonical bias (severe)')
    ax1.legend(handles=[clean_patch, bias_patch], loc='upper left', fontsize=10)

    plt.tight_layout()
    out = os.path.join(fig_dir, "fig2_contamination_hierarchy.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def fig3_augmentation_curve():
    """Figure 3: Augmentation curve."""
    aug_path = os.path.join(data_dir, "npn_augmentation_results.json")
    if not os.path.exists(aug_path):
        print("  [SKIP] npn_augmentation_results.json not found (still running?)")
        return

    with open(aug_path) as f:
        aug = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    configs = ["no_augment", "1x_augment", "3x_augment", "7x_augment"]
    x_labels = ["0× (canonical)", "1×", "3×", "7×"]
    x = np.arange(len(configs))

    available = [c for c in configs if c in aug]
    x_avail = [i for i, c in enumerate(configs) if c in aug]

    can_rs = [aug[c]["npn_invariance"]["canonical_rs"] for c in available]
    npn_rs = [aug[c]["npn_invariance"]["npn_averaged_rs"] for c in available]
    npn_std = [aug[c]["npn_invariance"]["mean_std"] for c in available]

    # Plot lines
    ax.plot(x_avail, can_rs, 'o-', color='#95a5a6', linewidth=2, markersize=8,
            label='Canonical $r_s$', zorder=3)
    ax.plot(x_avail, npn_rs, 's-', color='#3498db', linewidth=2.5, markersize=8,
            label='NPN-averaged $r_s$ (genuine)', zorder=3)

    # Reference lines
    ax.axhline(y=0.635, color=COLORS['clean'], linestyle='--', linewidth=1.5,
               label='Classical invariant ceiling ($r_s$ = 0.635)')
    ax.axhline(y=0.563, color=COLORS['baseline'], linestyle=':', linewidth=1.5,
               label='Linear baseline ($r_s$ = 0.563)')

    # Annotate NPN-averaged points
    for xi, nr in zip(x_avail, npn_rs):
        ax.annotate(f'{nr:.3f}', (xi, nr), textcoords="offset points",
                    xytext=(12, -5), fontsize=10, color='#2980b9', fontweight='bold')

    # Shade the "leakage gap"
    if len(x_avail) >= 2:
        ax.fill_between(x_avail, can_rs, npn_rs, alpha=0.12, color='#e74c3c',
                         label='Canonical leakage')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('NPN augmentation level', fontsize=12)
    ax.set_ylabel('Spearman $r_s$', fontsize=12)
    ax.set_title('Effect of NPN Augmentation on Genuine vs. Canonical Performance',
                  fontsize=13, pad=10)
    ax.set_ylim(0.15, 0.85)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(fig_dir, "fig3_augmentation_curve.pdf")
    fig.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def main():
    print("Generating paper figures...")
    print("\nFigure 1: Canonical vs NPN collapse")
    fig1_canonical_vs_npn()
    print("\nFigure 2: Contamination hierarchy")
    fig2_contamination_hierarchy()
    print("\nFigure 3: Augmentation curve")
    fig3_augmentation_curve()
    print("\nDone.")


if __name__ == "__main__":
    main()
