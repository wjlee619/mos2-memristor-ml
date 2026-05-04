"""
Hero figure: bipolar IV hysteresis loops for Chip#1 CC1-T14.
Run 33: negative sweep — stochastic filament nucleation.
Run 35: positive sweep — clean single-step SET.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

DATA_DIR = Path("/Users/wonjunlee/Desktop/imperial/Mres soft electronics/Data/15082024")
OUT = Path("/Users/wonjunlee/Desktop/취업 준비/mos2-memristor-ml/results/figures/00_iv_hysteresis_loops.png")

FILES = {
    "Run 33": DATA_DIR / "Chip#1-CC1-T14-MemEffect-Dark I-V Characterizer-1 Run 33 2024-08-15T16.38.34.csv",
    "Run 35": DATA_DIR / "Chip#1-CC1-T14-MemEffect-Dark I-V Characterizer-1 Run 35 2024-08-15T16.39.44.csv",
}

# From memeffect_sweep.csv (switching_state == 'switched')
ANNOT = {
    "Run 33": {"v_set": -12.8260, "v_reset": -3.3617, "i_on": 1.5833e-5},
    "Run 35": {"v_set":  19.4100, "v_reset":  1.4207, "i_on": 9.4779e-6},
}

# Nucleation window: Run 33 only
NUC_33 = (-13.5, -12.0)  # (lo, hi) in V — stochastic region from raw data inspection


def load_iv(path):
    """Parse Keithley CSV, return (v_fwd, i_fwd, v_ret, i_ret) as float arrays.
    Turnaround is exactly at row index 1999 (V = ±20 V).
    Summary rows (Min/Max/Mean/StdDev/CV) are dropped by requiring numeric Index.
    """
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        in_data = False
        for line in f:
            s = line.strip()
            if not in_data:
                if '"Index"' in s or ("SMU" in s and "Voltage" in s and "Current" in s):
                    in_data = True
                    continue
            else:
                parts = [p.strip().strip('"') for p in s.split(",")]
                if len(parts) == 4:
                    try:
                        int(parts[0])   # only numeric-index rows
                        rows.append((float(parts[2]), float(parts[3])))
                    except ValueError:
                        pass

    v_all = np.array([r[0] for r in rows])
    i_all = np.array([r[1] for r in rows])

    # Turnaround is at index 1999 (2000 forward points, 0-indexed)
    split = 1999
    v_fwd, i_fwd = v_all[:split + 1], np.abs(i_all[:split + 1])
    v_ret, i_ret = v_all[split + 1:], np.abs(i_all[split + 1:])
    return v_fwd, i_fwd, v_ret, i_ret


def log_y(ax, frac):
    """Return a y-value at fraction frac (0–1) through the log-scale y-axis."""
    lo, hi = ax.get_ylim()
    return 10 ** (np.log10(lo) + frac * (np.log10(hi) - np.log10(lo)))


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#fafafa")

C_FWD   = "#1f77b4"   # solid blue — forward
C_RET   = "#ff7f0e"   # dashed orange — return
C_SET   = "#d62728"   # red — V_SET line
C_RESET = "#7b2d8b"   # purple — V_RESET line
C_ION   = "#2ca02c"   # green — I_ON
C_IOFF  = "#8c564b"   # brown — I_OFF
C_NUC   = "#ff4444"   # nucleation shade

for ax, (label, fpath) in zip(axes, FILES.items()):
    v_fwd, i_fwd, v_ret, i_ret = load_iv(fpath)
    ann = ANNOT[label]
    neg = ann["v_set"] < 0

    # ── IV traces ──────────────────────────────────────────────────────────
    ax.semilogy(v_fwd, i_fwd, color=C_FWD, lw=1.6, label="Forward sweep", zorder=3)
    ax.semilogy(v_ret, i_ret, color=C_RET, lw=1.6, ls="--",
                label="Return sweep", zorder=3)

    # Fix y-axis limits before any annotation so log_y() is stable
    ax.set_ylim(5e-13, 5e-4)

    # ── Stochastic nucleation window (Run 33 only) ─────────────────────────
    if label == "Run 33":
        nlo, nhi = NUC_33
        ax.axvspan(nlo, nhi, color=C_NUC, alpha=0.12, zorder=1)
        ax.axvline(nlo, color=C_NUC, lw=0.7, ls=":", alpha=0.5)
        ax.axvline(nhi, color=C_NUC, lw=0.7, ls=":", alpha=0.5)

        # Label with arrow pointing into the shaded band
        ax.annotate(
            "stochastic\nnucleation\nwindow",
            xy=(-12.75, log_y(ax, 0.80)),
            xytext=(-10.5, log_y(ax, 0.88)),
            fontsize=8, color=C_NUC, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_NUC, lw=1.0),
        )
        # Mechanistic note inside the band
        ax.text(
            -12.75, log_y(ax, 0.55),
            "Filament nucleates, collapses\nand renucleates ×3\nbefore stabilising",
            fontsize=7.5, color="#880000", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_NUC,
                      alpha=0.85, lw=0.8),
            zorder=5,
        )

    # ── Clean SET annotation (Run 35 only) ────────────────────────────────
    if label == "Run 35":
        ax.annotate(
            "Clean single-step SET\n(3× jump in one data point)",
            xy=(ann["v_set"], log_y(ax, 0.70)),
            xytext=(ann["v_set"] - 6, log_y(ax, 0.80)),
            fontsize=8, color=C_SET, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_SET, lw=1.0),
        )

    # ── V_SET dashed vertical ─────────────────────────────────────────────
    ax.axvline(ann["v_set"], color=C_SET, lw=1.3, ls=":", zorder=4)
    vset_ha = "right" if neg else "left"
    ax.text(ann["v_set"], log_y(ax, 0.95),
            f"  $V_{{SET}}$={ann['v_set']:.1f} V  " if not neg else
            f"  $V_{{SET}}$\n  {ann['v_set']:.1f} V",
            color=C_SET, fontsize=8.5, va="top", ha=vset_ha, zorder=6)

    # ── V_RESET dashed vertical ───────────────────────────────────────────
    ax.axvline(ann["v_reset"], color=C_RESET, lw=1.3, ls=":", zorder=4)
    vreset_ha = "right" if neg else "left"
    ax.text(ann["v_reset"], log_y(ax, 0.22),
            f"  $V_{{RESET}}$\n  {ann['v_reset']:.2f} V",
            color=C_RESET, fontsize=8.5, va="bottom", ha=vreset_ha, zorder=6)

    # ── I_ON horizontal ───────────────────────────────────────────────────
    ax.axhline(ann["i_on"], color=C_ION, lw=1.0, ls="--", alpha=0.7, zorder=2)
    xlims = ax.get_xlim()
    x_label = xlims[0] + 0.03 * (xlims[1] - xlims[0])
    ax.text(x_label, ann["i_on"] * 2.5,
            f"$I_{{ON}}$ = {ann['i_on']*1e6:.1f} μA",
            color=C_ION, fontsize=8.5, va="bottom")

    # ── I_OFF horizontal (noise floor: mean of first 20 forward points) ───
    i_off_val = i_fwd[:20].mean()
    ax.axhline(i_off_val, color=C_IOFF, lw=1.0, ls=":", alpha=0.7, zorder=2)
    ax.text(x_label, i_off_val * 0.35,
            f"$I_{{OFF}}$ ≈ {i_off_val*1e12:.0f} pA",
            color=C_IOFF, fontsize=8.5, va="top")

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.set_xlabel("Applied Voltage (V)", fontsize=11)
    ax.set_ylabel("|Current| (A)", fontsize=11)
    hw = abs(ann["v_set"] - ann["v_reset"])
    sweep_dir = "negative" if neg else "positive"
    ax.set_title(
        f"{label}  —  {sweep_dir} sweep\n"
        f"$V_{{SET}}$ = {ann['v_set']:.1f} V  |  "
        f"Hysteresis window = {hw:.1f} V",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="lower right" if not neg else "lower left",
              framealpha=0.9)
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.set_facecolor("#f7f7f7")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.45, color="grey")
    ax.tick_params(labelsize=9)

fig.suptitle(
    "Bipolar memristive switching — Chip#1 CC1-T14  (MoS$_2$ memristor)",
    fontsize=13, fontweight="bold", y=1.01,
)
fig.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT}")

# Print numeric summary
for label, ann in ANNOT.items():
    v_fwd, i_fwd, v_ret, i_ret = load_iv(FILES[label])
    i_off_val = i_fwd[:20].mean()
    ratio = ann["i_on"] / i_off_val
    hw = abs(ann["v_set"] - ann["v_reset"])
    print(f"\n{label}:")
    print(f"  v_set              = {ann['v_set']:.2f} V")
    print(f"  v_reset            = {ann['v_reset']:.4f} V")
    print(f"  hysteresis window  = {hw:.2f} V")
    print(f"  i_on               = {ann['i_on']*1e6:.2f} uA")
    print(f"  i_off (noise flr)  = {i_off_val*1e12:.1f} pA")
    print(f"  on/off ratio       = {ratio:.0f}x")
    print(f"  forward pts        = {len(v_fwd)}")
    print(f"  return pts         = {len(v_ret)}")
