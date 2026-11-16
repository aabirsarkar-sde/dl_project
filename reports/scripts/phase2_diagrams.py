"""Phase 2 explanatory figures built from the real FD001 training data (no model results needed).

Writes light versions (for the IEEE report) to reports/figures/ and dark versions
(for the slide deck) to reports/figures/deck/.
Run from the repo root:  python reports/scripts/phase2_diagrams.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT_LIGHT = ROOT / "reports" / "figures"
OUT_DARK = OUT_LIGHT / "deck"
OUT_DARK.mkdir(parents=True, exist_ok=True)

cols = ["engine_id", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
train = pd.read_csv(ROOT / "data" / "train_FD001.txt", sep=r"\s+", header=None, names=cols)
train["RUL"] = train.groupby("engine_id")["cycle"].transform("max") - train["cycle"]
keep = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

THEMES = {
    "light": dict(bg="white", fg="#1a1a1a", muted="#666666", grid="#dddddd",
                  a1="#3B4CCA", a2="#D6339B", a3="#0E8FB5", box="#EEF0FB", out=OUT_LIGHT),
    "dark": dict(bg="#0B0E17", fg="#F2F4FF", muted="#9AA3C7", grid="#232A45",
                 a1="#7C86FF", a2="#F04FC4", a3="#5FD3F5", box="#161B33", out=OUT_DARK),
}


def style(ax, t):
    ax.set_facecolor(t["bg"])
    for s in ax.spines.values():
        s.set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9)
    ax.xaxis.label.set_color(t["fg"])
    ax.yaxis.label.set_color(t["fg"])
    ax.title.set_color(t["fg"])


def window_figure(t, name):
    """Engine 1, sensor s11: two windows of 30 cycles ending at different points, plus padding."""
    e = train[train.engine_id == 1]
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=t["bg"])
    style(ax, t)
    ax.plot(e.cycle, e.s11, color=t["muted"], lw=1)
    W = 30
    for end, col, lab in [(120, t["a1"], "window ending at cycle 120"), (e.cycle.max(), t["a2"], "window ending at last cycle")]:
        ax.axvspan(end - W + 0.5, end + 0.5, color=col, alpha=0.18)
        seg = e[(e.cycle > end - W) & (e.cycle <= end)]
        ax.plot(seg.cycle, seg.s11, color=col, lw=2, label=lab)
        rul = int(e.loc[e.cycle == end, "RUL"].iloc[0])
        ax.annotate(f"target: RUL = {rul}", xy=(end, seg.s11.iloc[-1]), xytext=(end - 28, e.s11.max() + 0.05),
                    color=col, fontsize=9, arrowprops=dict(arrowstyle="->", color=col, lw=1))
    # padding illustration for an early cycle
    ax.axvspan(0.5, 10.5, color=t["a3"], alpha=0.15)
    ax.text(1, e.s11.min() - 0.02, "cycle 10: 20 slots padded\nwith the first reading", color=t["a3"], fontsize=8, va="top")
    ax.set_xlabel("Cycle (engine 1)")
    ax.set_ylabel("Sensor s11 (raw)")
    ax.set_ylim(e.s11.min() - 0.15, e.s11.max() + 0.15)
    leg = ax.legend(loc="upper left", fontsize=8, frameon=False)
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    fig.tight_layout()
    fig.savefig(t["out"] / name, dpi=200, facecolor=t["bg"])
    plt.close(fig)


def heatmap_figure(t, name):
    """What the LSTM sees: one 30-cycle window of 14 normalised sensors, early vs late in life."""
    mu, sd = train[keep].mean(), train[keep].std()
    e = train[train.engine_id == 1]
    last = e.cycle.max()
    wins = {f"cycles 31-60  (RUL {int(e.RUL.iloc[59])}-{int(e.RUL.iloc[30])})": e.iloc[30:60],
            f"cycles {last - 29}-{last}  (RUL 29-0)": e.iloc[-30:]}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=t["bg"], sharey=True)
    for ax, (title, w) in zip(axes, wins.items()):
        style(ax, t)
        z = ((w[keep] - mu) / sd).values.T
        im = ax.imshow(z, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3,
                       extent=[w.cycle.min() - 0.5, w.cycle.max() + 0.5, len(keep) - 0.5, -0.5])
        ax.set_yticks(range(len(keep)))
        ax.set_yticklabels(keep, fontsize=7)
        ax.set_xlabel("Cycle")
        ax.set_title(title, fontsize=10)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("z-score (training mean/std)", color=t["fg"], fontsize=8)
    cb.ax.tick_params(colors=t["muted"], labelsize=7)
    cb.outline.set_edgecolor(t["grid"])
    fig.savefig(t["out"] / name, dpi=200, facecolor=t["bg"], bbox_inches="tight")
    plt.close(fig)


def arch_figure(t, name):
    """Hybrid architecture block diagram."""
    fig, ax = plt.subplots(figsize=(9, 3.4), facecolor=t["bg"])
    ax.set_facecolor(t["bg"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, title, sub, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    fc=t["box"], ec=col, lw=1.6))
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", color=t["fg"], fontsize=10, weight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", color=t["muted"], fontsize=7.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     color=t["muted"], lw=1.2))

    box(1, 25, 19, 11, "Raw window", "30 cycles x 14 sensors\nz-scored per fold", t["a1"])
    box(26, 25, 19, 11, "2-layer LSTM", "hidden 64, dropout 0.2", t["a1"])
    box(51, 25, 15, 11, "Last state", "64 values", t["a1"])
    box(1, 4, 19, 11, "Current row", "cycle + 14 sensors", t["a2"])
    box(26, 4, 19, 11, "Phase 1 features", "57 values: mean, slope,\nhealth (standardised)", t["a2"])
    box(68, 14, 11, 12, "Concat", "121 values", t["a3"])
    box(83, 14, 16, 12, "Head", "32 ReLU + dropout\n1 output = RUL/100", t["a3"])
    arrow(20.8, 30.5, 25.2, 30.5)
    arrow(45.8, 30.5, 50.2, 30.5)
    arrow(20.8, 9.5, 25.2, 9.5)
    arrow(66.8, 30.5, 71, 26.6)
    arrow(45.8, 9.5, 71, 13.4)
    arrow(79.8, 20, 82.2, 20)
    ax.text(1, 38.5, "Short-term sequence (learned)", color=t["a1"], fontsize=8.5)
    ax.text(1, 0.3, "Long-term context (engineered, incl. cycle and drift from first 10 cycles)", color=t["a2"], fontsize=8.5)
    fig.savefig(t["out"] / name, dpi=200, facecolor=t["bg"], bbox_inches="tight")
    plt.close(fig)


def degradation_figure(t, name):
    """Sensor s11 for all 100 engines vs cycles-to-failure: why the window alone cannot see early life."""
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=t["bg"])
    style(ax, t)
    for _, d in train.groupby("engine_id"):
        ax.plot(-d.RUL, d.s11, color=t["a1"], alpha=0.18, lw=0.7)
    ax.axvspan(-30, 0, color=t["a2"], alpha=0.15)
    ax.text(-29, train.s11.max() - 0.05, "last 30 cycles:\nclear upward drift", color=t["a2"], fontsize=8, va="top")
    ax.axvspan(-361, -150, color=t["a3"], alpha=0.08)
    ax.text(-355, train.s11.max() - 0.05, "RUL > 150: flat, mostly noise,\nengines look alike", color=t["a3"], fontsize=8, va="top")
    ax.set_xlabel("Cycles before failure (-RUL)")
    ax.set_ylabel("Sensor s11 (raw)")
    fig.tight_layout()
    fig.savefig(t["out"] / name, dpi=200, facecolor=t["bg"])
    plt.close(fig)


for key, t in THEMES.items():
    window_figure(t, "phase2_windows.png")
    heatmap_figure(t, "phase2_window_heatmap.png")
    arch_figure(t, "phase2_hybrid_architecture.png")
    degradation_figure(t, "phase2_s11_all_engines.png")
print("Wrote figures to", OUT_LIGHT, "and", OUT_DARK)
