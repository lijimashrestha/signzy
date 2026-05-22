"""
Signzy YOLOv8 Training Visualizer
----------------------------------
Requirements:  pip install pandas matplotlib
Usage:         Place this file in the same folder as results.csv, then run:
               python signzy_graphs.py
"""
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# ── CONFIG — change this path if results.csv is elsewhere ─────────────────────
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
epochs = df["epoch"]

# ── Theme ──────────────────────────────────────────────────────────────────────
BG        = "#0D0F14"
PANEL     = "#13161E"
GRID      = "#1E2230"
TEXT      = "#E8EAF0"
MUTED     = "#5B607A"
ACCENT    = "#7C83FD"

TRAIN_BOX = "#FF6B6B";  VAL_BOX   = "#EF476F"
TRAIN_CLS = "#FFD166";  VAL_CLS   = "#FCA311"
TRAIN_DFL = "#06D6A0";  VAL_DFL   = "#00B4D8"
MAP50_C   = "#A78BFA";  MAP5095_C = "#38BDF8"
PREC_C    = "#FB923C";  REC_C     = "#34D399"

plt.rcParams.update({
    "font.family":      "monospace",
    "text.color":       TEXT,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TEXT,
    "axes.titlecolor":  TEXT,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "axes.labelsize":   9,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "grid.color":       GRID,
    "grid.linewidth":   0.6,
    "legend.facecolor": "#1A1D27",
    "legend.edgecolor": GRID,
    "legend.labelcolor":TEXT,
    "legend.fontsize":  8,
    "figure.facecolor": BG,
    "lines.linewidth":  1.8,
})

def glow(ax, x, y, color, lw=2.5, alpha=0.08, layers=4):
    for i in range(layers, 0, -1):
        ax.plot(x, y, color=color, linewidth=lw + i * 2.5,
                alpha=alpha / i, zorder=1)

def style_ax(ax, title, ylabel):
    ax.set_title(title, pad=8)
    ax.set_xlabel("Epoch", labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 1 — Training vs. Validation Loss
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig1.suptitle("Signzy YOLOv8 — Training vs. Validation Loss",
              fontsize=14, fontweight="bold", color=TEXT, y=1.01)

for ax, (tc, vc, tr_col, vl_col, label) in zip(axes, [
    ("train/box_loss", "val/box_loss", TRAIN_BOX, VAL_BOX, "Box Loss"),
    ("train/cls_loss", "val/cls_loss", TRAIN_CLS, VAL_CLS, "Class Loss"),
    ("train/dfl_loss", "val/dfl_loss", TRAIN_DFL, VAL_DFL, "DFL Loss"),
]):
    tr, vl = df[tc], df[vl_col]
    glow(ax, epochs, tr, tr_col)
    glow(ax, epochs, vl, vc)
    ax.plot(epochs, tr, color=tr_col, label="Train", zorder=3)
    ax.plot(epochs, vl, color=vc,     label="Val",   zorder=3, linestyle="--")
    ax.fill_between(epochs, tr, vl, alpha=0.07, color=tr_col)
    style_ax(ax, label, "Loss")
    ax.legend(loc="upper right")

fig1.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 2 — mAP50 & mAP50-95
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(9, 4.5))

map50   = df["metrics/mAP50(B)"]
map5095 = df["metrics/mAP50-95(B)"]
p50_ep  = map50.idxmax() + 1;   p50_val  = map50.max()
p95_ep  = map5095.idxmax() + 1; p95_val  = map5095.max()

glow(ax2, epochs, map50,   MAP50_C)
glow(ax2, epochs, map5095, MAP5095_C)
ax2.plot(epochs, map50,   color=MAP50_C,   label=f"mAP50    (peak {p50_val:.4f} @ ep {p50_ep})")
ax2.plot(epochs, map5095, color=MAP5095_C, label=f"mAP50-95 (peak {p95_val:.4f} @ ep {p95_ep})",
         linestyle="--")
ax2.fill_between(epochs, map5095, map50, alpha=0.08, color=MAP50_C)
ax2.axhline(p50_val,  color=MAP50_C,   linewidth=0.7, linestyle=":", alpha=0.6)
ax2.axhline(p95_val,  color=MAP5095_C, linewidth=0.7, linestyle=":", alpha=0.6)
ax2.scatter([p50_ep],  [p50_val],  color=MAP50_C,   s=60, zorder=5)
ax2.scatter([p95_ep],  [p95_val],  color=MAP5095_C, s=60, zorder=5)
style_ax(ax2, "Signzy YOLOv8 — mAP over Epochs", "mAP Score")
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax2.legend(loc="lower right")
fig2.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 3 — Precision vs. Recall
# ══════════════════════════════════════════════════════════════════════════════
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4.5))
fig3.suptitle("Signzy YOLOv8 — Precision & Recall over 50 Epochs",
              fontsize=13, fontweight="bold", color=TEXT, y=1.01)

prec, rec = df["metrics/precision(B)"], df["metrics/recall(B)"]

# Left — time series
glow(ax3a, epochs, prec, PREC_C)
glow(ax3a, epochs, rec,  REC_C)
ax3a.plot(epochs, prec, color=PREC_C, label=f"Precision (final {prec.iloc[-1]:.4f})")
ax3a.plot(epochs, rec,  color=REC_C,  label=f"Recall    (final {rec.iloc[-1]:.4f})",
          linestyle="--")
ax3a.fill_between(epochs, prec, rec, alpha=0.06, color=ACCENT)
style_ax(ax3a, "Precision & Recall over Epochs", "Score")
ax3a.legend(loc="lower right")

# Right — P–R trajectory scatter
sc = ax3b.scatter(rec, prec, c=epochs, cmap="plasma",
                  s=30, zorder=3, edgecolors="none", alpha=0.9)
ax3b.plot(rec, prec, color=MUTED, linewidth=0.8, alpha=0.4, zorder=2)
cbar = fig3.colorbar(sc, ax=ax3b, pad=0.02)
cbar.set_label("Epoch", color=TEXT, fontsize=8)
cbar.ax.yaxis.set_tick_params(color=MUTED)
plt.setp(cbar.ax.get_yticklabels(), color=TEXT, fontsize=7)
ax3b.scatter([rec.iloc[0]],  [prec.iloc[0]],  color="#FFFFFF", s=80,
             zorder=5, label="Epoch 1",  marker="^")
ax3b.scatter([rec.iloc[-1]], [prec.iloc[-1]], color="#FFD700", s=80,
             zorder=5, label="Epoch 50", marker="*")
style_ax(ax3b, "Precision–Recall Trajectory", "Precision")
ax3b.set_xlabel("Recall", labelpad=6)
ax3b.legend(loc="lower left")

fig3.tight_layout()

# ── Show all graphs ────────────────────────────────────────────────────────────
print("✓ All 3 graphs ready — close any window to see the next one.")
plt.show()