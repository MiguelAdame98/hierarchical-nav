

# If repeated "Action scores" blocks appear, dedup by (action, rank, seq)
# We aggregate only the "rank#.. w=.. seq=[...]" lines under each action.
# ---------------------------------

# ==============================================================
# MiniGrid State-Visitation Heatmap + Policy Arrows (from logs)
# - Counts visits per tile across all sequences in the log
# - Draws black arrow per tile = most frequent forward direction
# - Highlights the very first winning action at the start tile
#   (computed from the first action of every sequence)
# Pure NumPy/Matplotlib. No seaborn, no external deps.
# ==============================================================
# ==============================================================
# MiniGrid visitation heatmap + uniform policy arrows
# - All arrows same size
# - Start cell framed (instead of big arrow)
# - No axes, tight layout, slim side colorbar
# - Winner first action parsed from "Action scores" (fallback to seq counts)
# ==============================================================

import os, re, ast
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -----------------------------
# INPUT
# -----------------------------


# -----------------------------
# CONFIG (tweak freely)
# -----------------------------
LOG_TEXT = r"""[TOP-K FIRST-ACTION VOTING]
  k=10 weight=linear alpha=0.85 include_pruned=True
  #01 S=38.558  first=forward depth=7 seq=['forward', 'left', 'forward', 'forward', 'forward', 'right', 'forward'] pruned=None
        └ new_gain=6.316  B*dist_start=5.000  C*graph=27.242  [graph_mode=rbf_softmin]
  #02 S=36.476  first=forward depth=7 seq=['forward', 'left', 'forward', 'forward', 'forward', 'left', 'left'] pruned=novelty_floor
        └ new_gain=5.234  B*dist_start=4.000  C*graph=27.242  [graph_mode=rbf_softmin]
  #03 S=36.306  first=forward depth=7 seq=['forward', 'left', 'forward', 'forward', 'forward', 'right', 'right'] pruned=novelty_floor
        └ new_gain=5.064  B*dist_start=4.000  C*graph=27.242  [graph_mode=rbf_softmin]
  #04 S=36.281  first=forward depth=7 seq=['forward', 'left', 'forward', 'forward', 'forward', 'left', 'forward'] pruned=None
        └ new_gain=6.433  B*dist_start=3.000  C*graph=26.848  [graph_mode=rbf_softmin]
  #05 S=36.238  first=forward depth=7 seq=['forward', 'left', 'forward', 'forward', 'forward', 'left', 'right'] pruned=novelty_floor
        └ new_gain=4.996  B*dist_start=4.000  C*graph=27.242  [graph_mode=rbf_softmin]
  #06 S=36.235  first=forward depth=6 seq=['forward', 'left', 'forward', 'forward', 'left', 'right'] pruned=novelty_floor
        └ new_gain=5.963  B*dist_start=3.000  C*graph=27.272  [graph_mode=rbf_softmin]
  #07 S=36.090  first=forward depth=7 seq=['forward', 'left', 'forward', 'forward', 'forward', 'right', 'left'] pruned=novelty_floor
        └ new_gain=4.848  B*dist_start=4.000  C*graph=27.242  [graph_mode=rbf_softmin]
  #08 S=36.031  first=left    depth=7 seq=['left', 'forward', 'left', 'forward', 'forward', 'forward', 'forward'] pruned=None
        └ new_gain=4.313  B*dist_start=5.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #09 S=36.026  first=left    depth=7 seq=['left', 'forward', 'forward', 'left', 'forward', 'right', 'forward'] pruned=None
        └ new_gain=5.305  B*dist_start=4.000  C*graph=26.721  [graph_mode=rbf_softmin]
  #10 S=35.527  first=left    depth=7 seq=['left', 'forward', 'forward', 'left', 'forward', 'forward', 'left'] pruned=None
        └ new_gain=4.719  B*dist_start=4.000  C*graph=26.808  [graph_mode=rbf_softmin]
  #11 S=35.509  first=left    depth=7 seq=['left', 'forward', 'forward', 'left', 'forward', 'forward', 'right'] pruned=None
        └ new_gain=4.702  B*dist_start=4.000  C*graph=26.808  [graph_mode=rbf_softmin]
  #12 S=35.422  first=left    depth=7 seq=['left', 'forward', 'forward', 'forward', 'left', 'forward', 'forward'] pruned=None
        └ new_gain=3.701  B*dist_start=5.000  C*graph=26.721  [graph_mode=rbf_softmin]
  #13 S=35.416  first=forward depth=6 seq=['forward', 'left', 'forward', 'forward', 'left', 'left'] pruned=novelty_floor
        └ new_gain=5.144  B*dist_start=3.000  C*graph=27.272  [graph_mode=rbf_softmin]
  #14 S=35.386  first=forward depth=7 seq=['forward', 'left', 'forward', 'forward', 'left', 'forward', 'left'] pruned=None
        └ new_gain=6.423  B*dist_start=2.000  C*graph=26.963  [graph_mode=rbf_softmin]
  #15 S=35.319  first=forward depth=7 seq=['forward', 'left', 'forward', 'forward', 'left', 'forward', 'forward'] pruned=None
        └ new_gain=5.511  B*dist_start=3.000  C*graph=26.808  [graph_mode=rbf_softmin]
  #16 S=34.713  first=left    depth=7 seq=['left', 'forward', 'left', 'forward', 'forward', 'right', 'forward'] pruned=None
        └ new_gain=3.995  B*dist_start=4.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #17 S=34.711  first=forward depth=7 seq=['forward', 'left', 'forward', 'left', 'forward', 'right', 'forward'] pruned=None
        └ new_gain=5.809  B*dist_start=2.000  C*graph=26.901  [graph_mode=rbf_softmin]
  #18 S=34.675  first=left    depth=7 seq=['left', 'forward', 'forward', 'left', 'forward', 'right', 'right'] pruned=novelty_floor
        └ new_gain=4.867  B*dist_start=3.000  C*graph=26.808  [graph_mode=rbf_softmin]
  #19 S=34.657  first=forward depth=7 seq=['forward', 'left', 'forward', 'left', 'forward', 'forward', 'forward'] pruned=None
        └ new_gain=4.939  B*dist_start=3.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #20 S=34.564  first=forward depth=7 seq=['forward', 'left', 'forward', 'left', 'forward', 'forward', 'right'] pruned=None
        └ new_gain=5.845  B*dist_start=2.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #21 S=34.494  first=left    depth=7 seq=['left', 'forward', 'left', 'forward', 'forward', 'forward', 'right'] pruned=None
        └ new_gain=3.775  B*dist_start=4.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #22 S=34.485  first=forward depth=7 seq=['forward', 'left', 'forward', 'forward', 'left', 'forward', 'right'] pruned=None
        └ new_gain=5.522  B*dist_start=2.000  C*graph=26.963  [graph_mode=rbf_softmin]
  #23 S=34.481  first=forward depth=5 seq=['forward', 'left', 'forward', 'forward', 'right'] pruned=None
        └ new_gain=4.209  B*dist_start=3.000  C*graph=27.272  [graph_mode=rbf_softmin]
  #24 S=34.411  first=forward depth=7 seq=['forward', 'left', 'forward', 'left', 'forward', 'forward', 'left'] pruned=None
        └ new_gain=5.692  B*dist_start=2.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #25 S=34.315  first=forward depth=7 seq=['forward', 'left', 'forward', 'left', 'forward', 'right', 'right'] pruned=novelty_floor
        └ new_gain=6.414  B*dist_start=1.000  C*graph=26.901  [graph_mode=rbf_softmin]
  #26 S=34.247  first=forward depth=7 seq=['forward', 'left', 'forward', 'left', 'forward', 'right', 'left'] pruned=novelty_floor
        └ new_gain=6.346  B*dist_start=1.000  C*graph=26.901  [graph_mode=rbf_softmin]
  #27 S=34.246  first=left    depth=7 seq=['left', 'forward', 'left', 'forward', 'forward', 'forward', 'left'] pruned=None
        └ new_gain=3.527  B*dist_start=4.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #28 S=34.039  first=left    depth=7 seq=['left', 'forward', 'forward', 'left', 'forward', 'left', 'left'] pruned=novelty_floor
        └ new_gain=4.231  B*dist_start=3.000  C*graph=26.808  [graph_mode=rbf_softmin]
  #29 S=33.975  first=forward depth=5 seq=['forward', 'left', 'forward', 'left', 'right'] pruned=novelty_floor
        └ new_gain=4.703  B*dist_start=2.000  C*graph=27.272  [graph_mode=rbf_softmin]
  #30 S=33.833  first=left    depth=7 seq=['left', 'forward', 'forward', 'left', 'forward', 'left', 'right'] pruned=novelty_floor
        └ new_gain=4.025  B*dist_start=3.000  C*graph=26.808  [graph_mode=rbf_softmin]
  #31 S=33.711  first=left    depth=7 seq=['left', 'forward', 'forward', 'forward', 'left', 'forward', 'left'] pruned=None
        └ new_gain=2.990  B*dist_start=4.000  C*graph=26.721  [graph_mode=rbf_softmin]
  #32 S=33.701  first=left    depth=7 seq=['left', 'forward', 'forward', 'left', 'forward', 'right', 'left'] pruned=novelty_floor
        └ new_gain=3.893  B*dist_start=3.000  C*graph=26.808  [graph_mode=rbf_softmin]
  #33 S=33.552  first=left    depth=7 seq=['left', 'forward', 'forward', 'forward', 'left', 'forward', 'right'] pruned=None
        └ new_gain=2.831  B*dist_start=4.000  C*graph=26.721  [graph_mode=rbf_softmin]
  #34 S=33.327  first=left    depth=6 seq=['left', 'forward', 'forward', 'forward', 'right', 'forward'] pruned=None
        └ new_gain=2.479  B*dist_start=4.000  C*graph=26.848  [graph_mode=rbf_softmin]
  #35 S=33.162  first=left    depth=6 seq=['left', 'forward', 'left', 'forward', 'right', 'forward'] pruned=None
        └ new_gain=3.443  B*dist_start=3.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #36 S=33.127  first=forward depth=7 seq=['forward', 'left', 'forward', 'left', 'forward', 'left', 'left'] pruned=None
        └ new_gain=5.225  B*dist_start=1.000  C*graph=26.901  [graph_mode=rbf_softmin]
  #37 S=33.106  first=forward depth=5 seq=['forward', 'left', 'forward', 'left', 'left'] pruned=novelty_floor
        └ new_gain=3.834  B*dist_start=2.000  C*graph=27.272  [graph_mode=rbf_softmin]
  #38 S=32.768  first=forward depth=7 seq=['forward', 'left', 'forward', 'left', 'forward', 'left', 'forward'] pruned=novelty_floor
        └ new_gain=5.918  B*dist_start=0.000  C*graph=26.849  [graph_mode=rbf_softmin]
  #39 S=32.767  first=left    depth=7 seq=['left', 'forward', 'left', 'forward', 'forward', 'right', 'right'] pruned=novelty_floor
        └ new_gain=3.049  B*dist_start=3.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #40 S=32.756  first=forward depth=7 seq=['forward', 'left', 'forward', 'left', 'forward', 'left', 'right'] pruned=novelty_floor
        └ new_gain=4.855  B*dist_start=1.000  C*graph=26.901  [graph_mode=rbf_softmin]
  #41 S=32.396  first=left    depth=6 seq=['left', 'forward', 'left', 'forward', 'right', 'right'] pruned=novelty_floor
        └ new_gain=3.678  B*dist_start=2.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #42 S=32.268  first=forward depth=4 seq=['forward', 'left', 'forward', 'right'] pruned=None
        └ new_gain=2.996  B*dist_start=2.000  C*graph=27.272  [graph_mode=rbf_softmin]
  #43 S=32.246  first=left    depth=7 seq=['left', 'forward', 'left', 'forward', 'forward', 'right', 'left'] pruned=novelty_floor
        └ new_gain=2.528  B*dist_start=3.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #44 S=32.229  first=left    depth=7 seq=['left', 'forward', 'forward', 'left', 'forward', 'left', 'forward'] pruned=None
        └ new_gain=3.511  B*dist_start=2.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #45 S=32.215  first=left    depth=6 seq=['left', 'forward', 'forward', 'forward', 'left', 'left'] pruned=novelty_floor
        └ new_gain=2.367  B*dist_start=3.000  C*graph=26.848  [graph_mode=rbf_softmin]
  #46 S=32.085  first=left    depth=7 seq=['left', 'forward', 'left', 'forward', 'forward', 'left', 'forward'] pruned=None
        └ new_gain=3.367  B*dist_start=2.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #47 S=31.998  first=left    depth=7 seq=['left', 'forward', 'left', 'forward', 'forward', 'left', 'right'] pruned=novelty_floor
        └ new_gain=2.280  B*dist_start=3.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #48 S=31.765  first=left    depth=7 seq=['left', 'forward', 'left', 'forward', 'forward', 'left', 'left'] pruned=novelty_floor
        └ new_gain=2.047  B*dist_start=3.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #49 S=31.488  first=left    depth=5 seq=['left', 'forward', 'forward', 'right', 'forward'] pruned=None
        └ new_gain=1.639  B*dist_start=3.000  C*graph=26.849  [graph_mode=rbf_softmin]
  #50 S=31.352  first=left    depth=6 seq=['left', 'forward', 'left', 'forward', 'right', 'left'] pruned=novelty_floor
        └ new_gain=2.633  B*dist_start=2.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #51 S=31.289  first=left    depth=5 seq=['left', 'forward', 'left', 'forward', 'left'] pruned=None
        └ new_gain=2.571  B*dist_start=2.000  C*graph=26.719  [graph_mode=rbf_softmin]
  #52 S=31.219  first=left    depth=6 seq=['left', 'forward', 'forward', 'forward', 'right', 'left'] pruned=novelty_floor
        └ new_gain=1.371  B*dist_start=3.000  C*graph=26.848  [graph_mode=rbf_softmin]
  #53 S=31.101  first=left    depth=6 seq=['left', 'forward', 'forward', 'forward', 'right', 'right'] pruned=novelty_floor
        └ new_gain=1.253  B*dist_start=3.000  C*graph=26.848  [graph_mode=rbf_softmin]
  #54 S=31.069  first=left    depth=5 seq=['left', 'forward', 'forward', 'right', 'left'] pruned=novelty_floor
        └ new_gain=2.220  B*dist_start=2.000  C*graph=26.849  [graph_mode=rbf_softmin]
  #55 S=31.060  first=left    depth=6 seq=['left', 'forward', 'forward', 'forward', 'left', 'right'] pruned=novelty_floor
        └ new_gain=1.212  B*dist_start=3.000  C*graph=26.848  [graph_mode=rbf_softmin]
  #56 S=31.021  first=left    depth=5 seq=['left', 'forward', 'forward', 'left', 'left'] pruned=novelty_floor
        └ new_gain=2.171  B*dist_start=2.000  C*graph=26.849  [graph_mode=rbf_softmin]
  #57 S=30.763  first=left    depth=4 seq=['left', 'forward', 'right', 'forward'] pruned=None
        └ new_gain=1.914  B*dist_start=2.000  C*graph=26.849  [graph_mode=rbf_softmin]
  #58 S=30.649  first=left    depth=5 seq=['left', 'forward', 'forward', 'left', 'right'] pruned=novelty_floor
        └ new_gain=1.800  B*dist_start=2.000  C*graph=26.849  [graph_mode=rbf_softmin]
  #59 S=30.376  first=left    depth=4 seq=['left', 'forward', 'right', 'left'] pruned=novelty_floor
        └ new_gain=2.527  B*dist_start=1.000  C*graph=26.849  [graph_mode=rbf_softmin]
  #60 S=30.303  first=forward depth=3 seq=['forward', 'left', 'left'] pruned=novelty_floor
        └ new_gain=2.031  B*dist_start=1.000  C*graph=27.272  [graph_mode=rbf_softmin]
  #61 S=30.231  first=left    depth=4 seq=['left', 'forward', 'left', 'right'] pruned=novelty_floor
        └ new_gain=2.381  B*dist_start=1.000  C*graph=26.849  [graph_mode=rbf_softmin]
  #62 S=30.146  first=left    depth=5 seq=['left', 'forward', 'forward', 'right', 'right'] pruned=novelty_floor
        └ new_gain=1.297  B*dist_start=2.000  C*graph=26.849  [graph_mode=rbf_softmin]
  #63 S=30.027  first=left    depth=4 seq=['left', 'forward', 'right', 'right'] pruned=novelty_floor
        └ new_gain=2.177  B*dist_start=1.000  C*graph=26.849  [graph_mode=rbf_softmin]
  #64 S=29.876  first=left    depth=4 seq=['left', 'forward', 'left', 'left'] pruned=novelty_floor
        └ new_gain=2.027  B*dist_start=1.000  C*graph=26.849  [graph_mode=rbf_softmin]
  #65 S=29.764  first=forward depth=2 seq=['forward', 'right'] pruned=None
        └ new_gain=1.492  B*dist_start=1.000  C*graph=27.272  [graph_mode=rbf_softmin]
  #66 S=29.546  first=forward depth=3 seq=['forward', 'left', 'right'] pruned=novelty_floor
        └ new_gain=1.274  B*dist_start=1.000  C*graph=27.272  [graph_mode=rbf_softmin]
  #67 S=28.040  first=left    depth=2 seq=['left', 'right'] pruned=novelty_floor
        └ new_gain=1.191  B*dist_start=0.000  C*graph=26.849  [graph_mode=rbf_softmin]
  #68 S=27.627  first=left    depth=2 seq=['left', 'left'] pruned=novelty_floor
        └ new_gain=0.778  B*dist_start=0.000  C*graph=26.849  [graph_mode=rbf_softmin]
  #69 S=27.120  first=right   depth=1 seq=['right'] pruned=None
        └ new_gain=0.271  B*dist_start=0.000  C*graph=26.849  [graph_mode=rbf_softmin]
  #70 S=-inf  first=left    depth=7 seq=['left', 'forward', 'forward', 'left', 'forward', 'forward', 'forward'] pruned=wall_ahead
        └ new_gain=-inf  B*dist_start=5.000  C*graph=26.808  [graph_mode=rbf_softmin]
  #71 S=-inf  first=forward depth=6 seq=['forward', 'left', 'forward', 'forward', 'forward', 'forward'] pruned=wall_ahead
        └ new_gain=-inf  B*dist_start=5.000  C*graph=27.149  [graph_mode=rbf_softmin]
  #72 S=-inf  first=left    depth=5 seq=['left', 'forward', 'forward', 'forward', 'forward'] pruned=wall_ahead
        └ new_gain=-inf  B*dist_start=4.000  C*graph=26.749  [graph_mode=rbf_softmin]
  #73 S=-inf  first=forward depth=2 seq=['forward', 'forward'] pruned=wall_ahead
        └ new_gain=-inf  B*dist_start=2.000  C*graph=27.272  [graph_mode=rbf_softmin]
  Action scores:
    left    → 6.000  via 3 hits
       - rank#08 w=3.000 S=36.031 seq=['left', 'forward', 'left', 'forward', 'forward', 'forward', 'forward']
       - rank#09 w=2.000 S=36.026 seq=['left', 'forward', 'forward', 'left', 'forward', 'right', 'forward']
       - rank#10 w=1.000 S=35.527 seq=['left', 'forward', 'forward', 'left', 'forward', 'forward', 'left']
    right   → 0.000  via 0 hits
    forward → 49.000  via 7 hits
       - rank#01 w=10.000 S=38.558 seq=['forward', 'left', 'forward', 'forward', 'forward', 'right', 'forward']
       - rank#02 w=9.000 S=36.476 seq=['forward', 'left', 'forward', 'forward', 'forward', 'left', 'left']
       - rank#03 w=8.000 S=36.306 seq=['forward', 'left', 'forward', 'forward', 'forward', 'right', 'right']
       - rank#04 w=7.000 S=36.281 seq=['forward', 'left', 'forward', 'forward', 'forward', 'left', 'forward']
       - rank#05 w=6.000 S=36.238 seq=['forward', 'left', 'forward', 'forward', 'forward', 'left', 'right']
       - rank#06 w=5.000 S=36.235 seq=['forward', 'left', 'forward', 'forward', 'left', 'right']
       - rank#07 w=4.000 S=36.090 seq=['forward', 'left', 'forward', 'forward', 'forward', 'right', 'left']
"""


OUTDIR      = "figs_minigrid"
SAVE_FIGS   = True
DPI         = 300

# Canvas/grid
PADDING_STEPS = 2      # margin around agent trajectories
START_DIR     = 0      # 0=N, 1=E, 2=S, 3=W
START_POS     = (0, 0) # relative coords; we auto-crop around paths

# Visual style
CMAP_NAME  = "Reds"    # white->red; no yellow background
ARROW_LEN  = 0.55      # uniform arrow length (cells)
ARROW_W    = 0.012     # shaft width
HEAD_W     = 6         # head width in points
HEAD_L     = 7         # head length in points
ARROW_ALPHA = 0.95

USE_TEX    = False

# -----------------------------
# Theme
# -----------------------------
def set_paper_theme():
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": USE_TEX,
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

# -----------------------------
# Parsing
# -----------------------------
SEQ_RE = re.compile(r"seq=\[([^\]]+)\]")
SCORES_RE = re.compile(r"Action scores:\s*(.*?)\Z", re.S)

def parse_action_sequences(text):
    """Return list[list[str]] and counts of first actions from seq blocks."""
    seqs, first_counts = [], {"forward": 0, "left": 0, "right": 0}
    for m in SEQ_RE.finditer(text):
        raw = "[" + m.group(1) + "]"
        try:
            actions = ast.literal_eval(raw)
        except Exception:
            actions = [s.strip().strip("'\"") for s in m.group(1).split(",")]
        actions = [a.strip().lower() for a in actions if isinstance(a, str) and a.strip()]
        if not actions:
            continue
        seqs.append(actions)
        a0 = actions[0]
        if a0 in first_counts:
            first_counts[a0] += 1
    return seqs, first_counts

def parse_action_scores_winner(text):
    """
    Parse the 'Action scores:' block and return the winner ('forward','left','right')
    and a dict of numeric scores. Returns (winner, scores_dict) or (None, {}).
    """
    # Grab the last "Action scores:" block if multiple
    blocks = list(re.finditer(r"Action scores:\s*(.+?)(?:\n\s*\n|\Z)", text, re.S))
    if not blocks:
        return None, {}
    block = blocks[-1].group(1)
    scores = {}
    for act in ("left", "right", "forward"):
        m = re.search(rf"{act}\s*→\s*([-\d\.]+)", block)
        if m:
            try:
                scores[act] = float(m.group(1))
            except Exception:
                pass
    if not scores:
        return None, {}
    winner = max(scores.items(), key=lambda kv: kv[1])[0]
    return winner, scores

# -----------------------------
# MiniGrid-ish simulator
# -----------------------------
DIRS = np.array([
    (0, 1),   # 0: North
    (1, 0),   # 1: East
    (0, -1),  # 2: South
    (-1, 0),  # 3: West
], dtype=int)

def simulate(seqs, start_pos=(0,0), start_dir=0):
    """
    Track:
      - visits[(x,y)] = count
      - fdir[(x,y)]   = [nN, nE, nS, nW] for 'forward' taken from that tile
    """
    visits, fdir = {}, {}
    def add_visit(p): visits[p] = visits.get(p, 0) + 1
    def add_fwd(p, d):
        arr = fdir.get(p)
        if arr is None:
            arr = np.zeros(4, dtype=int); fdir[p] = arr
        arr[d] += 1

    for actions in seqs:
        x, y = start_pos
        d = start_dir
        add_visit((x, y))
        for a in actions:
            a = a.lower()
            if a == "left":
                d = (d - 1) % 4
                add_visit((x, y))
            elif a == "right":
                d = (d + 1) % 4
                add_visit((x, y))
            elif a == "forward":
                add_fwd((x, y), d)
                dx, dy = DIRS[d]
                x, y = x + int(dx), y + int(dy)
                add_visit((x, y))
            else:
                add_visit((x, y))
    return visits, fdir

# -----------------------------
# Dense arrays
# -----------------------------
def densify(visits, fdir, pad=PADDING_STEPS):
    xs = [p[0] for p in visits]; ys = [p[1] for p in visits]
    xmin, xmax = min(xs)-pad, max(xs)+pad
    ymin, ymax = min(ys)-pad, max(ys)+pad
    W, H = xmax - xmin + 1, ymax - ymin + 1

    heat = np.zeros((H, W), dtype=float)
    dirs = np.zeros((H, W, 2), dtype=float)

    for (x, y), c in visits.items():
        i, j = y - ymin, x - xmin
        heat[i, j] += float(c)

    for (x, y), counts in fdir.items():
        i, j = y - ymin, x - xmin
        if counts.sum() == 0: continue
        d = int(np.argmax(counts))
        dx, dy = DIRS[d]
        dirs[i, j, 0], dirs[i, j, 1] = dx, dy

    start_ij = (-ymin, -xmin)
    extent = (xmin - 0.5, xmax + 0.5, ymin - 0.5, ymax + 0.5)
    return heat, dirs, start_ij, extent

# -----------------------------
# Winner first action (direction only used for consistency; we FRAME the start)
# -----------------------------
def winner_first_action(text, first_counts, start_dir=START_DIR):
    winner, scores = parse_action_scores_winner(text)
    if winner is None:
        # fallback to sequence first-action counts
        acts = ["forward", "left", "right"]
        winner = acts[int(np.argmax([first_counts.get(a, 0) for a in acts]))]
    if winner == "forward": d = start_dir
    elif winner == "left":  d = (start_dir - 1) % 4
    else:                   d = (start_dir + 1) % 4
    return winner, d

# -----------------------------
# Plot
# -----------------------------
def plot_heatmap(heat, dirs, start_ij, extent, first_act):
    H, W = heat.shape

    # Size proportional to grid with tight bounds (no wasted space)
    cell = 0.4
    fig_w = max(4.0, min(8.0, W * cell + 0.6))  # room for colorbar
    fig_h = max(4.0, min(8.0, H * cell))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Colormap: white for zero, red for high
    cmap = plt.get_cmap(CMAP_NAME).copy()
    cmap.set_under("#ffffff")
    heat_ma = np.ma.masked_where(heat <= 0, heat)

    im = ax.imshow(
        heat_ma, origin="lower", cmap=cmap, interpolation="nearest",
        extent=extent, aspect="equal", vmin=1  # <=0 rendered as 'under' (white)
    )

    # Uniform-length arrows
    y_idx, x_idx = np.nonzero(np.any(dirs != 0.0, axis=2))
    if len(x_idx) > 0:
        U = dirs[y_idx, x_idx, 0]; V = dirs[y_idx, x_idx, 1]
        # normalize to unit then scale to ARROW_LEN (exact same size everywhere)
        norms = np.maximum(np.sqrt(U*U + V*V), 1e-9)
        U = (U / norms) * ARROW_LEN
        V = (V / norms) * ARROW_LEN
        X = x_idx + extent[0] + 0.5
        Y = y_idx + extent[2] + 0.5
        ax.quiver(
            X, Y, U, V,
            angles="xy", scale_units="xy", scale=1.0,
            width=ARROW_W, headwidth=HEAD_W, headlength=HEAD_L, headaxislength=HEAD_L,
            pivot="middle", color="black", alpha=ARROW_ALPHA, minlength=0
        )

    # Frame the start cell (no oversized arrow)
    si, sj = start_ij
    sx = sj + extent[0] + 0.5
    sy = si + extent[2] + 0.5
    start_box = Rectangle((sx-0.5, sy-0.5), 1.0, 1.0, fill=False,
                          edgecolor="black", linewidth=2.2, zorder=4)
    ax.add_patch(start_box)

    # No axes/ticks/grid – keep it tight
    ax.set_axis_off()

    # Slim colorbar at the right
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.02)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("state visitation", rotation=90, labelpad=8)

    fig.subplots_adjust(left=0.01, right=0.97, top=0.99, bottom=0.01)

    if SAVE_FIGS:
        os.makedirs(OUTDIR, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(os.path.join(OUTDIR, f"minigrid_heatmap.{ext}"))

    plt.show()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    set_paper_theme()

    # 1) Parse sequences and action-scores winner
    seqs, first_counts = parse_action_sequences(LOG_TEXT)
    if not seqs:
        raise RuntimeError("No seq=[...] found in LOG_TEXT.")
    first_act, _ = winner_first_action(LOG_TEXT, first_counts, start_dir=START_DIR)
    # (FYI: if your 'Action scores' say forward wins, this picks 'forward'.)

    # 2) Simulate
    visits, fdir = simulate(seqs, start_pos=START_POS, start_dir=START_DIR)

    # 3) Densify
    heat, dirs, start_ij, extent = densify(visits, fdir, pad=PADDING_STEPS)

    # 4) Plot
    plot_heatmap(heat, dirs, start_ij, extent, first_act)
