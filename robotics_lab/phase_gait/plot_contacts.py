"""Plot measured and desired foot contacts from evaluate.py JSON artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("traces", type=Path, nargs="+")
args = parser.parse_args()

fig, axes = plt.subplots(len(args.traces), 1, figsize=(10, 2.1 * len(args.traces)), sharex=True, squeeze=False)
for ax, path in zip(axes[:, 0], args.traces):
    data = json.loads(path.read_text())
    sample_count = min(len(data["rows"]), round(2.5 / data["dt"]))
    rows = np.asarray(data["rows"][:sample_count], dtype=bool)
    t = np.arange(sample_count + 1) * data["dt"]
    for foot, name in enumerate(data["feet"]):
        for column, y_offset, color in ((foot, -0.25, "#c1c9d5"), (foot + 4, 0.05, "#176b62")):
            mask = rows[:, column] & rows[:, 8]
            edges = np.diff(np.r_[False, mask, False].astype(int))
            starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            ax.broken_barh(list(zip(t[starts], t[stops] - t[starts])), (foot + y_offset, 0.20), color=color)
    ax.set_yticks(range(4), [name.replace("_foot", "") for name in data["feet"]])
    ax.set_ylim(3.5, -0.5)
    ax.set_title(path.parent.name, loc="left", fontsize=11)
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
axes[-1, 0].set_xlabel("Seconds after warmup")
axes[-1, 0].set_xlim(0, 2.5)
fig.legend(handles=[Patch(color="#c1c9d5", label="Target stance"), Patch(color="#176b62", label="Measured contact")],
           loc="upper right", bbox_to_anchor=(0.98, 1.0), frameon=False, ncol=2)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(args.output, dpi=160, bbox_inches="tight")
