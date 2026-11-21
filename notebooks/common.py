"""Shared setup for the datathon notebooks: paths, data loading, chart style."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "Competition Dataset").exists())
DATA = ROOT / "data" / "Competition Dataset"

# ---- Palette: colorblind-validated categorical order (use in this order, never cycle past 8)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW = SERIES[:4]
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # light -> dark, for ordered groups
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUES = LinearSegmentedColormap.from_list(
    "seq_blues", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

thousands = mtick.FuncFormatter(lambda x, _: f"{x:,.0f}")
pct = mtick.PercentFormatter(1)


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "figure.figsize": (11, 4),
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 9,
    })


def hbar(ax, s, fmt="{:,.0f}", color=BLUE, title=None, xlabel=None):
    """Horizontal ranked bar chart with value labels."""
    s = s.dropna().sort_values()
    bars = ax.barh([str(i) for i in s.index], s.values, color=color, height=0.68)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.xaxis.set_major_formatter(thousands)
    for b, v in zip(bars, s.values):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, "  " + fmt.format(v),
                va="center", fontsize=8, color=INK2)
    ax.margins(x=0.15)
    ax.tick_params(axis="y", length=0, labelcolor=INK2)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    return ax


def load(name, columns=None, drop=()):
    """Read a parquet file: strings -> category, decimals/doubles -> float32, dates -> datetime64[ns]."""
    t = pq.read_table(DATA / f"{name}.parquet", columns=columns)
    t = t.select([c for c in t.column_names if c not in drop])
    fields = [pa.field(f.name, pa.float32()) if (pa.types.is_decimal(f.type) or pa.types.is_floating(f.type)) else f
              for f in t.schema]
    df = t.cast(pa.schema(fields)).to_pandas(strings_to_categorical=True, date_as_object=False)
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype("datetime64[ns]")
    return df
