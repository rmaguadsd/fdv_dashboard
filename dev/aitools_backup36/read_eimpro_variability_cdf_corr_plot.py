#!/usr/bin/env python
r"""
Generate a variability plot of RBER vs WL from a CSV, color-coded by pagetype,
split (faceted) by readtype and dut.

- Tries to use seaborn FacetGrid for clean faceting.
- Falls back to pure matplotlib if seaborn is unavailable.
- Attempts to auto-detect column names with common aliases (case-insensitive):
  WL: wl, wordline, word_line, page
  RBER: rber, raw_ber, rawber, ber, error_rate, ber_raw
  pagetype: pagetype, ptype, page_type, pagemaptype, maptype, deck, type
  readtype: readtype, rtype, read_type, readmode, read_mode
  dut: dut, device, unit, chip, die, die_id, dut_id

Usage (PowerShell):
  python tools/scripts/read_eimpro_plot.py -i C:/Users/rmaguad/Documents/Work/logs/read_eimpro/read_v_eimpro.csv

Output:
  Saves a PNG next to the input (or to --output path) and also shows the plot.
"""

from __future__ import print_function, unicode_literals
import argparse
import os
import sys
import textwrap

# Optional deps
try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    print("error: pandas is required. Please install with 'pip install pandas'", file=sys.stderr)
    raise

try:  # optional
    import seaborn as sns
    _HAVE_SEABORN = True
except Exception:
    _HAVE_SEABORN = False

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import LogLocator
from matplotlib import transforms as mtransforms
import numpy as np


ALIASES = {
    "WL": ["wl", "wordline", "word_line", "page"],
    "RBER": ["rber", "raw_ber", "rawber", "ber", "error_rate", "ber_raw"],
    "pagetype": ["pagetype", "ptype", "page_type", "pagemaptype", "maptype", "type"],
    "readtype": ["readtype", "rtype", "read_type", "readmode", "read_mode"],
    "dut": ["dut", "device", "unit", "chip", "die", "die_id", "dut_id"],
    # Optional grouping/label for title context
    "parttype": ["parttype", "part_type", "ptype_group"],
    # Optional: if present, will use to define deck ranges more precisely
    "deck": ["deck"],
}

# Only these are required; others in ALIASES are optional
REQUIRED_KEYS = ["WL", "RBER", "pagetype", "readtype", "dut"]


def find_columns(df):
    cols_lower = dict((c.lower(), c) for c in df.columns)
    resolved = {}
    for target, names in ALIASES.items():
        for name in names:
            if name in cols_lower:
                resolved[target] = cols_lower[name]
                break
    # Validate only required keys
    missing = [k for k in REQUIRED_KEYS if k not in resolved]
    if missing:
        alias_strs = []
        for k, v in ALIASES.items():
            alias_strs.append("{0}: {1}".format(k, ", ".join(v)))
        msg = (
            "Missing required columns: " + ", ".join(missing) + "\n" +
            "Available columns: " + ", ".join(list(df.columns)) + "\n" +
            "Accepted aliases: " + "; ".join(alias_strs)
        )
        raise ValueError(msg)
    return resolved


def load_data(path):
    # Robust CSV read: try fast C engine first, then fall back to Python engine with bad-line skipping
    try:
        df = pd.read_csv(path)
    except Exception:
        # Fallback: autodetect separator, skip malformed lines
        try:
            df = pd.read_csv(path, engine='python', sep=None, on_bad_lines='skip')
        except TypeError:
            df = pd.read_csv(path, engine='python', sep=None, error_bad_lines=False)

    if df.empty:
        raise ValueError("CSV is empty: " + path)

    # Trim whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    cols = find_columns(df)

    # Standardize expected column names for downstream code
    rename_map = {
        cols["WL"]: "WL",
        cols["RBER"]: "RBER",
        cols["pagetype"]: "pagetype",
        cols["readtype"]: "readtype",
        cols["dut"]: "dut",
    }
    if "parttype" in cols:
        rename_map[cols["parttype"]] = "parttype"
    if "deck" in cols:
        rename_map[cols["deck"]] = "deck"
    df = df.rename(columns=rename_map)

    # Coerce types
    df["WL"] = pd.to_numeric(df["WL"], errors="coerce")
    df["RBER"] = pd.to_numeric(df["RBER"], errors="coerce")

    # Drop rows with missing core fields
    df = df.dropna(subset=["WL", "RBER", "pagetype", "readtype", "dut"]).copy()

    # Remove non-positive RBER for log scaling
    df = df[df["RBER"] > 0].copy()

    # Sort by WL for nicer plots
    df = df.sort_values(["dut", "readtype", "WL"])

    # Normalize categorical types to strings
    for c in ["pagetype", "readtype", "dut", "parttype"]:
        df[c] = df[c].astype(str)

    # If a deck column exists, normalize common names to LD/MD/UP when possible
    if "deck" in df.columns:
        def _norm_deck(x):
            s = str(x).strip().upper()
            if s.startswith("L"):  # lower
                return "LD"
            if s.startswith("M"):  # middle
                return "MD"
            if s.startswith("U"):  # upper
                return "UP"
            return s
        try:
            df["deck"] = df["deck"].apply(_norm_deck)
        except Exception:
            pass

    return df


def _facet_axes_iter(g):
    # Helper to iterate axes robustly across seaborn versions
    axes = getattr(g, 'axes', None)
    if axes is None:
        ax = getattr(g, 'ax', None)
        return [ax] if ax is not None else []
    try:
        return [a for a in axes.flatten() if a is not None]
    except Exception:
        out = []
        for row in axes:
            try:
                for a in row:
                    if a is not None:
                        out.append(a)
            except Exception:
                if row is not None:
                    out.append(row)
        return out


def _sanitize_label(val):
    """Return a filename-safe string for a label value."""
    try:
        s = str(val)
    except Exception:
        s = "group"
    for ch in '\\/:*?"<>|\n\r\t':
        s = s.replace(ch, "_")
    s = s.strip()
    return s or "group"


def _ensure_min_log_ticks(ax, min_decades=2.0):
    """Ensure the y-axis on a log scale spans at least `min_decades` decades.
    Expands the upper limit if necessary to create at least two major log ticks.
    """
    try:
        y0, y1 = ax.get_ylim()
        eps = 1e-12
        y0 = max(y0, eps)
        y1 = max(y1, y0 * 1.01)
        lo, hi = np.log10(y0), np.log10(y1)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return
        span = hi - lo
        if span < float(min_decades):
            new_hi = lo + float(min_decades)
            ax.set_ylim(10.0 ** lo, 10.0 ** new_hi)
        # Ensure major ticks at powers of 10
        try:
            ax.yaxis.set_major_locator(LogLocator(base=10.0))
        except Exception:
            pass
    except Exception:
        pass


def plot_with_seaborn(df, title):
    # Smaller fonts for axes and legend
    sns.set_context(
        "notebook",
        rc={
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        },
    )
    sns.set_style("whitegrid")
    g = sns.FacetGrid(
        df, row="readtype", col="dut", hue="pagetype",
        sharex=True, sharey=True, margin_titles=True, despine=False, height=3.2, aspect=1.4
    )
    try:
        g.map_dataframe(sns.scatterplot, x="WL", y="RBER", s=18, alpha=0.7)
    except Exception:
        g.map(plt.scatter, "WL", "RBER", s=18, alpha=0.7)

    # Log scale, labels, and deck annotations per DUT facet
    axes = getattr(g, 'axes', None)
    if axes is None:
        axes = [[getattr(g, 'ax', None)]]
    for r_idx, row_axes in enumerate(axes):
        if row_axes is None:
            continue
        # Ensure list-like
        try:
            row_list = list(row_axes)
        except Exception:
            row_list = [row_axes]
        for c_idx, ax in enumerate(row_list):
            if ax is None:
                continue
            ax.set_yscale('log')
            ax.set_xlabel("WL", fontsize=9)
            ax.set_ylabel("RBER (log)", fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            # Determine the DUT for this column index and annotate decks
            try:
                dut_key = g.col_names[c_idx]
            except Exception:
                dut_key = None
            if dut_key is not None:
                try:
                    df_dut = df[df["dut"] == str(dut_key)]
                    if not df_dut.empty:
                        _annotate_decks(ax, df_dut)
                except Exception:
                    pass
            _ensure_min_log_ticks(ax, min_decades=2.0)

    # Legend with title (fallback for older seaborn), positioned inside (front of the plot)
    try:
        g.add_legend(title="pagetype")
    except TypeError:
        g.add_legend()
        try:
            if g._legend is not None:
                g._legend.set_title('pagetype')
        except Exception:
            pass
    try:
        if getattr(g, '_legend', None) is not None:
            try:
                # Place legend inside at upper-right
                g._legend.set_bbox_to_anchor((0.98, 0.98))
                g._legend._loc = 1  # upper right
                g._legend.get_frame().set_alpha(0.85)
                # Smaller legend fonts
                try:
                    g._legend.get_title().set_fontsize(9)
                except Exception:
                    pass
                try:
                    for txt in g._legend.get_texts():
                        txt.set_fontsize(8)
                except Exception:
                    pass
            except Exception:
                pass
            # No extra right margin since legend is inside
            g.fig.subplots_adjust(top=0.9)
        else:
            g.fig.subplots_adjust(top=0.9)
    except Exception:
        g.fig.subplots_adjust(top=0.9)
    # Build title with optional parttype context
    try:
        if "parttype" in df.columns:
            pts = sorted(set(df["parttype"].astype(str)))
            if len(pts) == 1:
                title = f"{title} — parttype {pts[0]}"
            elif 1 < len(pts) <= 4:
                title = f"{title} — parttypes {', '.join(pts)}"
            else:
                title = f"{title} — {len(pts)} parttypes"
    except Exception:
        pass
    g.fig.suptitle(title)
    return g.fig


def plot_with_matplotlib(df, title):
    # Facet by readtype (rows) and dut (cols)
    readtypes = sorted(df["readtype"].unique())
    duts = sorted(df["dut"].unique())
    nrows, ncols = max(1, len(readtypes)), max(1, len(duts))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 3.5*nrows), squeeze=False)
    try:
        cmap = plt.get_cmap('tab10')
    except Exception:
        cmap = plt.get_cmap('Set1')

    for r_idx, rtype in enumerate(readtypes):
        for c_idx, dut in enumerate(duts):
            ax = axes[r_idx][c_idx]
            sub = df[(df["readtype"] == rtype) & (df["dut"] == dut)]
            if sub.empty:
                ax.set_visible(False)
                continue
            # Scatter by pagetype
            for i, (ptype, grp) in enumerate(sub.groupby("pagetype")):
                ax.scatter(grp["WL"], grp["RBER"], s=15, alpha=0.7, color=cmap(i % 10), label=str(ptype))
            ax.set_yscale('log')
            ax.set_xlabel("WL", fontsize=9)
            ax.set_ylabel("RBER (log)", fontsize=9)
            ax.set_title("dut={0}, readtype={1}".format(dut, rtype), fontsize=10)
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, which='both', ls='--', alpha=0.4)
            # Deck bands for this DUT
            try:
                df_dut = df[df["dut"] == dut]
                if not df_dut.empty:
                    _annotate_decks(ax, df_dut)
            except Exception:
                pass
            _ensure_min_log_ticks(ax, min_decades=2.0)
            # No per-axes legend; we'll add a single figure-level legend outside
            
    # Build a consolidated legend from all axes
    labels_map = {}
    for row in axes:
        for ax in row:
            h, l = ax.get_legend_handles_labels()
            for handle, lab in zip(h, l):
                labels_map[lab] = handle
    if labels_map:
        labels_sorted = sorted(labels_map.keys())
        handles_sorted = [labels_map[k] for k in labels_sorted]
        # Place legend inside at upper-right
        leg = fig.legend(handles_sorted, labels_sorted, title='pagetype', loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=8)
        try:
            leg.get_frame().set_alpha(0.85)
            try:
                leg.get_title().set_fontsize(9)
            except Exception:
                pass
        except Exception:
            pass

    # Build title with optional parttype context
    try:
        if "parttype" in df.columns:
            pts = sorted(set(df["parttype"].astype(str)))
            if len(pts) == 1:
                title = f"{title} — parttype {pts[0]}"
            elif 1 < len(pts) <= 4:
                title = f"{title} — parttypes {', '.join(pts)}"
            else:
                title = f"{title} — {len(pts)} parttypes"
    except Exception:
        pass
    fig.suptitle(title)
    # Use full width since legend is inside
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _scatter_by_pagetype(ax, sub):
    """Helper: scatter plot of WL vs RBER colored by pagetype on a given axes."""
    try:
        cmap = plt.get_cmap('tab10')
    except Exception:
        cmap = plt.get_cmap('Set1')
    for i, (ptype, grp) in enumerate(sub.groupby("pagetype")):
        ax.scatter(grp["WL"], grp["RBER"], s=15, alpha=0.7, color=cmap(i % 10), label=str(ptype))
    ax.set_yscale('log')
    ax.set_xlabel("WL", fontsize=9)
    ax.set_ylabel("RBER (log)", fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    _ensure_min_log_ticks(ax, min_decades=2.0)


def _compute_deck_ranges_for_dut(df_dut):
    """
    Compute deck ranges (x0, x1, label) for a given DUT dataframe.
    If a 'deck' column exists, infer ranges from its min/max WL per deck label.
    Else, split WL into three contiguous thirds labeled LD/MD/UP.
    """
    # Prefer explicit deck column if present
    if "deck" in df_dut.columns:
        out = []
        for lab in ["LD", "MD", "UP"]:
            sub = df_dut[df_dut["deck"].astype(str).str.upper() == lab]
            if not sub.empty:
                try:
                    x0 = float(sub["WL"].min())
                    x1 = float(sub["WL"].max())
                    if x1 >= x0:
                        out.append((x0, x1, lab))
                except Exception:
                    continue
        if out:
            return out
    # Fallback: split into 3 contiguous chunks by sorted unique WL
    try:
        arr = np.unique(df_dut["WL"].values.astype(float))
    except Exception:
        return []
    n = int(arr.size)
    if n < 3:
        return []
    i1 = max(1, n // 3)
    i2 = max(i1 + 1, (2 * n) // 3)
    thirds = [arr[:i1], arr[i1:i2], arr[i2:]]
    labels = ["LD", "MD", "UP"]
    out = []
    for seg, lab in zip(thirds, labels):
        if seg.size == 0:
            continue
        out.append((float(seg.min()), float(seg.max()), lab))
    return out


def _sanitize_label(s):
    """Make a string safe for filenames."""
    try:
        s = str(s)
    except Exception:
        s = "group"
    bad = "\\/:*?\"<>|\n\r\t"
    for ch in bad:
        s = s.replace(ch, "_")
    s = s.strip()
    if not s:
        s = "group"
    return s


def _annotate_decks(ax, df_context):
    """Annotate deck spans and labels (LD/MD/UP) for the provided context dataframe on this axes."""
    ranges = _compute_deck_ranges_for_dut(df_context)
    if not ranges:
        return
    y0, y1 = ax.get_ylim()
    # y for top label (near top of axes), and bottom placement along x-axis
    try:
        if ax.get_yscale() == 'log' and y0 > 0 and y1 > 0:
            y_text_top = np.power(10.0, np.log10(y0) + 0.9 * (np.log10(y1) - np.log10(y0)))
        else:
            y_text_top = y0 + 0.9 * (y1 - y0)
    except Exception:
        y_text_top = y1
    bottom_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    colors = [(0.2, 0.2, 0.2, 0.06), (0.1, 0.1, 0.1, 0.09), (0.2, 0.2, 0.2, 0.06)]
    for idx, (x0, x1, lab) in enumerate(ranges):
        ax.axvspan(x0, x1, color=colors[idx % len(colors)], lw=0)
        x_mid = x0 + 0.5 * (x1 - x0)
        # Top label
        try:
            ax.text(x_mid, y_text_top, lab, ha='center', va='center', fontsize=9, color='black')
        except Exception:
            pass
        # Bottom label on x-axis area
        try:
            ax.text(x_mid, 0.02, lab, ha='center', va='bottom', fontsize=8, color='black', transform=bottom_transform)
        except Exception:
            pass


def plot_dut_page(df, dut):
    """
    Build a figure containing subplots for one DUT, split by readtype, color by pagetype.
    Returns the matplotlib.figure.Figure.
    """
    readtypes = sorted(df["readtype"].unique())
    nrows, ncols = max(1, len(readtypes)), 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5, 3.0*nrows), squeeze=False)
    for r_idx, rtype in enumerate(readtypes):
        ax = axes[r_idx][0]
        sub = df[(df["readtype"] == rtype)]
        if sub.empty:
            ax.set_visible(False)
            continue
        _scatter_by_pagetype(ax, sub)
        _annotate_decks(ax, df)  # annotate decks for this DUT
        ax.set_title("dut={0}, readtype={1}".format(dut, rtype), fontsize=10)
    # Figure-level legend placed inside (front)
    labels_map = {}
    for ax in axes.flatten():
        if not ax.get_visible():
            continue
        h, l = ax.get_legend_handles_labels()
        for handle, lab in zip(h, l):
            labels_map[lab] = handle
    if labels_map:
        labels_sorted = sorted(labels_map.keys())
        handles_sorted = [labels_map[k] for k in labels_sorted]
        leg = fig.legend(handles_sorted, labels_sorted, title='pagetype', loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=8)
        try:
            leg.get_frame().set_alpha(0.85)
            leg.get_title().set_fontsize(9)
        except Exception:
            pass
    # Use full width since legend is inside
    fig.tight_layout(rect=(0, 0, 1, 1))
    return fig


def _safe_log10(series):
    """Compute log10 safely for positive values; non-positive filtered earlier."""
    try:
        return np.log10(series.values)
    except Exception:
        # Fallback using pandas
        return np.log10(series.astype(float))


def _rcdf_xy(values):
    """Compute reverse CDF (tail) for a 1D array-like of positive values.
    Returns (x_sorted, tail_prob) where tail_prob[i] = P(X >= x_sorted[i]).
    """
    try:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.array([]), np.array([])
        arr.sort()
        x_unique, counts = np.unique(arr, return_counts=True)
        remaining = counts[::-1].cumsum()[::-1].astype(float)
        tail = remaining / float(arr.size)
        return x_unique, tail
    except Exception:
        return np.array([]), np.array([])


def build_summary_text(df):
    """
    Build a human-readable summary describing commonalities and differences between DUTs and readtypes.
    Returns a string.
    """
    lines = []
    duts = sorted(df["dut"].unique())
    rtypes = sorted(df["readtype"].unique())

    # Common/unique pagetypes
    pagetypes_by_dut = dict((d, set(df[df["dut"] == d]["pagetype"].unique())) for d in duts)
    if duts:
        common_p = set.intersection(*list(pagetypes_by_dut.values())) if len(duts) > 1 else pagetypes_by_dut[duts[0]]
    else:
        common_p = set()
    lines.append("Common pagetypes across DUTs: {0}".format(
        ", ".join(sorted(common_p)) if common_p else "(none)"
    ))
    for d in duts:
        uniq = pagetypes_by_dut[d].difference(set.union(*(pagetypes_by_dut[x] for x in duts if x != d))) if len(duts) > 1 else set()
        if uniq:
            lines.append("Unique pagetypes for {0}: {1}".format(d, ", ".join(sorted(uniq))))

    # Readtypes coverage
    rtypes_by_dut = dict((d, set(df[df["dut"] == d]["readtype"].unique())) for d in duts)
    if duts:
        common_r = set.intersection(*list(rtypes_by_dut.values())) if len(duts) > 1 else rtypes_by_dut[duts[0]]
    else:
        common_r = set()
    lines.append("Common readtypes across DUTs: {0}".format(
        ", ".join(sorted(common_r)) if common_r else "(none)"
    ))

    # Median RBER per (dut, readtype)
    med = df.groupby(["dut", "readtype"])['RBER'].median()
    try:
        med = med.sort_index()
    except Exception:
        pass
    lines.append("Median RBER by readtype per DUT:")
    for d in duts:
        parts = []
        for r in rtypes:
            val = med.get((d, r))
            if val is not None and not (isinstance(val, float) and (np.isnan(val))):
                parts.append("{0}:{1:.2e}".format(r, float(val)))
        if parts:
            lines.append("  {0}: ".format(d) + ", ".join(parts))

    # Trend with WL per (dut, readtype): compute slope of linear fit on (WL, log10(RBER))
    pos_slope = 0
    neg_slope = 0
    zero_slope = 0
    slope_lines = []
    for d in duts:
        for r in rtypes:
            sub = df[(df["dut"] == d) & (df["readtype"] == r)]
            if len(sub) < 3:
                continue
            try:
                x = sub["WL"].values.astype(float)
                y = _safe_log10(sub["RBER"])
                # Guard against identical X
                if np.allclose(np.std(x), 0):
                    continue
                coeffs = np.polyfit(x, y, 1)
                slope = float(coeffs[0])
                if slope > 0:
                    pos_slope += 1
                elif slope < 0:
                    neg_slope += 1
                else:
                    zero_slope += 1
                slope_lines.append("  {0}, {1}: slope={2:.3g}".format(d, r, slope))
            except Exception:
                continue
    if slope_lines:
        lines.append("WL vs log10(RBER) trend (per DUT, readtype):")
        lines.extend(slope_lines[:20])  # keep it concise
        lines.append("Summary of slopes: +:{0}, -:{1}, ~0:{2}".format(pos_slope, neg_slope, zero_slope))

    # Deck anomalies: for each DUT, define decks; within each readtype and deck, report WLs with outlier log10(RBER)
    lines.append("Deck anomalies (per DUT/readtype within LD/MD/UP):")
    max_lines = 30
    count = 0
    for d in duts:
        df_d = df[df["dut"] == d]
        deck_ranges = _compute_deck_ranges_for_dut(df_d)
        if not deck_ranges:
            continue
        for r in rtypes:
            sub = df_d[df_d["readtype"] == r]
            if sub.empty:
                continue
            for (x0, x1, lab) in deck_ranges:
                deck_sub = sub[(sub["WL"] >= x0) & (sub["WL"] <= x1)]
                if len(deck_sub) < 5:
                    continue
                try:
                    logr = _safe_log10(deck_sub["RBER"])
                    med = float(np.median(logr))
                    mad = float(np.median(np.abs(logr - med)))
                    if mad <= 0:
                        mad = float(np.std(logr))
                    thresh = 3.0 * mad
                    dev = np.abs(logr - med)
                    mask = dev > thresh
                    if np.any(mask):
                        wl_list = list(deck_sub.loc[mask, "WL"])[:6]
                        wl_fmt = ", ".join(str(int(w)) if float(w).is_integer() else "{0:.2f}".format(float(w)) for w in wl_list)
                        lines.append("  {0}, {1}, {2}: {3} outlier(s) at WL {4}".format(d, r, lab, int(np.sum(mask)), wl_fmt))
                        count += 1
                        if count >= max_lines:
                            lines.append("  (more anomalies omitted)")
                            break
                except Exception:
                    continue
            if count >= max_lines:
                break
        if count >= max_lines:
            break

    # Build stats table by pagetype and deck, split by readtype
    def _derive_deck_labels(all_df):
        if "deck" in all_df.columns:
            return all_df["deck"].astype(str).str.upper()
        # derive per-DUT
        lab = pd.Series(index=all_df.index, dtype=object)
        for d in sorted(all_df["dut"].unique()):
            sub = all_df[all_df["dut"] == d]
            ranges = _compute_deck_ranges_for_dut(sub)
            for (x0, x1, l) in ranges:
                idx = sub[(sub["WL"] >= x0) & (sub["WL"] <= x1)].index
                lab.loc[idx] = l
        lab = lab.fillna("UNK")
        return lab

    try:
        lines.append("\nStats by pagetype and deck (split by readtype):")
        deck_order = {"LD": 0, "MD": 1, "UP": 2}
        for r in rtypes:
            df_rt = df[df["readtype"] == r]
            if df_rt.empty:
                continue
            deck_series = _derive_deck_labels(df_rt)
            df_stats = df_rt.copy()
            df_stats["_deck"] = deck_series
            rows = []
            for (ptype, dlab), sub in df_stats.groupby(["pagetype", "_deck"]):
                if sub.empty:
                    continue
                try:
                    n = int(len(sub))
                    r = sub["RBER"].values
                    mean = float(np.mean(r))
                    std = float(np.std(r))
                    med = float(np.median(r))
                    p90 = float(np.percentile(r, 90))
                    rmin = float(np.min(r))
                    rmax = float(np.max(r))
                    rows.append((str(ptype), str(dlab), n, mean, std, med, p90, rmin, rmax))
                except Exception:
                    continue
            rows.sort(key=lambda r: (r[0], deck_order.get(r[1], 9)))
            if rows:
                lines.append("  readtype={0}".format(r))
                header = "{:<14}  {:<3}  {:>6}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}".format(
                    "pagetype", "deck", "N", "mean_RBER", "stdev_RBER", "median_RBER", "p90_RBER", "min_RBER", "max_RBER")
                lines.append(header)
                lines.append("-" * len(header))
                for ptype, dlab, n, mean, std, med, p90, rmin, rmax in rows:
                    lines.append(
                        "{:<14}  {:<3}  {:>6}  {:>11.2e}  {:>11.2e}  {:>11.2e}  {:>11.2e}  {:>11.2e}  {:>11.2e}".format(
                            ptype, dlab, n, mean, std, med, p90, rmin, rmax)
                    )
    except Exception:
        # Do not fail the whole summary if table fails
        pass

    # Wrap lines for nicer PDF text
    wrapped = []
    for ln in lines:
        wrapped.extend(textwrap.wrap(ln, width=110))
    return "\n".join(wrapped)


def generate_pdf(df, out_pdf_path):
    """Generate a multi-page PDF: one page per DUT with plots, and a final summary page."""
    duts = sorted(df["dut"].unique())
    with PdfPages(out_pdf_path) as pdf:
        for d in duts:
            sub = df[df["dut"] == d]
            if sub.empty:
                continue
            fig = plot_dut_page(sub, d)
            # Include parttype in title if available
            ttl = "RBER vs WL by pagetype — DUT {0}".format(d)
            try:
                if "parttype" in sub.columns:
                    pts = sorted(set(sub["parttype"].astype(str)))
                    if len(pts) == 1:
                        ttl = f"{ttl} — parttype {pts[0]}"
                    elif 1 < len(pts) <= 4:
                        ttl = f"{ttl} — parttypes {', '.join(pts)}"
                    else:
                        ttl = f"{ttl} — {len(pts)} parttypes"
            except Exception:
                pass
            fig.suptitle(ttl)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)

        # Summary page
        summary = build_summary_text(df)
        fig = plt.figure(figsize=(8.5, 11))
        ttl = "Summary: commonalities and differences (DUTs vs readtypes)"
        try:
            if "parttype" in df.columns:
                pts = sorted(set(df["parttype"].astype(str)))
                if len(pts) == 1:
                    ttl = f"{ttl} — parttype {pts[0]}"
                elif 1 < len(pts) <= 4:
                    ttl = f"{ttl} — parttypes {', '.join(pts)}"
                else:
                    ttl = f"{ttl} — {len(pts)} parttypes"
        except Exception:
            pass
        fig.suptitle(ttl, fontsize=14)
        fig.text(0.06, 0.96, "", fontsize=1)  # spacer
        fig.text(0.06, 0.93, summary, fontsize=10, family='monospace', va='top')
        pdf.savefig(fig)
        plt.close(fig)


def generate_rcdf_pagetype_only_png(df, out_png_path):
    """Reverse CDF of RBER colored by pagetype, faceted by readtype (rows) and dut (columns)."""
    sdf = df.copy()
    try:
        sdf["RBER"] = pd.to_numeric(sdf["RBER"], errors='coerce')
    except Exception:
        pass
    sdf = sdf[np.isfinite(sdf.get("RBER", np.nan))]
    if "readtype" not in sdf.columns:
        sdf["readtype"] = "all"
    if "pagetype" not in sdf.columns:
        sdf["pagetype"] = "all"
    readtypes = sorted(sdf["readtype"].astype(str).unique())
    duts = sorted(sdf["dut"].astype(str).unique()) if "dut" in sdf.columns else ["all"]
    nrows, ncols = max(1, len(readtypes)), max(1, len(duts))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.4*ncols, 3.6*nrows), squeeze=False)
    try:
        cmap = plt.get_cmap('tab10')
    except Exception:
        cmap = plt.get_cmap('Set1')
    for r_idx, rtype in enumerate(readtypes):
        for c_idx, dut in enumerate(duts):
            ax = axes[r_idx][c_idx]
            if "dut" in sdf.columns and dut != "all":
                sub = sdf[(sdf["readtype"].astype(str) == str(rtype)) & (sdf["dut"].astype(str) == str(dut))]
            else:
                sub = sdf[sdf["readtype"].astype(str) == str(rtype)]
            if sub.empty:
                ax.set_visible(False)
                continue
            for i, (ptype, grp) in enumerate(sorted(sub.groupby("pagetype"), key=lambda kv: str(kv[0]))):
                x, y = _rcdf_xy(grp["RBER"].values)
                if x.size == 0:
                    continue
                ax.plot(x, y, label=str(ptype), color=cmap(i % 10), lw=1.5, alpha=0.9)
            ax.set_xscale('log')
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("RBER (log)", fontsize=9)
            ax.set_ylabel("P(RBER ≥ x)", fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            try:
                leg = ax.legend(title='pagetype', loc='upper right', fontsize=8)
                leg.get_title().set_fontsize(9)
                leg.get_frame().set_alpha(0.85)
            except Exception:
                pass
            ax.grid(True, which='both', ls='--', alpha=0.4)
            try:
                if "dut" in sdf.columns and dut != "all":
                    ax.set_title(f"dut={dut}, readtype={rtype}", fontsize=10)
                else:
                    ax.set_title(f"readtype={rtype}", fontsize=10)
            except Exception:
                pass
    ttl = "Reverse CDF of RBER by pagetype"
    try:
        if "parttype" in sdf.columns:
            pts = sorted(set(sdf["parttype"].astype(str)))
            if len(pts) == 1:
                ttl = f"{ttl} — parttype {pts[0]}"
            elif 1 < len(pts) <= 4:
                ttl = f"{ttl} — parttypes {', '.join(pts)}"
            else:
                ttl = f"{ttl} — {len(pts)} parttypes"
    except Exception:
        pass
    fig.suptitle(ttl)
    try:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    except Exception:
        pass
    try:
        fig.savefig(out_png_path, dpi=200)
        print("saved: {0}".format(out_png_path))
    finally:
        plt.close(fig)


def generate_rcdf_pagetype_deck_png(df, out_png_path):
    """Reverse CDF of RBER based on pagetype and deck.
    Facet by pagetype (rows) and dut (columns); lines colored by deck (LD/MD/UP/UNK).
    Derives deck from WL thirds per DUT if deck not present.
    """
    sdf = df.copy()
    try:
        sdf["RBER"] = pd.to_numeric(sdf["RBER"], errors='coerce')
    except Exception:
        pass
    sdf = sdf[np.isfinite(sdf.get("RBER", np.nan))]
    if "pagetype" not in sdf.columns:
        sdf["pagetype"] = "all"
    if "WL" in sdf.columns:
        try:
            sdf["WL"] = pd.to_numeric(sdf["WL"], errors='coerce')
        except Exception:
            pass
    # Determine deck labels
    if "deck" in sdf.columns:
        def _norm_deck(x):
            s = str(x).strip().upper()
            if s.startswith("L"): return "LD"
            if s.startswith("M"): return "MD"
            if s.startswith("U"): return "UP"
            return s or "UNK"
        try:
            sdf["_deck"] = sdf["deck"].apply(_norm_deck)
        except Exception:
            sdf["_deck"] = "UNK"
    else:
        dlabels = pd.Series(index=sdf.index, dtype=object)
        try:
            if "dut" in sdf.columns and "WL" in sdf.columns:
                for d in sorted(sdf["dut"].astype(str).unique()):
                    sub = sdf[sdf["dut"].astype(str) == str(d)]
                    ranges = _compute_deck_ranges_for_dut(sub)
                    for (x0, x1, lab) in ranges:
                        try:
                            idx = sub[(sub["WL"].astype(float) >= x0) & (sub["WL"].astype(float) <= x1)].index
                            dlabels.loc[idx] = lab
                        except Exception:
                            continue
        except Exception:
            pass
        sdf["_deck"] = dlabels.fillna("UNK")

    # Facet by pagetype and dut
    ptypes = sorted(sdf["pagetype"].astype(str).unique())
    duts = sorted(sdf["dut"].astype(str).unique()) if "dut" in sdf.columns else ["all"]
    nrows, ncols = max(1, len(ptypes)), max(1, len(duts))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.4*ncols, 3.6*nrows), squeeze=False)
    try:
        cmap = plt.get_cmap('tab10')
    except Exception:
        cmap = plt.get_cmap('Set1')
    deck_order = {"LD": 0, "MD": 1, "UP": 2, "UNK": 3}
    all_decks = sorted(sdf["_deck"].astype(str).unique(), key=lambda s: deck_order.get(s, 99))
    deck_colors = {lab: cmap(i % 10) for i, lab in enumerate(all_decks)}

    for r_idx, ptype in enumerate(ptypes):
        for c_idx, dut in enumerate(duts):
            ax = axes[r_idx][c_idx]
            if "dut" in sdf.columns and dut != "all":
                sub = sdf[(sdf["pagetype"].astype(str) == str(ptype)) & (sdf["dut"].astype(str) == str(dut))]
            else:
                sub = sdf[sdf["pagetype"].astype(str) == str(ptype)]
            if sub.empty:
                ax.set_visible(False)
                continue
            for i, (dlab, grp) in enumerate(sorted(sub.groupby("_deck"), key=lambda kv: deck_order.get(str(kv[0]), 99))):
                x, y = _rcdf_xy(grp["RBER"].values)
                if x.size == 0:
                    continue
                ax.plot(x, y, label=str(dlab), color=deck_colors.get(dlab, cmap(i % 10)), lw=1.5, alpha=0.9)
            ax.set_xscale('log')
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("RBER (log)", fontsize=9)
            ax.set_ylabel("P(RBER ≥ x)", fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            try:
                leg = ax.legend(title='deck', loc='upper right', fontsize=8)
                leg.get_title().set_fontsize(9)
                leg.get_frame().set_alpha(0.85)
            except Exception:
                pass
            ax.grid(True, which='both', ls='--', alpha=0.4)
            try:
                if "dut" in sdf.columns and dut != "all":
                    ax.set_title(f"pagetype={ptype}, dut={dut}", fontsize=10)
                else:
                    ax.set_title(f"pagetype={ptype}", fontsize=10)
            except Exception:
                pass

    ttl = "Reverse CDF of RBER by pagetype and deck"
    try:
        if "parttype" in sdf.columns:
            pts = sorted(set(sdf["parttype"].astype(str)))
            if len(pts) == 1:
                ttl = f"{ttl} — parttype {pts[0]}"
            elif 1 < len(pts) <= 4:
                ttl = f"{ttl} — parttypes {', '.join(pts)}"
            else:
                ttl = f"{ttl} — {len(pts)} parttypes"
    except Exception:
        pass
    fig.suptitle(ttl)
    try:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    except Exception:
        pass
    try:
        fig.savefig(out_png_path, dpi=200)
        print("saved: {0}".format(out_png_path))
    finally:
        plt.close(fig)


def generate_rcdf_pagetype_deck_per_dut_pngs(df, out_dir, base_prefix):
    """Generate RCDF of RBER by pagetype and deck, split by readtype (side-by-side), one DUT per PNG.
    - Rows: pagetype, Cols: readtype, for a single dut.
    - Lines colored by deck (LD/MD/UP/UNK). Derives deck from WL thirds per DUT if not present.
    Saves files: {out_dir}/{base_prefix}_rcdf_pagetype_deck_dut_{dut}.png
    """
    sdf = df.copy()
    try:
        sdf["RBER"] = pd.to_numeric(sdf["RBER"], errors='coerce')
    except Exception:
        pass
    sdf = sdf[np.isfinite(sdf.get("RBER", np.nan))]
    # Normalize WL
    if "WL" in sdf.columns:
        try:
            sdf["WL"] = pd.to_numeric(sdf["WL"], errors='coerce')
        except Exception:
            pass
    # Deck labels
    if "deck" in sdf.columns:
        def _norm_deck(x):
            s = str(x).strip().upper()
            if s.startswith("L"): return "LD"
            if s.startswith("M"): return "MD"
            if s.startswith("U"): return "UP"
            return s or "UNK"
        try:
            sdf["_deck"] = sdf["deck"].apply(_norm_deck)
        except Exception:
            sdf["_deck"] = "UNK"
    else:
        dlabels = pd.Series(index=sdf.index, dtype=object)
        try:
            if "dut" in sdf.columns and "WL" in sdf.columns:
                for d in sorted(sdf["dut"].astype(str).unique()):
                    sub = sdf[sdf["dut"].astype(str) == str(d)]
                    ranges = _compute_deck_ranges_for_dut(sub)
                    for (x0, x1, lab) in ranges:
                        try:
                            idx = sub[(sub["WL"].astype(float) >= x0) & (sub["WL"].astype(float) <= x1)].index
                            dlabels.loc[idx] = lab
                        except Exception:
                            continue
        except Exception:
            pass
        sdf["_deck"] = dlabels.fillna("UNK")

    try:
        cmap = plt.get_cmap('tab10')
    except Exception:
        cmap = plt.get_cmap('Set1')
    deck_order = {"LD": 0, "MD": 1, "UP": 2, "UNK": 3}
    all_decks = sorted(sdf["_deck"].astype(str).unique(), key=lambda s: deck_order.get(s, 99))
    deck_colors = {lab: cmap(i % 10) for i, lab in enumerate(all_decks)}

    duts = sorted(sdf["dut"].astype(str).unique()) if "dut" in sdf.columns else ["all"]
    for dut in duts:
        if dut == "all":
            df_d = sdf
        else:
            df_d = sdf[sdf["dut"].astype(str) == str(dut)]
        if df_d.empty:
            continue
        ptypes = sorted(df_d.get("pagetype", pd.Series(["all"])) .astype(str).unique())
        rtypes = sorted(df_d.get("readtype", pd.Series(["all"])) .astype(str).unique())
        nrows, ncols = max(1, len(ptypes)), max(1, len(rtypes))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.4*ncols, 3.4*nrows), squeeze=False)
        for r_idx, ptype in enumerate(ptypes):
            for c_idx, rtype in enumerate(rtypes):
                ax = axes[r_idx][c_idx]
                sub = df_d[(df_d.get("pagetype").astype(str) == str(ptype)) & (df_d.get("readtype").astype(str) == str(rtype))]
                if sub.empty:
                    ax.set_visible(False)
                    continue
                # plot per deck group
                for i, (dlab, grp) in enumerate(sorted(sub.groupby("_deck"), key=lambda kv: deck_order.get(str(kv[0]), 99))):
                    x, y = _rcdf_xy(grp["RBER"].values)
                    if x.size == 0:
                        continue
                    ax.plot(x, y, label=str(dlab), color=deck_colors.get(dlab, cmap(i % 10)), lw=1.5, alpha=0.9)
                ax.set_xscale('log')
                ax.set_ylim(0.0, 1.0)
                ax.set_xlabel("RBER (log)", fontsize=9)
                if c_idx == 0:
                    ax.set_ylabel("P(RBER ≥ x)", fontsize=9)
                ax.tick_params(axis='both', labelsize=8)
                ax.grid(True, which='both', ls='--', alpha=0.4)
                # show legend on last column of each row
                if c_idx == (ncols - 1) and all_decks:
                    try:
                        handles = [plt.Line2D([0],[0], color=deck_colors[lab], lw=2, label=str(lab)) for lab in all_decks]
                        leg = ax.legend(handles=handles, title='deck', loc='upper right', fontsize=8)
                        leg.get_title().set_fontsize(9)
                        leg.get_frame().set_alpha(0.85)
                    except Exception:
                        pass
                # small title per cell
                try:
                    ax.set_title(f"{ptype} · {rtype}", fontsize=9)
                except Exception:
                    pass
        # Title with dut and optional parttype
        ttl = f"RCDF of RBER by pagetype and deck — DUT {dut}"
        try:
            if "parttype" in df_d.columns:
                pts = sorted(set(df_d["parttype"].astype(str)))
                if len(pts) == 1:
                    ttl = f"{ttl} — parttype {pts[0]}"
                elif 1 < len(pts) <= 4:
                    ttl = f"{ttl} — parttypes {', '.join(pts)}"
                else:
                    ttl = f"{ttl} — {len(pts)} parttypes"
        except Exception:
            pass
        fig.suptitle(ttl)
        try:
            fig.tight_layout(rect=(0, 0, 1, 0.94))
        except Exception:
            pass
        out_path = os.path.join(out_dir, f"{base_prefix}_rcdf_pagetype_deck_dut_{_sanitize_label(dut)}.png")
        try:
            fig.savefig(out_path, dpi=200)
            print("saved: {0}".format(out_path))
        finally:
            plt.close(fig)
def generate_summary_png(df, out_png_path):
    """Generate a summary PNG using the same content as the PDF summary page."""
    summary = build_summary_text(df)
    fig = plt.figure(figsize=(8.5, 11))
    ttl = "Summary: commonalities and differences (DUTs vs readtypes)"
    try:
        if "parttype" in df.columns:
            pts = sorted(set(df["parttype"].astype(str)))
            if len(pts) == 1:
                ttl = f"{ttl} — parttype {pts[0]}"
            elif 1 < len(pts) <= 4:
                ttl = f"{ttl} — parttypes {', '.join(pts)}"
            else:
                ttl = f"{ttl} — {len(pts)} parttypes"
    except Exception:
        pass
    fig.suptitle(ttl, fontsize=14)
    fig.text(0.06, 0.96, "", fontsize=1)  # spacer
    fig.text(0.06, 0.93, summary, fontsize=10, family='monospace', va='top')
    try:
        fig.savefig(out_png_path, dpi=200)
        print("saved: {0}".format(out_png_path))
    finally:
        plt.close(fig)


def generate_readtype_correlation_png(df, out_png_path):
    """Correlation of RBER between readtypes on matching (dut, blk, page, pagetype).
    - Facet rows by pagetype and readtype-pair (stacked), columns by dut.
    - Scatter in log10 space; light dashed y=x reference; color by deck; annotate R^2.
    """
    # Prepare a working copy and identify required columns
    dfl = df.copy()
    # Find block column (case-insensitive search)
    blk_col = None
    for c in dfl.columns:
        cl = str(c).strip().lower()
        if cl in ("blk", "block", "bl"):
            blk_col = c
            break
    if blk_col is None:
        raise ValueError("correlation: requires a block column (aliases: blk, block, bl)")
    dfl = dfl.rename(columns={blk_col: "blk"})
    # Ensure page column exists (prefer explicit 'page'; else copy from WL)
    if "page" not in dfl.columns:
        if "WL" in dfl.columns:
            dfl["page"] = dfl["WL"]
        else:
            raise ValueError("correlation: requires 'page' or WL to derive page")
    # Basic coercions
    for col in ("RBER", "page"):
        if col in dfl.columns:
            try:
                dfl[col] = pd.to_numeric(dfl[col], errors='coerce')
            except Exception:
                pass
    # Filter rows with needed data
    need = ["dut", "blk", "page", "pagetype", "readtype", "RBER"]
    for c in need:
        if c not in dfl.columns:
            raise ValueError(f"correlation: missing required column: {c}")
    dfl = dfl.dropna(subset=need).copy()
    if dfl.empty:
        raise ValueError("correlation: no rows after filtering")
    # Build key and deck label per key
    dfl["_key"] = dfl[["dut", "blk", "page", "pagetype"]].astype(str).agg("|".join, axis=1)
    # Deck labels: use existing deck or derive from WL per DUT if possible
    deck_map = {}
    if "deck" in dfl.columns:
        def _norm_deck(x):
            s = str(x).strip().upper()
            if s.startswith("L"): return "LD"
            if s.startswith("M"): return "MD"
            if s.startswith("U"): return "UP"
            return s or "UNK"
        try:
            deck_map = dfl.assign(_deck=dfl["deck"].apply(_norm_deck)).groupby("_key")["_deck"].agg(lambda s: str(list(s)[0])).to_dict()
        except Exception:
            deck_map = {}
    if not deck_map and "WL" in dfl.columns:
        # derive from WL thirds per DUT
        dlabels = pd.Series(index=dfl.index, dtype=object)
        try:
            for d in sorted(dfl["dut"].astype(str).unique()):
                sub = dfl[dfl["dut"].astype(str) == str(d)]
                ranges = _compute_deck_ranges_for_dut(sub)
                for (x0, x1, lab) in ranges:
                    try:
                        idx = sub[(sub["WL"].astype(float) >= x0) & (sub["WL"].astype(float) <= x1)].index
                        dlabels.loc[idx] = lab
                    except Exception:
                        continue
        except Exception:
            pass
        deck_map = dfl.assign(_deck=dlabels.fillna("UNK")).groupby("_key")["_deck"].agg(lambda s: str(list(s)[0])).to_dict()
    # Pivot to wide per readtype
    try:
        wide = dfl.pivot_table(index=["_key", "dut"], columns="readtype", values="RBER", aggfunc='median')
    except Exception:
        wide = dfl.pivot_table(index=["_key", "dut"], columns="readtype", values="RBER", aggfunc='mean')
    wide = wide.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    if wide.empty or wide.shape[1] < 2:
        raise ValueError("correlation: need at least two readtypes with overlapping keys")
    readtypes = list(wide.columns)
    pairs = [(readtypes[i], readtypes[j]) for i in range(len(readtypes)) for j in range(i+1, len(readtypes))]
    duts = sorted(wide.reset_index()["dut"].astype(str).unique())
    # Map key->pagetype (for faceting)
    pt_map = dfl.groupby("_key")["pagetype"].agg(lambda s: str(list(s)[0])).to_dict()
    ptypes = sorted(set(pt_map.values()))
    # Colors for deck
    try:
        cmap = plt.get_cmap('tab10')
    except Exception:
        cmap = plt.get_cmap('Set1')
    deck_order = {"LD": 0, "MD": 1, "UP": 2, "UNK": 3}
    all_decks = sorted(set(deck_map.values()), key=lambda s: deck_order.get(str(s), 99))
    deck_colors = {lab: cmap(i % 10) for i, lab in enumerate(all_decks)}

    total_rows = len(ptypes) * len(pairs)
    ncols = max(1, len(duts))
    fig, axes = plt.subplots(nrows=total_rows, ncols=ncols, figsize=(4.8*ncols, 3.4*total_rows), squeeze=False)
    row_idx = 0
    for ptype in ptypes:
        for (rt_a, rt_b) in pairs:
            for c_idx, dut in enumerate(duts):
                ax = axes[row_idx][c_idx]
                wsub = wide.reset_index()
                wsub = wsub[(wsub["dut"].astype(str) == str(dut))]
                if wsub.empty or rt_a not in wsub.columns or rt_b not in wsub.columns:
                    ax.set_visible(False)
                    continue
                # Filter keys for this pagetype
                keys = wsub["_key"].values
                keep = [k for k in keys if pt_map.get(k, None) == ptype]
                if not keep:
                    ax.set_visible(False)
                    continue
                m = wsub["_key"].isin(keep).values
                xa = wsub.loc[m, rt_a].values
                yb = wsub.loc[m, rt_b].values
                kvec = wsub.loc[m, "_key"].values
                mask = np.isfinite(xa) & np.isfinite(yb)
                xa, yb, kvec = xa[mask], yb[mask], kvec[mask]
                if xa.size == 0:
                    ax.set_visible(False)
                    continue
                # Colors by deck
                dvec = [deck_map.get(k, "UNK") for k in kvec]
                cols = [deck_colors.get(lab, cmap(0)) for lab in dvec]
                # Scatter in log space
                xlog = np.log10(np.clip(xa, 1e-20, None))
                ylog = np.log10(np.clip(yb, 1e-20, None))
                ax.scatter(xlog, ylog, s=12, c=cols, alpha=0.85, edgecolors='none')
                # y=x reference
                try:
                    lo = float(np.nanmin(np.concatenate([xlog, ylog])))
                    hi = float(np.nanmax(np.concatenate([xlog, ylog])))
                    hi = max(hi, lo + 0.3)
                    ax.plot([lo, hi], [lo, hi], ls='--', lw=1.0, color=(0.6,0.6,0.6,0.6))
                    ax.set_xlim(lo, hi)
                    ax.set_ylim(lo, hi)
                except Exception:
                    pass
                # R^2 annotation (Pearson on log space)
                try:
                    if xlog.size >= 2:
                        xm = xlog - np.mean(xlog)
                        ym = ylog - np.mean(ylog)
                        denom = float(np.sqrt((xm**2).sum()) * np.sqrt((ym**2).sum()))
                        r = float((xm*ym).sum() / denom) if denom > 0 else np.nan
                        r2 = r*r if np.isfinite(r) else np.nan
                        ax.text(0.02, 0.96, (f"R²={r2:.3f}" if np.isfinite(r2) else "R²=nan"), transform=ax.transAxes,
                                ha='left', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                except Exception:
                    pass
                ax.set_xlabel(f"log10 RBER ({rt_a})", fontsize=9)
                ax.set_ylabel(f"log10 RBER ({rt_b})", fontsize=9)
                ax.tick_params(axis='both', labelsize=8)
                ax.grid(True, ls='--', alpha=0.4)
                try:
                    ax.set_title(f"pagetype={ptype} — dut={dut} — {rt_a} vs {rt_b}", fontsize=10)
                except Exception:
                    pass
                # Legend only once per row on last column
                if c_idx == ncols - 1 and all_decks:
                    try:
                        present = [lab for lab in all_decks if lab in set(dvec)]
                        handles = [plt.Line2D([0],[0], marker='o', color='w', label=str(lab), markerfacecolor=deck_colors[lab], markersize=6) for lab in present]
                        if handles:
                            leg = ax.legend(handles=handles, title='deck', loc='lower right', fontsize=8)
                            leg.get_title().set_fontsize(9)
                            leg.get_frame().set_alpha(0.85)
                    except Exception:
                        pass
            row_idx += 1
    ttl = "RBER correlation between readtypes on matching (blk, page, pagetype) — split by DUT"
    try:
        if "parttype" in dfl.columns:
            pts = sorted(set(dfl["parttype"].astype(str)))
            if len(pts) == 1:
                ttl = f"{ttl} — parttype {pts[0]}"
            elif 1 < len(pts) <= 4:
                ttl = f"{ttl} — parttypes {', '.join(pts)}"
            else:
                ttl = f"{ttl} — {len(pts)} parttypes"
    except Exception:
        pass
    fig.suptitle(ttl)
    try:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    except Exception:
        pass
    try:
        fig.savefig(out_png_path, dpi=200)
        print("saved: {0}".format(out_png_path))
    finally:
        plt.close(fig)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Variability plot of RBER vs WL, color by pagetype, split by readtype/dut")
    parser.add_argument("-i", "--input", default=r"C:\\Users\\rmaguad\\Documents\\Work\\logs\\read_eimpro\\read_v_eimpro.csv", help="Path to input CSV")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path (default: alongside input)")
    parser.add_argument("--pdf-output", default=None, help="Output PDF path for per-DUT pages + summary (default: alongside input)")
    parser.add_argument("--no-pdf", action="store_true", help="Do not generate the per-DUT PDF")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    parser.add_argument("--format", choices=["png", "pdf", "both"], default="both", help="Select output type")
    args = parser.parse_args(argv)

    in_path = args.input
    if not os.path.isfile(in_path):
        print("error: input file not found: {0}".format(in_path), file=sys.stderr)
        return 2

    try:
        df = load_data(in_path)
    except Exception as e:
        print("error: failed to load/prepare data: {0}".format(e), file=sys.stderr)
        return 3

    title = "RBER vs WL by pagetype - split by readtype and dut"
    if _HAVE_SEABORN:
        fig = plot_with_seaborn(df, title)
    else:
        print("note: seaborn not found; using matplotlib fallback.")
        fig = plot_with_matplotlib(df, title)

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_dir = os.path.dirname(in_path)
        out_path = os.path.join(out_dir, "{0}_variability.png".format(base))

    # Determine desired outputs based on --format (backward-compat with --no-pdf)
    fmt = args.format or "both"
    if args.format == "both" and args.no_pdf:
        fmt = "png"

    # Save PNG if requested
    if fmt in ("png", "both"):
        try:
            fig.savefig(out_path, dpi=200)
            print("saved: {0}".format(out_path))
        except Exception as e:
            print("warning: failed to save figure: {0}".format(e), file=sys.stderr)

        # Always emit a companion summary PNG next to the plot
        try:
            base = os.path.splitext(os.path.basename(in_path))[0]
            out_dir = os.path.dirname(in_path)
            summary_path = os.path.join(out_dir, f"{base}_summary.png")
            generate_summary_png(df, summary_path)
        except Exception as e:
            print("warning: failed to save summary PNG: {0}".format(e), file=sys.stderr)

        # RCDF by pagetype (original)
        try:
            rcdf_pt_path = os.path.join(out_dir, f"{base}_rcdf.png")
            generate_rcdf_pagetype_only_png(df, rcdf_pt_path)
        except Exception as e:
            print("warning: failed to save RCDF (pagetype) PNG: {0}".format(e), file=sys.stderr)

        # RCDF by pagetype and deck (additional)
        try:
            rcdf_pd_path = os.path.join(out_dir, f"{base}_rcdf_pagetype_deck.png")
            generate_rcdf_pagetype_deck_png(df, rcdf_pd_path)
        except Exception as e:
            print("warning: failed to save RCDF (pagetype/deck) PNG: {0}".format(e), file=sys.stderr)

        # RCDF by pagetype and deck, split by readtype, one DUT per page (additional)
        try:
            generate_rcdf_pagetype_deck_per_dut_pngs(df, out_dir, base)
        except Exception as e:
            print("warning: failed to save per-DUT RCDF (pagetype/deck) PNGs: {0}".format(e), file=sys.stderr)

        # Correlation between readtypes on matching (dut, blk, page, pagetype)
        try:
            corr_path = os.path.join(out_dir, f"{base}_correlation.png")
            generate_readtype_correlation_png(df, corr_path)
        except Exception as e:
            print("warning: failed to save correlation PNG: {0}".format(e), file=sys.stderr)

        # Additionally, if a parttype column exists, generate one set per parttype
        try:
            if "parttype" in df.columns:
                out_dir = os.path.dirname(in_path)
                base = os.path.splitext(os.path.basename(in_path))[0]
                for pt in sorted(df["parttype"].astype(str).unique()):
                    sub = df[df["parttype"] == pt]
                    if sub.empty:
                        continue
                    safe = _sanitize_label(pt)
                    ptitle = f"RBER vs WL by pagetype - split by readtype and dut — parttype {pt}"
                    try:
                        if _HAVE_SEABORN:
                            pfig = plot_with_seaborn(sub, ptitle)
                        else:
                            pfig = plot_with_matplotlib(sub, ptitle)
                    except Exception:
                        pfig = plt.figure(figsize=(10, 6))
                        pfig.suptitle(ptitle)
                        pfig.text(0.5, 0.5, "Plot could not be generated for this group.", ha='center', va='center')
                    var_path = os.path.join(out_dir, f"{base}_parttype_{safe}_variability.png")
                    try:
                        pfig.savefig(var_path, dpi=200)
                        print(f"saved: {var_path}")
                    except Exception as e:
                        print(f"warning: failed to save per-parttype variability for {pt}: {e}", file=sys.stderr)
                    finally:
                        plt.close(pfig)
                    # Summary per parttype
                    sum_path = os.path.join(out_dir, f"{base}_parttype_{safe}_summary.png")
                    try:
                        generate_summary_png(sub, sum_path)
                    except Exception as e:
                        print(f"warning: failed to save per-parttype summary for {pt}: {e}", file=sys.stderr)
                    # RCDF per parttype (pagetype only)
                    try:
                        rcdf_pt_g_path = os.path.join(out_dir, f"{base}_parttype_{safe}_rcdf.png")
                        generate_rcdf_pagetype_only_png(sub, rcdf_pt_g_path)
                    except Exception as e:
                        print(f"warning: failed to save per-parttype RCDF (pagetype) for {pt}: {e}", file=sys.stderr)
                    # RCDF per parttype (pagetype + deck)
                    try:
                        rcdf_pd_g_path = os.path.join(out_dir, f"{base}_parttype_{safe}_rcdf_pagetype_deck.png")
                        generate_rcdf_pagetype_deck_png(sub, rcdf_pd_g_path)
                    except Exception as e:
                        print(f"warning: failed to save per-parttype RCDF (pagetype/deck) for {pt}: {e}", file=sys.stderr)
                    # RCDF per parttype (pagetype + deck), per DUT with readtypes side-by-side
                    try:
                        generate_rcdf_pagetype_deck_per_dut_pngs(sub, out_dir, f"{base}_parttype_{safe}")
                    except Exception as e:
                        print(f"warning: failed to save per-DUT per-parttype RCDF (pagetype/deck) for {pt}: {e}", file=sys.stderr)
                    # Correlation per parttype
                    try:
                        corr_g_path = os.path.join(out_dir, f"{base}_parttype_{safe}_correlation.png")
                        generate_readtype_correlation_png(sub, corr_g_path)
                    except Exception as e:
                        print(f"warning: failed to save per-parttype correlation for {pt}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"warning: failed to generate per-parttype outputs: {e}", file=sys.stderr)

    # Generate multi-page PDF (per-DUT pages + summary)
    if fmt in ("pdf", "both"):
        if args.pdf_output:
            pdf_path = args.pdf_output
        else:
            base = os.path.splitext(os.path.basename(in_path))[0]
            out_dir = os.path.dirname(in_path)
            pdf_path = os.path.join(out_dir, "{0}_per_dut.pdf".format(base))
        try:
            generate_pdf(df, pdf_path)
            print("saved: {0}".format(pdf_path))
        except Exception as e:
            print("warning: failed to generate PDF: {0}".format(e), file=sys.stderr)

    if not args.no_show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
