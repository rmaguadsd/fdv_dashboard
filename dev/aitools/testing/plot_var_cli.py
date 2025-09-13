#!/usr/bin/env python
"""
Generate WL vs RBER variability plot for a selected testname from FDV OUTPUT logs.

Usage (PowerShell examples):
  python plot_var_cli.py --files "C:\\path\\to\\Output_site...txt" \
    --sel "EIMPRO_ECC|19|K450917_753_-13_3|2.5|12|25" --out plot.png [--fallback 1]

Notes:
- --files accepts one or more files or directories (recursively lists files in directories).
- --sel is "testname|pr|fuseid|vcc|tm|temp"; fields after testname are optional and can be empty.
- By default WL is taken from WL_<n> token in tname; if missing and --fallback 1, PAGE tokens are used.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import sys

# Import helpers from the web app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add aitools/
from fdv_report2_webapp import _read_rows_from_paths, _build_variability_records  # type: ignore


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--files', nargs='+', required=True, help='FDV OUTPUT files or directories')
    p.add_argument('--sel', required=True, help='Selection: testname|pr|fuseid|vcc|tm|temp')
    p.add_argument('--out', required=True, help='Output PNG file')
    p.add_argument('--fallback', type=int, default=1, help='Allow PAGE fallback when WL missing (1=yes,0=no)')
    args = p.parse_args()

    paths = []
    for s in args.files:
        pth = Path(s)
        if not pth.exists():
            print(f"warn: path not found: {pth}", file=sys.stderr)
        paths.append(pth)
    rows: List[Dict[str, str]] = _read_rows_from_paths(paths)
    if not rows:
        print('no rows read from input files', file=sys.stderr)
        return 2

    # We accept all fdv_files (wildcard) so users don’t have to specify --fdv explicitly
    selectors = ['']  # any FDV
    parts = [x.strip() for x in args.sel.split('|')]
    while len(parts) < 6:
        parts.append('')
    entries: List[Tuple[str, ...]] = [(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])]

    recs = _build_variability_records(rows, selectors, entries, allow_page_fallback=bool(args.fallback))
    if not recs:
        # Relax FuseID
        entries2 = [(parts[0], parts[1], '', parts[3], parts[4], parts[5])]
        recs = _build_variability_records(rows, selectors, entries2, allow_page_fallback=bool(args.fallback))
    if not recs:
        # Testname-only
        entries3 = [(parts[0], '', '', '', '', '')]
        recs = _build_variability_records(rows, selectors, entries3, allow_page_fallback=bool(args.fallback))
    if not recs:
        print('no data for this selection', file=sys.stderr)
        return 3

    # Plot like the web app
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    duts = sorted({r['dut'] for r in recs})
    n = max(1, len(duts))
    ncols = min(3, n)
    from math import ceil
    nrows = int(ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 3.6*nrows), squeeze=False)
    cmap = plt.get_cmap('tab10')
    top = sorted(recs, key=lambda r: r['RBER'], reverse=True)[:20]
    top_set = set((t['dut'], t['WL'], t['RBER']) for t in top)
    for idx, dut in enumerate(duts):
        r_i = idx // ncols
        c_i = idx % ncols
        ax = axes[r_i][c_i]
        sub = [rc for rc in recs if rc['dut'] == dut]
        if not sub:
            ax.set_visible(False)
            continue
        by_pt: Dict[str, List[Dict]] = {}
        for rc in sub:
            by_pt.setdefault(rc['pagetype'], []).append(rc)
        for i, (pt, arr) in enumerate(sorted(by_pt.items(), key=lambda kv: str(kv[0]))):
            xs = [rc['WL'] for rc in arr]
            ys = [rc['RBER'] for rc in arr]
            ax.scatter(xs, ys, s=18, alpha=0.7, color=cmap(i % 10), label=str(pt or '-'))
        outs = [rc for rc in sub if (rc['dut'], rc['WL'], rc['RBER']) in top_set]
        if outs:
            ax.scatter([o['WL'] for o in outs], [o['RBER'] for o in outs], s=60, facecolors='none', edgecolors='red', linewidths=1.2, label='top20')
        ax.set_yscale('log')
        ax.set_xlabel('WL')
        ax.set_ylabel('RBER (log)')
        ax.set_title(dut)
        ax.grid(True, which='both', ls='--', alpha=0.4)
    total_axes = nrows * ncols
    for j in range(len(duts), total_axes):
        r_i = j // ncols
        c_i = j % ncols
        axes[r_i][c_i].set_visible(False)
    # Put legend from first visible axis
    first_ax = None
    for r in range(nrows):
        for c in range(ncols):
            if axes[r][c].get_visible():
                first_ax = axes[r][c]
                break
        if first_ax is not None:
            break
    if first_ax is not None:
        handles, labels = first_ax.get_legend_handles_labels()
        if labels:
            fig.legend(handles, labels, title='pagetype', loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=8)
    fig.suptitle(f"RBER vs WL by pagetype — {parts[0]}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    raise SystemExit(main())
