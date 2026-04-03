#!/usr/bin/env python
"""
FDV EIMPRO Web Application

A comprehensive web app for parsing FDV log files, generating CSVs, and visualizing data.

Features:
- Parse FDV log files following guide_to_fdvlog.txt standards
- Generate CSV files with parsed data
- View CSV data in table format with sorting and filtering
- Generate scatter plots (X,Y charts) from selected columns
- Generate cumulative distribution (CDF) plots split by category columns
- Interactive plot generation with seaborn/matplotlib backend

Usage:
    python fdv_eimpro_webapp.py

    Then open http://localhost:5058 in your browser.
"""

from __future__ import annotations
import os
import sys
import re
import io
import json
import uuid
import tempfile
import hashlib
import colorsys
import statistics as stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from functools import lru_cache
import time

# Flask and data processing
from flask import Flask, render_template, request, redirect, url_for, flash, Response, send_file, jsonify
import threading

# Data processing
try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print("error: pandas and numpy are required. Please install with 'pip install pandas numpy'", file=sys.stderr)
    raise

# Plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-GUI backend
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator
    import seaborn as sns
    _HAVE_SEABORN = True
except ImportError:
    _HAVE_SEABORN = False
    import matplotlib.pyplot as plt

# Force a non-GUI Matplotlib backend for server-side rendering
os.environ.setdefault("MPLBACKEND", "Agg")

_HERE = Path(__file__).parent
app = Flask(__name__, template_folder=str(_HERE / 'templates'))
app.secret_key = os.environ.get("FDV_EIMPRO_WEBAPP_SECRET", "dev-secret-eimpro")

# Configure temporary directory
def _init_tempdir():
    override_tmp = (os.environ.get('FDV_EIMPRO_TMPDIR') or r'D:\fdv_eimpro_tmp').strip()
    best_path = None
    if override_tmp:
        try:
            p = Path(override_tmp)
            p.mkdir(parents=True, exist_ok=True)
            best_path = p
        except Exception:
            pass
    if best_path is None:
        for pth in [r'D:\fdv_eimpro_tmp', r'C:\fdv_eimpro_tmp']:
            try:
                p = Path(pth)
                p.mkdir(parents=True, exist_ok=True)
                best_path = p
                break
            except Exception:
                pass
    if best_path is None:
        best_path = Path(tempfile.gettempdir())
    os.environ['FDV_EIMPRO_TMPDIR'] = str(best_path)
    os.environ['TMP'] = str(best_path)
    os.environ['TEMP'] = str(best_path)
    tempfile.tempdir = str(best_path)
    return best_path

TMPDIR = _init_tempdir()

# ============================================================================
# FDV Log Parsing
# ============================================================================

class FDVLogParser:
    """Parse FDV log files according to guide_to_fdvlog.txt standards."""
    
    # Field aliases for flexible column detection
    ALIASES = {
        "WL": ["wl", "wordline", "word_line"],
        "RBER": ["rber", "raw_ber", "rawber", "ber", "error_rate", "ber_raw"],
        "pagetype": ["pagetype", "ptype", "page_type", "pagemaptype", "maptype", "pgtype"],
        "readtype": ["readtype", "rtype", "read_type", "readmode", "read_mode"],
        "dut": ["dut", "device", "unit", "chip", "die", "die_id", "dut_id"],
        "VCC": ["vcc", "vdd", "vcore"],
        "VCCQ": ["vccq", "vio"],
        "TEMP": ["temp", "temperature", "temp_c"],
    }
    
    FDV_OUTPUT_PATTERN = r'FDV OUTPUT \[([^\]]+)\]: ([^\]]*)'
    FDV_POLL_PATTERN = r'FDV POLL \[([^\]]+)\]: ([^\]]*)'
    TEST_TIME_START = r'Test Start Date.*?:\s*(\d+_\d+_\d+).*?Test Start Time:\s*(\d+):(\d+):(\d+)'
    TEST_TIME_END = r'Test End Date.*?:\s*(\d+_\d+_\d+).*?Test End Time:\s*(\d+):(\d+):(\d+)'
    
    def __init__(self):
        self.records = []
        self.test_start_time = None
        self.test_end_time = None
    
    def parse_file(self, filepath: str) -> pd.DataFrame:
        """Parse an FDV log file and return a DataFrame."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Extract test times
        self._extract_test_times(lines)
        
        # Parse FDV OUTPUT and FDV POLL lines
        for line in lines:
            try:
                if 'FDV OUTPUT' in line:
                    self._parse_fdv_output(line)
                elif 'FDV POLL' in line:
                    self._parse_fdv_poll(line)
            except Exception as e:
                # Skip malformed lines
                pass
        
        # Convert to DataFrame
        if not self.records:
            raise ValueError("No FDV records found in log file")
        
        df = pd.DataFrame(self.records)
        return df
    
    def _extract_test_times(self, lines: List[str]):
        """Extract test start and end times."""
        content = '\n'.join(lines)
        
        start_match = re.search(self.TEST_TIME_START, content)
        if start_match:
            date_str, hour, minute, second = start_match.groups()
            # Convert date format YYYY_MM_DD to datetime
            year, month, day = date_str.split('_')
            self.test_start_time = datetime(int(year), int(month), int(day), 
                                           int(hour), int(minute), int(second))
        
        end_match = re.search(self.TEST_TIME_END, content)
        if end_match:
            date_str, hour, minute, second = end_match.groups()
            year, month, day = date_str.split('_')
            self.test_end_time = datetime(int(year), int(month), int(day),
                                         int(hour), int(minute), int(second))
    
    def _parse_test_params(self, params_str: str) -> Dict[str, str]:
        """Parse test conditions/parameters."""
        params = {}
        for token in params_str.split(','):
            token = token.strip()
            if '=' in token:
                key, value = token.split('=', 1)
                params[key.strip()] = value.strip()
        return params
    
    def _extract_test_params(self, tname: str, test_params: Dict[str, str]) -> Dict[str, Any]:
        """Extract structured test parameters from tname and test conditions."""
        record = {}
        
        # Extract from test parameters
        record['VCC'] = float(test_params.get('VCC', 0)) if test_params.get('VCC', '0').replace('.','').isdigit() else None
        record['VCCQ'] = float(test_params.get('VCCQ', 0)) if test_params.get('VCCQ', '0').replace('.','').isdigit() else None
        record['TEMP'] = float(test_params.get('TEMP', 0)) if test_params.get('TEMP', '0').replace('.','').isdigit() else None
        record['TM'] = test_params.get('TM', '')
        
        # Extract from tname
        # Pattern: BLK_<number>, PG_<number>, PGTYPE_<type>, WL_<number>, etc.
        tname_upper = tname.upper()
        
        # Extract product type / pagetype (MLC, QLC, TLC, SSLC, DSLC)
        pagetype_match = re.search(r'(MLC|QLC|TLC|SSLC|DSLC|LP|UP|XP|TP)', tname_upper)
        if pagetype_match:
            record['pagetype'] = pagetype_match.group(1)
        else:
            record['pagetype'] = 'UNKNOWN'
        
        # Extract WL (wordline)
        wl_match = re.search(r'WL[_:](\d+)', tname_upper)
        if wl_match:
            record['WL'] = int(wl_match.group(1))
        
        # Extract BLK (block address)
        blk_match = re.search(r'BLK[_:](\d+)', tname_upper)
        if blk_match:
            record['BLK'] = int(blk_match.group(1))
        
        # Extract PG (page)
        pg_match = re.search(r'PG[_:](\d+)', tname_upper)
        if pg_match:
            record['PG'] = int(pg_match.group(1))
        
        # Extract status (C0, E0, F0, etc.)
        status_match = re.search(r'([CEF]\d)', tname_upper)
        if status_match:
            record['status'] = status_match.group(1)
        
        # Extract plane operation (SP, MP)
        plane_match = re.search(r'_(SP|MP)_', tname_upper)
        if plane_match:
            record['plane_op'] = plane_match.group(1)
        
        # Extract test name (base name before parameters)
        # Usually at beginning before BLK/PG/PGTYPE
        testname_match = re.match(r'^([A-Z_]+?)(?:_BLK|_PG|_PGTYPE|_WL)', tname_upper)
        if testname_match:
            record['testname'] = testname_match.group(1).lower()
        else:
            record['testname'] = tname.lower()
        
        # Extract DECK (LD, MD, UP)
        deck_match = re.search(r'DECK[_:]([A-Z]+)|_(LD|MD|UP)_', tname_upper)
        if deck_match:
            record['deck'] = deck_match.group(1) or deck_match.group(2)
        
        # Extract SHMOO parameters
        shmoo_match = re.search(r'SHMOO[_:](\d+)', tname_upper)
        if shmoo_match:
            record['shmoo'] = int(shmoo_match.group(1))
        
        return record
    
    def _parse_fdv_output(self, line: str):
        """Parse FDV OUTPUT line."""
        match = re.search(self.FDV_OUTPUT_PATTERN, line)
        if not match:
            return
        
        header, data = match.groups()
        
        # Extract path/file and test name
        # Format: path/filename.FDV::tname,test_conditions
        parts = header.split('::')
        if len(parts) < 2:
            return
        
        file_part = parts[0]
        tname_and_cond = parts[1]
        
        # Split tname and conditions
        tname_parts = tname_and_cond.split(',', 1)
        tname = tname_parts[0]
        conditions_str = tname_parts[1] if len(tname_parts) > 1 else ''
        
        # Parse test conditions
        test_params = self._parse_test_params(conditions_str)
        
        # Parse data fields: DUT,Result,Bytes,FailBytes,FailRate,FailBits,FailBitRate,RBERLimit,...
        data_fields = data.split(',')
        if len(data_fields) < 3:
            return
        
        dut = data_fields[0].strip()
        result = data_fields[1].strip() if len(data_fields) > 1 else 'UNKNOWN'
        rber = None
        
        # Try to extract numeric values (FailRate or RBER might be in position 4 or 5)
        try:
            if len(data_fields) > 4:
                rber = float(data_fields[4])
        except (ValueError, IndexError):
            try:
                # Alternative: might be raw RBER value
                if len(data_fields) > 5:
                    rber = float(data_fields[5])
            except (ValueError, IndexError):
                pass
        
        # Extract test parameters
        record = self._extract_test_params(tname, test_params)
        record['dut'] = dut.replace('DUT', '').strip()
        record['result'] = result
        record['tname'] = tname
        record['log_type'] = 'FDV_OUTPUT'
        
        if rber is not None:
            record['RBER'] = rber
        
        self.records.append(record)
    
    def _parse_fdv_poll(self, line: str):
        """Parse FDV POLL line."""
        match = re.search(self.FDV_POLL_PATTERN, line)
        if not match:
            return
        
        header, data = match.groups()
        
        # Extract path/file and test name
        parts = header.split('::')
        if len(parts) < 2:
            return
        
        tname_and_cond = parts[1]
        tname_parts = tname_and_cond.split(',', 1)
        tname = tname_parts[0]
        conditions_str = tname_parts[1] if len(tname_parts) > 1 else ''
        
        # Parse test conditions
        test_params = self._parse_test_params(conditions_str)
        
        # Parse data: DUT 0, measurement, ...
        data_fields = data.split(',')
        if len(data_fields) < 2:
            return
        
        dut_part = data_fields[0].strip()
        measurement = None
        
        # Extract DUT number and measurement
        dut_match = re.match(r'DUT(\d+)', dut_part)
        if dut_match:
            dut = dut_match.group(1)
        else:
            dut = dut_part
        
        try:
            if len(data_fields) > 1:
                # Skip the "0" and get measurement
                measurement = float(data_fields[2]) if len(data_fields) > 2 else float(data_fields[1])
        except (ValueError, IndexError):
            pass
        
        # Extract test parameters
        record = self._extract_test_params(tname, test_params)
        record['dut'] = str(dut)
        record['measurement'] = measurement
        record['tname'] = tname
        record['log_type'] = 'FDV_POLL'
        
        # For POLL data, measurement might be RBER or other metric
        if measurement is not None and measurement > 0:
            record['value'] = measurement
        
        self.records.append(record)


# ============================================================================
# Plot Generation
# ============================================================================

def find_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Find and resolve column names using aliases."""
    cols_lower = {c.lower(): c for c in df.columns}
    resolved = {}
    
    for target, names in FDVLogParser.ALIASES.items():
        for name in names:
            if name in cols_lower:
                resolved[target] = cols_lower[name]
                break
    
    return resolved


def generate_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                         hue_col: Optional[str] = None, 
                         title: str = "Scatter Plot") -> bytes:
    """Generate a scatter plot from DataFrame."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if hue_col and hue_col in df.columns:
        # Color by category
        categories = df[hue_col].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
        
        for i, cat in enumerate(categories):
            subset = df[df[hue_col] == cat]
            ax.scatter(subset[x_col], subset[y_col], 
                      label=str(cat), alpha=0.6, s=30, color=colors[i])
        
        ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(df[x_col], df[y_col], alpha=0.6, s=30)
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf.getvalue()


def generate_cdf_plot(df: pd.DataFrame, value_col: str, category_col: str,
                     split_col: Optional[str] = None,
                     title: str = "CDF Plot") -> bytes:
    """Generate a cumulative distribution plot with categories."""
    
    if value_col not in df.columns or category_col not in df.columns:
        raise ValueError(f"Missing columns: {value_col} or {category_col}")
    
    # Remove NaN values
    df_clean = df[[value_col, category_col]].dropna()
    if df_clean.empty:
        raise ValueError("No valid data for CDF plot")
    
    categories = sorted(df_clean[category_col].unique())
    
    if split_col and split_col in df.columns:
        splits = sorted(df[split_col].unique())
        nrows = len(splits)
        fig, axes = plt.subplots(nrows=nrows, figsize=(10, 4*nrows))
        if nrows == 1:
            axes = [axes]
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
    
    for ax_idx, ax in enumerate(axes):
        split_val = splits[ax_idx] if split_col and split_col in df.columns else None
        
        for cat_idx, cat in enumerate(categories):
            if split_col and split_col in df.columns and split_val is not None:
                subset = df_clean[(df[category_col] == cat) & (df[split_col] == split_val)]
            else:
                subset = df_clean[df_clean[category_col] == cat]
            
            if subset.empty:
                continue
            
            values = np.sort(subset[value_col].dropna().values)
            cdf = np.arange(1, len(values) + 1) / len(values)
            
            ax.plot(values, cdf, marker='o', linestyle='-', label=str(cat),
                   color=colors[cat_idx], alpha=0.7, markersize=4)
        
        ax.set_xlabel(value_col)
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'{title} ({split_val})' if split_val else title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf.getvalue()


# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """Main landing page."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and parsing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file
        upload_path = TMPDIR / f"{uuid.uuid4()}_{file.filename}"
        file.save(str(upload_path))
        
        # Parse log file
        parser = FDVLogParser()
        df = parser.parse_file(str(upload_path))
        
        # Save CSV
        csv_filename = f"parsed_{uuid.uuid4()}.csv"
        csv_path = TMPDIR / csv_filename
        df.to_csv(str(csv_path), index=False)
        
        # Return CSV info and data preview
        return jsonify({
            'success': True,
            'csv_id': csv_filename,
            'row_count': len(df),
            'columns': list(df.columns),
            'preview': df.head(10).to_dict('records'),
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/csv/<csv_id>/data', methods=['GET'])
def get_csv_data(csv_id: str):
    """Get CSV data with optional filtering and sorting."""
    try:
        csv_path = TMPDIR / csv_id
        if not csv_path.exists():
            return jsonify({'error': 'CSV not found'}), 404
        
        df = pd.read_csv(str(csv_path))
        
        # Apply filters
        filters = request.args.get('filters', '{}')
        try:
            filter_dict = json.loads(filters)
            for col, val in filter_dict.items():
                if col in df.columns and val:
                    df = df[df[col].astype(str).str.contains(str(val), case=False)]
        except:
            pass
        
        # Apply sorting
        sort_col = request.args.get('sort_col')
        sort_order = request.args.get('sort_order', 'asc')
        if sort_col and sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=(sort_order == 'asc'))
        
        # Pagination
        page = int(request.args.get('page', 0))
        per_page = int(request.args.get('per_page', 100))
        start = page * per_page
        end = start + per_page
        
        return jsonify({
            'success': True,
            'total_rows': len(df),
            'page': page,
            'per_page': per_page,
            'data': df.iloc[start:end].to_dict('records'),
            'columns': list(df.columns),
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/csv/<csv_id>/download', methods=['GET'])
def download_csv(csv_id: str):
    """Download the CSV file."""
    try:
        csv_path = TMPDIR / csv_id
        if not csv_path.exists():
            return jsonify({'error': 'CSV not found'}), 404
        
        return send_file(str(csv_path), as_attachment=True, download_name=csv_id)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/plot/scatter', methods=['POST'])
def create_scatter_plot():
    """Create a scatter plot from selected columns."""
    try:
        data = request.json
        csv_id = data.get('csv_id')
        x_col = data.get('x_col')
        y_col = data.get('y_col')
        hue_col = data.get('hue_col')
        title = data.get('title', 'Scatter Plot')
        
        if not all([csv_id, x_col, y_col]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        csv_path = TMPDIR / csv_id
        if not csv_path.exists():
            return jsonify({'error': 'CSV not found'}), 404
        
        df = pd.read_csv(str(csv_path))
        
        # Validate columns
        if x_col not in df.columns or y_col not in df.columns:
            return jsonify({'error': 'Invalid columns'}), 400
        
        # Generate plot
        plot_bytes = generate_scatter_plot(df, x_col, y_col, hue_col, title)
        
        # Save plot
        plot_id = f"scatter_{uuid.uuid4()}.png"
        plot_path = TMPDIR / plot_id
        with open(str(plot_path), 'wb') as f:
            f.write(plot_bytes)
        
        return jsonify({
            'success': True,
            'plot_id': plot_id,
            'plot_url': f'/api/plot/{plot_id}',
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/plot/cdf', methods=['POST'])
def create_cdf_plot():
    """Create a CDF plot from selected columns."""
    try:
        data = request.json
        csv_id = data.get('csv_id')
        value_col = data.get('value_col')
        category_col = data.get('category_col')
        split_col = data.get('split_col')
        title = data.get('title', 'CDF Plot')
        
        if not all([csv_id, value_col, category_col]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        csv_path = TMPDIR / csv_id
        if not csv_path.exists():
            return jsonify({'error': 'CSV not found'}), 404
        
        df = pd.read_csv(str(csv_path))
        
        # Validate columns
        if value_col not in df.columns or category_col not in df.columns:
            return jsonify({'error': 'Invalid columns'}), 400
        
        # Generate plot
        plot_bytes = generate_cdf_plot(df, value_col, category_col, split_col, title)
        
        # Save plot
        plot_id = f"cdf_{uuid.uuid4()}.png"
        plot_path = TMPDIR / plot_id
        with open(str(plot_path), 'wb') as f:
            f.write(plot_bytes)
        
        return jsonify({
            'success': True,
            'plot_id': plot_id,
            'plot_url': f'/api/plot/{plot_id}',
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/plot/<plot_id>')
def get_plot(plot_id: str):
    """Retrieve a saved plot."""
    try:
        plot_path = TMPDIR / plot_id
        if not plot_path.exists():
            return 'Plot not found', 404
        
        return send_file(str(plot_path), mimetype='image/png')
    
    except Exception as e:
        return f'Error: {str(e)}', 500


@app.route('/api/stats/<csv_id>', methods=['GET'])
def get_stats(csv_id: str):
    """Get basic statistics for the CSV."""
    try:
        csv_path = TMPDIR / csv_id
        if not csv_path.exists():
            return jsonify({'error': 'CSV not found'}), 404
        
        df = pd.read_csv(str(csv_path))
        
        stats_data = {}
        for col in df.columns:
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if not numeric_col.isna().all():
                    stats_data[col] = {
                        'mean': float(numeric_col.mean()),
                        'std': float(numeric_col.std()),
                        'min': float(numeric_col.min()),
                        'max': float(numeric_col.max()),
                        'count': int(numeric_col.count()),
                    }
            except:
                stats_data[col] = {
                    'unique_count': int(df[col].nunique()),
                    'type': 'categorical',
                }
        
        return jsonify({'success': True, 'stats': stats_data})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting FDV EIMPRO WebApp...")
    print("Visit http://localhost:5058")
    app.run(debug=False, host='0.0.0.0', port=5058, threaded=True)
