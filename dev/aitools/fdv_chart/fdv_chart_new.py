#!/usr/bin/env python3
"""
FDV Chart Web Application - SIMPLIFIED

A simple web app for parsing FDV log files, generating CSVs, and viewing data.

Features:
- Parse FDV log files
- Generate CSV files with parsed data
- View CSV data in a simple scrollable table
- Download CSV

Usage:
    python fdv_chart_new.py
    Then open http://localhost:5058 in your browser.
"""

from __future__ import print_function
import os
import sys
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional

# Flask
from flask import Flask, render_template, request, jsonify, send_file

# Data processing
try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("error: pandas and numpy are required", file=sys.stderr)
    sys.exit(1)

# Plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    pass

_HERE = Path(__file__).parent
app = Flask(__name__, template_folder=str(_HERE / 'templates'))
app.secret_key = "fdv-chart-secret"

# Setup temp directory
TMPDIR = Path(r'D:\fdv_chart_tmp')
TMPDIR.mkdir(parents=True, exist_ok=True)

# Global state
current_csv_id = None
current_df = None


def _parse_fdv_log(file_path: str) -> pd.DataFrame:
    """Parse FDV log file and return DataFrame."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Simple parsing - extract data from log file
        data = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Split by whitespace or comma
            if ',' in line:
                parts = [x.strip() for x in line.split(',')]
            else:
                parts = line.split()
            if parts:
                data.append(parts)
        
        # Create DataFrame with generic column names if needed
        if data:
            # Try to infer columns from first line
            df = pd.DataFrame(data)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        raise Exception(f"Failed to parse file: {str(e)}")


@app.route('/')
def index():
    """Main page."""
    return render_template('simple.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and parse a file."""
    global current_csv_id, current_df
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Parse file
        temp_path = TMPDIR / f"upload_{uuid.uuid4().hex}"
        file.save(str(temp_path))
        
        # Parse the log file
        df = _parse_fdv_log(str(temp_path))
        temp_path.unlink()  # Delete temp file
        
        if df.empty:
            return jsonify({'success': False, 'error': 'File is empty'}), 400
        
        # Save as CSV
        csv_id = f"parsed_{uuid.uuid4().hex}.csv"
        csv_path = TMPDIR / csv_id
        df.to_csv(str(csv_path), index=False)
        
        # Update globals
        current_csv_id = csv_id
        current_df = df
        
        return jsonify({
            'success': True,
            'csv_id': csv_id,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/csv/<csv_id>/data')
def get_csv_data(csv_id: str):
    """Get all CSV data."""
    try:
        csv_path = TMPDIR / csv_id
        if not csv_path.exists():
            return jsonify({'success': False, 'error': 'CSV not found'}), 404
        
        df = pd.read_csv(str(csv_path))
        
        # Convert to JSON-safe format
        data = []
        for _, row in df.iterrows():
            row_dict = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    row_dict[col] = ''
                elif isinstance(val, (np.integer, np.floating)):
                    if np.isinf(val) or np.isnan(val):
                        row_dict[col] = ''
                    else:
                        row_dict[col] = str(val)
                else:
                    row_dict[col] = str(val)
            data.append(row_dict)
        
        return jsonify({
            'success': True,
            'total_rows': len(df),
            'columns': list(df.columns),
            'data': data
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/csv/<csv_id>/download')
def download_csv(csv_id: str):
    """Download CSV file."""
    try:
        csv_path = TMPDIR / csv_id
        if not csv_path.exists():
            return jsonify({'success': False, 'error': 'CSV not found'}), 404
        
        return send_file(str(csv_path), as_attachment=True, download_name=csv_id)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting FDV Chart WebApp...")
    print("Visit http://localhost:5058")
    app.run(host='0.0.0.0', port=5058, debug=False)
