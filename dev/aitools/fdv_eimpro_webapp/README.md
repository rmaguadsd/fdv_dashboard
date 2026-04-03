# FDV EIMPRO Log Parser & Visualizer Web App

A comprehensive web application for parsing FDV (Functional Diagnostic Vector) and POLL log files, generating CSV exports, and creating interactive visualizations.

## Features

✨ **Core Features:**
- 📤 **FDV Log Parsing** - Parse complex FDV OUTPUT and FDV POLL log files according to `guide_to_fdvlog.txt` specifications
- 📋 **CSV Generation** - Automatically extract and organize test data into structured CSV format
- 🔍 **Data Exploration** - Browse, filter, and search parsed data with pagination
- 📊 **Scatter Plots** - Generate X-Y scatter plots with optional category-based coloring
- 📈 **CDF Plots** - Create cumulative distribution functions split by categories
- 📉 **Statistics** - View comprehensive statistical summaries (mean, std, min, max, count)
- 💾 **Export** - Download generated CSV and plots for offline analysis

## Supported Log Formats

### FDV OUTPUT
Functional test data with pass/fail results and bit error rates.

```
FDV OUTPUT [D:\NAND\150S\FDV\STAGING\RMAGUAD/BASIC_ERASE_PROGRAM_PAGE_READ_EIMPRO/BASIC_PROGRAM_PAGE_READ_EIMPRO_QLC.FDV::READ_ECC_BLK_778_PG_44_PGTYPE_LP_WL_2_SB_8_BL_3,SSYNC=TRUE,TRC=,DUTTEMP=-999,TAC=5.725000,SPECOFFSET=0.125,TM=12,VCC=2.5,VCCQ=1.2,TEMP=25]: DUT1,PASS,18592,29,0.0016,29,0.00019,0.008,FAILCOUNT_ONLY,
```

### FDV POLL
Parametric/characterization data with measurement values.

```
FDV POLL [D:\NAND\150S\FDV\STAGING\RMAGUAD/char/array_char/<TR_SSLC>.FDV::<POLL_TR_C0_SP_READ_BLK_364_PG_82_PGTYPE_SSLC_WL_4_SB_10_BL_1,SSYNC=FALSE,TRC=,DUTTEMP=-999,TAC=6.300000,SPECOFFSET=0,TM=12,VCC=2.5,VCCQ=1.2,TEMP=25]: DUT1 0,33.205807,39839,[Per = 3334.000000/4000000.000000, nTECCount: 39839.000000]
```

## Extracted Data Fields

The parser automatically extracts and organizes the following fields:

| Field | Source | Description |
|-------|--------|-------------|
| `dut` | Test data | Device Under Test identifier (DUT1, DUT2, etc.) |
| `testname` | Test name | Base test operation name (READ, PROGRAM, ERASE, etc.) |
| `tname` | Test name | Full test name including parameters |
| `pagetype` | Test name | Page type (SLC, MLC, TLC, QLC, SSLC, DSLC, LP, UP, XP, TP) |
| `WL` | Test name | Wordline address |
| `BLK` | Test name | Block address |
| `PG` | Test name | Page address |
| `status` | Test name | Status code (C0, E0, F0, etc.) |
| `plane_op` | Test name | Plane operation (SP=Single Plane, MP=Multi Plane) |
| `deck` | Test name | Deck info (LD=Lower, MD=Middle, UP=Upper) |
| `VCC` | Test conditions | Core voltage |
| `VCCQ` | Test conditions | I/O voltage |
| `TEMP` | Test conditions | Temperature (°C) |
| `TM` | Test conditions | Test mode |
| `RBER` | Test data | Raw Bit Error Rate (FDV OUTPUT) |
| `value` / `measurement` | Test data | Measurement value (FDV POLL) |
| `result` | Test data | Test result (PASS/FAIL for FDV OUTPUT) |
| `log_type` | Metadata | Log type (FDV_OUTPUT or FDV_POLL) |

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup

1. **Navigate to the app directory:**
   ```powershell
   cd "d:\FDV\git\fdv_dashboard\dev\aitools\fdv_eimpro_webapp"
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

### Launch the Web Application

#### Option 1: Direct Python Execution
```powershell
python fdv_eimpro_webapp.py
```

#### Option 2: Using Python 3 with py launcher
```powershell
py -3 fdv_eimpro_webapp.py
```

#### Expected Output
```
Starting FDV EIMPRO WebApp...
Visit http://localhost:5058
 * Running on http://0.0.0.0:5058
 * Debug mode: off
```

### Access the Application
Open your web browser and navigate to:
```
http://localhost:5058
```

## User Guide

### 1. Upload & Parse Log File

1. Click the **"📤 Upload & Parse"** tab
2. Click "Select FDV Log File" and choose your `.txt` or `.log` file
3. Click **"Parse File"**
4. The app will:
   - Parse all FDV OUTPUT and FDV POLL lines
   - Extract test parameters and conditions
   - Generate a structured CSV
   - Show preview of first 10 rows
5. Click **"⬇️ Download CSV"** to save the generated CSV file

### 2. View & Filter Data

1. Click the **"📋 View Data"** tab
2. Use filters to search:
   - Search all columns with text filter
   - Filter by specific column and value
3. Results are paginated (50 rows per page)
4. Click column headers to sort (future enhancement)

### 3. Create Scatter Plots

1. Click the **"📈 Scatter Plot"** tab
2. Select:
   - **X-Axis Column**: Numeric column for horizontal axis
   - **Y-Axis Column**: Numeric column for vertical axis
   - **Color By (Optional)**: Categorical column to color-code points
   - **Plot Title**: Custom title for the plot
3. Click **"📈 Generate Plot"**
4. Plot appears below in PNG format

**Example Scatter Plots:**
- WL (X) vs RBER (Y) colored by pagetype
- VCC (X) vs RBER (Y) colored by dut
- BLK (X) vs value (Y) colored by status

### 4. Create CDF Plots

1. Click the **"📊 CDF Plot"** tab
2. Select:
   - **Value Column**: Numeric column for the distribution
   - **Category Column**: Column to split into separate CDF curves
   - **Split By (Optional)**: Additional categorical column for faceted plots
   - **Plot Title**: Custom title
3. Click **"📊 Generate Plot"**
4. Cumulative distribution plot appears below

**Example CDF Plots:**
- RBER distribution split by pagetype (each pagetype gets a curve)
- Value distribution split by dut (one subplot per DUT)
- WL distribution split by deck (LD, MD, UP)

### 5. View Statistics

1. Click the **"📉 Statistics"** tab
2. View automatically calculated statistics:
   - **Numeric columns**: mean, std deviation, min, max, count
   - **Categorical columns**: unique value count

### 6. Help & Documentation

Click the **"ℹ️ Help"** tab for comprehensive documentation, field descriptions, and feature overview.

## Architecture

### Backend (Flask)

**Main Components:**

| Component | Purpose |
|-----------|---------|
| `FDVLogParser` | Parses FDV log files, extracts test parameters |
| `find_columns()` | Resolves column names using aliases |
| `generate_scatter_plot()` | Creates matplotlib scatter plots |
| `generate_cdf_plot()` | Creates matplotlib CDF plots |
| Flask Routes | RESTful API endpoints for UI |

**Key Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/upload` | POST | Upload and parse log file |
| `/api/csv/<id>/data` | GET | Retrieve CSV data with filters/sorting |
| `/api/csv/<id>/download` | GET | Download CSV file |
| `/api/plot/scatter` | POST | Generate scatter plot |
| `/api/plot/cdf` | POST | Generate CDF plot |
| `/api/plot/<id>` | GET | Retrieve saved plot image |
| `/api/stats/<id>` | GET | Get statistical summaries |

### Frontend (HTML/CSS/JavaScript)

**Tabs:**
1. **Upload & Parse** - File upload, parsing, preview, download
2. **View Data** - Data table with filtering and pagination
3. **Scatter Plot** - Interactive scatter plot generation
4. **CDF Plot** - Interactive CDF plot generation
5. **Statistics** - Statistical summaries in card grid
6. **Help** - Documentation and usage guide

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FDV_EIMPRO_TMPDIR` | `D:\fdv_eimpro_tmp` | Temporary directory for uploads/plots |
| `FDV_EIMPRO_WEBAPP_SECRET` | `dev-secret-eimpro` | Flask session secret key |

### Set Environment Variables (PowerShell)

```powershell
$env:FDV_EIMPRO_TMPDIR = "D:\my_custom_temp"
$env:FDV_EIMPRO_WEBAPP_SECRET = "my-secret-key"
python fdv_eimpro_webapp.py
```

## Performance Considerations

- **Large Files**: Parser uses streaming line-by-line processing
- **Memory**: Pandas DataFrames loaded into memory
- **Plots**: Generated on-demand using matplotlib (non-GUI backend)
- **Caching**: Column aliases cached for repeated operations
- **Cleanup**: Temporary files auto-managed (consider periodic cleanup)

## Troubleshooting

### Port Already in Use
```powershell
# Find process using port 5058
netstat -ano | findstr :5058

# Kill process (replace PID)
taskkill /PID <PID> /F
```

### Missing Dependencies
```powershell
pip install --upgrade -r requirements.txt
```

### Parse Errors
- Verify log file format matches FDV OUTPUT or FDV POLL specification
- Check encoding (UTF-8 recommended)
- Look for malformed lines - parser skips these gracefully

### Plot Generation Issues
- Ensure matplotlib backend is set to 'Agg' (non-GUI)
- Check disk space in temporary directory
- Verify pandas column names are correct

## Data Privacy & Security

- 🔒 Files are stored temporarily in local temp directory
- 🧹 Implement cleanup script to remove old files (optional)
- 🚫 Don't run on untrusted networks without HTTPS
- 🔑 Change `FDV_EIMPRO_WEBAPP_SECRET` for production use

## Example Use Cases

### Analysis Scenario 1: RBER Variability by Page Type
1. Upload log file → parse
2. Create scatter plot: WL (X) vs RBER (Y), colored by pagetype
3. Create CDF plot: RBER distribution split by pagetype
4. Download CSV for further statistical analysis

### Analysis Scenario 2: Temperature Effect Analysis
1. Upload log file → parse
2. Filter data where TEMP varies
3. Create scatter plot: TEMP (X) vs value (Y), colored by testname
4. View statistics to identify temperature sensitivity

### Analysis Scenario 3: DUT Comparison
1. Upload log file → parse
2. Create CDF plot: RBER split by dut (shows performance comparison)
3. View statistics tab for detailed per-DUT metrics
4. Download CSV to generate custom reports

## Extending the App

### Add Custom Plot Type
Edit `fdv_eimpro_webapp.py` and add new route:

```python
@app.route('/api/plot/custom', methods=['POST'])
def create_custom_plot():
    # Your plot generation logic here
    pass
```

### Add New Field Extraction
Update `FDVLogParser.ALIASES` and `_extract_test_params()` method.

### Add Data Processing
Create functions in backend and expose via new API endpoints.

## Contributing

For bug reports or feature requests, please document:
- Error message or unexpected behavior
- Steps to reproduce
- Log file sample (sanitized if needed)
- Expected vs. actual behavior

## License

Internal use only - FDV Dashboard Project

## Support

For issues or questions, refer to:
1. **In-app Help Tab** - Features and usage guide
2. **guide_to_fdvlog.txt** - FDV log file format specification
3. **README_PLOT_FEATURES_v2.md** - Plot feature documentation
4. **PLOT_FEATURES_GUIDE.md** - Advanced plotting guide

---

**Version:** 1.0  
**Port:** 5058  
**Framework:** Flask + Pandas + Matplotlib  
**Last Updated:** March 2026
