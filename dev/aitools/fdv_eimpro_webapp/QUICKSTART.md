# FDV EIMPRO WebApp - Quick Start Guide

## Installation (One-time setup)

### 1. Install Dependencies
```powershell
cd d:\FDV\git\fdv_dashboard\dev\aitools\fdv_eimpro_webapp
pip install -r requirements.txt
```

Or with Python 3:
```powershell
py -3 -m pip install -r requirements.txt
```

## Launch the Application

### Option 1: Using PowerShell Launcher Script (Recommended)
```powershell
cd d:\FDV\git\fdv_dashboard\dev\aitools\fdv_eimpro_webapp
.\launch_eimpro_webapp.ps1
```

### Option 2: Direct Python Execution
```powershell
cd d:\FDV\git\fdv_dashboard\dev\aitools\fdv_eimpro_webapp
python fdv_eimpro_webapp.py
```

### Option 3: Python 3 Launcher
```powershell
cd d:\FDV\git\fdv_dashboard\dev\aitools\fdv_eimpro_webapp
py -3 fdv_eimpro_webapp.py
```

## Access the Web Application

Once the app starts, open your browser and go to:
```
http://localhost:5058
```

You should see the FDV EIMPRO Log Parser & Visualizer interface.

## Basic Workflow

### Step 1: Parse Your Log File
1. Click the **"📤 Upload & Parse"** tab
2. Select your FDV log file (.txt or .log)
3. Click **"Parse File"**
4. View the preview and download the CSV

### Step 2: Explore Your Data
1. Click the **"📋 View Data"** tab
2. Browse through the parsed records
3. Use filters to find specific records
4. View statistics in the **"📉 Statistics"** tab

### Step 3: Create Visualizations
1. **Scatter Plots**: Click **"📈 Scatter Plot"** tab
   - Select X and Y columns
   - Optionally color by category
   - Click "Generate Plot"

2. **CDF Plots**: Click **"📊 CDF Plot"** tab
   - Select value column (distribution)
   - Select category column (colors)
   - Optionally split by another column
   - Click "Generate Plot"

### Step 4: Export Results
- Download CSV from Upload tab
- Right-click plots to save as PNG
- Use data for reports or further analysis

## Keyboard Shortcuts

- `Tab` key - Switch between tabs
- `Ctrl+C` in terminal - Stop the server

## Troubleshooting

### Port 5058 Already in Use
Find and kill the process:
```powershell
netstat -ano | findstr :5058
taskkill /PID <PID> /F
```

### Dependencies Not Found
Reinstall all packages:
```powershell
py -3 -m pip install --upgrade -r requirements.txt
```

### Parse Errors on Upload
- Verify your log file is in FDV format
- Check for UTF-8 encoding
- Try a smaller sample file first

## Data Cleanup

To clean up old temporary files:
```powershell
$tmpDir = "D:\fdv_eimpro_tmp"
if (Test-Path $tmpDir) {
    Get-ChildItem $tmpDir -Force | Remove-Item -Force -Recurse
}
```

## Need Help?

1. Click the **"ℹ️ Help"** tab in the app for full documentation
2. Refer to `README.md` for detailed feature descriptions
3. Check `guide_to_fdvlog.txt` for log file format specifications

## Tips & Tricks

### For Large Log Files
- Parser handles files up to several GB
- Data is loaded into memory, so ensure sufficient RAM
- Consider splitting very large files

### For Better Visualizations
- Use "Color By" column in scatter plots for categorical insights
- Split CDF plots by DUT to compare device performance
- Export plots as PNG for presentations

### Data Analysis Workflow
```
Upload Log → Parse → View CSV → Filter Data → Generate Plots → Export Results
```

---

**Quick Reference:**
- **Port**: 5058
- **URL**: http://localhost:5058
- **Temp Directory**: D:\fdv_eimpro_tmp
- **Supported Formats**: FDV OUTPUT, FDV POLL
- **Output Types**: CSV, PNG plots
