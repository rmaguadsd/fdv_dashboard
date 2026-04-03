# FDV EIMPRO WebApp - Deployment & Configuration Guide

## Project Structure

```
fdv_eimpro_webapp/
├── fdv_eimpro_webapp.py          # Main Flask application
├── __init__.py                    # Package initialization
├── requirements.txt               # Python dependencies
├── launch_eimpro_webapp.ps1      # PowerShell launcher script
├── README.md                      # Full documentation
├── QUICKSTART.md                  # Quick start guide
├── DEPLOYMENT.md                  # This file
└── templates/
    └── index.html                 # Single-page web interface
```

## Installation Steps

### 1. Prerequisites
- Python 3.7 or higher
- pip package manager
- 500 MB disk space (for dependencies + temp files)
- 2 GB RAM recommended

### 2. Install Python Dependencies

Navigate to the app directory:
```powershell
cd "d:\FDV\git\fdv_dashboard\dev\aitools\fdv_eimpro_webapp"
```

Install required packages:
```powershell
# Option A: Using pip directly
py -3 -m pip install -r requirements.txt

# Option B: Using system python
python -m pip install -r requirements.txt

# Option C: With upgrade flag (if issues)
py -3 -m pip install --upgrade -r requirements.txt
```

### 3. Verify Installation

Test that all dependencies are installed:
```powershell
py -3 -c "import flask, pandas, numpy, matplotlib, seaborn; print('All dependencies OK!')"
```

## Running the Application

### Method 1: PowerShell Launcher (Recommended)
```powershell
cd "d:\FDV\git\fdv_dashboard\dev\aitools\fdv_eimpro_webapp"
.\launch_eimpro_webapp.ps1
```

### Method 2: Direct Python
```powershell
cd "d:\FDV\git\fdv_dashboard\dev\aitools\fdv_eimpro_webapp"
py -3 fdv_eimpro_webapp.py
```

### Method 3: With Virtual Environment (Optional)
```powershell
cd "d:\FDV\git\fdv_dashboard\dev\aitools"
& .\.venv\Scripts\Activate.ps1
python fdv_eimpro_webapp.py
```

## Configuration

### Environment Variables

Set before running the app:

```powershell
# Set temporary directory
$env:FDV_EIMPRO_TMPDIR = "D:\fdv_eimpro_tmp"

# Set secret key for sessions
$env:FDV_EIMPRO_WEBAPP_SECRET = "your-secure-secret-key"

# Then launch
py -3 fdv_eimpro_webapp.py
```

### Port Configuration

To change the default port (5058), edit `fdv_eimpro_webapp.py`:

Find the line:
```python
app.run(debug=False, host='0.0.0.0', port=5058, threaded=True)
```

Change `port=5058` to your desired port number.

### Debug Mode

For development, enable debug mode in the last line:
```python
app.run(debug=True, host='0.0.0.0', port=5058, threaded=True)
```

## Accessing the Application

Once started, open your browser to:
```
http://localhost:5058
```

### Remote Access

To allow connections from other machines, the app binds to `0.0.0.0`:

- **From same machine**: http://localhost:5058
- **From network**: http://<your-machine-ip>:5058

Example network access:
```
http://192.168.1.100:5058
```

## Performance Tuning

### For Large Log Files (>100 MB)

1. **Increase RAM allocation** if available
2. **Process in chunks** - Split very large files
3. **Monitor memory usage** - Task Manager > Performance

### Parallel Processing (Future Enhancement)

The app uses multi-threading. For CPU-intensive operations:
```python
# Modify in fdv_eimpro_webapp.py
app.run(debug=False, host='0.0.0.0', port=5058, threaded=True, processes=2)
```

## Data Storage & Cleanup

### Temporary File Location

By default: `D:\fdv_eimpro_tmp`

Files stored here:
- Uploaded log files (`.txt`, `.log`)
- Generated CSV files
- Generated plot images (`.png`)

### Manual Cleanup

Remove old files:
```powershell
$tmpDir = $env:FDV_EIMPRO_TMPDIR
if (-not $tmpDir) { $tmpDir = "D:\fdv_eimpro_tmp" }

# Remove all temp files older than 7 days
Get-ChildItem $tmpDir -Force | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)} | Remove-Item -Force

# Or remove all temp files
Remove-Item "$tmpDir\*" -Force -Recurse
```

### Automatic Cleanup Script

Create `cleanup_temp.ps1`:
```powershell
# Run daily via Windows Task Scheduler
$tmpDir = "D:\fdv_eimpro_tmp"
$daysOld = 7

Get-ChildItem $tmpDir -Force -ErrorAction SilentlyContinue | 
    Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-$daysOld)} | 
    Remove-Item -Force -Recurse

Write-Host "Cleaned up files older than $daysOld days from $tmpDir"
```

## Troubleshooting

### Issue: Port Already in Use
```powershell
# Find process using port 5058
netstat -ano | findstr :5058

# Example output: TCP    0.0.0.0:5058    0.0.0.0:0    LISTENING    12345

# Kill the process (replace PID)
taskkill /PID 12345 /F
```

### Issue: Module Not Found Error
```
ImportError: No module named 'flask'
```

Solution: Reinstall dependencies
```powershell
py -3 -m pip install --force-reinstall -r requirements.txt
```

### Issue: File Permission Denied
```
PermissionError: [Errno 13] Permission denied: 'D:\\fdv_eimpro_tmp\\...'
```

Solution: Check temp directory permissions
```powershell
# Create temp directory if needed
New-Item -ItemType Directory -Path "D:\fdv_eimpro_tmp" -Force -ErrorAction SilentlyContinue

# Check ownership
icacls D:\fdv_eimpro_tmp
```

### Issue: matplotlib Backend Error
```
RuntimeError: 'Agg' not supported
```

Solution: Reinstall matplotlib
```powershell
py -3 -m pip install --force-reinstall matplotlib
```

## Monitoring

### Check if App is Running
```powershell
# Test HTTP connection
Test-NetConnection localhost -Port 5058

# Or open URL in browser
Start-Process "http://localhost:5058"
```

### View Logs

Output appears in the terminal/console. To save logs to file:
```powershell
py -3 fdv_eimpro_webapp.py | Tee-Object -FilePath "fdv_eimpro.log"
```

## Integration with Launch Script

To add to the main FDV launcher (`launch_fdv_apps.ps1`):

Edit `launch_fdv_apps.py` and add:
```python
# Add EIMPRO app alongside Report and Poll
processes.append({
    'name': 'EIMPRO',
    'cmd': python_exe,
    'args': [str(here / 'fdv_eimpro_webapp' / 'fdv_eimpro_webapp.py')],
    'port': 5058
})
```

Or create a combined launcher `launch_all_apps.ps1`:
```powershell
# Launch all three FDV apps
Start-Process powershell -ArgumentList "-NoExit", "-Command `& '.\launch_fdv_apps.ps1'"
Start-Process powershell -ArgumentList "-NoExit", "-Command `& '.\fdv_eimpro_webapp\launch_eimpro_webapp.ps1'"
Start-Process powershell -ArgumentList "-NoExit", "-Command `& 'py -3 .\fdv_portal.py'"
```

## Production Deployment

### For Production Use:

1. **Use a Production WSGI Server**:
   ```powershell
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5058 fdv_eimpro_webapp:app
   ```

2. **Enable HTTPS** (if accessing over network):
   - Use nginx as reverse proxy
   - Install SSL certificate
   - Configure proxy forwarding

3. **Set Secure Session Key**:
   ```powershell
   $env:FDV_EIMPRO_WEBAPP_SECRET = (New-Guid).Guid
   ```

4. **Disable Debug Mode**:
   - Already disabled in current code
   - Verify `debug=False` in app.run()

5. **Set Up Logging**:
   ```python
   import logging
   logging.basicConfig(
       filename='fdv_eimpro.log',
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   ```

## Backup & Recovery

### Backup Parsed Data
```powershell
$backupDate = Get-Date -Format "yyyy-MM-dd_HHmm"
Copy-Item "D:\fdv_eimpro_tmp\*.csv" "D:\Backups\fdv_eimpro_$backupDate\"
```

### Archive Old Results
```powershell
Compress-Archive -Path "D:\fdv_eimpro_tmp\*" -DestinationPath "D:\Archive\fdv_eimpro_$(Get-Date -f yyyyMMdd).zip"
```

## Security Considerations

1. **Network Access**: Only expose on trusted networks
2. **File Uploads**: Parser validates log file format
3. **Temporary Files**: Regularly clean up temp directory
4. **Session Secret**: Change default secret key
5. **Input Validation**: Parser handles malformed input gracefully

## Support & Documentation

- **Quick Start**: See `QUICKSTART.md`
- **Full Documentation**: See `README.md`
- **FDV Format Guide**: See `../guide_to_fdvlog.txt`
- **API Reference**: See `README.md` > Architecture section

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Mar 2026 | Initial release |

---

**Last Updated**: March 2026  
**Maintained By**: FDV Dashboard Team
