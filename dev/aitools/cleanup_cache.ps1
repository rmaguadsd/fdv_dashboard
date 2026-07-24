#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Cleanup old cache files from fdv_chart_cache directory
    
.DESCRIPTION
    Removes files that are more than 3 weeks (21 days) old from d:\fdv\fdv_chart_cache
    Runs every Monday at 12 AM via Windows Task Scheduler
#>

$CachePath = "d:\fdv\fdv_chart_cache"
$DaysOld = 21
$LogPath = "d:\fdv\logs\cache_cleanup.log"

# Ensure log directory exists
$LogDir = Split-Path -Parent $LogPath
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Function to log messages
function Write-Log {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] $Message"
    Write-Host $LogMessage
    Add-Content -Path $LogPath -Value $LogMessage -Encoding UTF8
}

Write-Log "=== Cache Cleanup Started ==="

# Check if cache directory exists
if (-not (Test-Path $CachePath)) {
    Write-Log "ERROR: Cache directory not found: $CachePath"
    exit 1
}

try {
    # Calculate cutoff date (3 weeks ago)
    $CutoffDate = (Get-Date).AddDays(-$DaysOld)
    Write-Log "Looking for files older than $DaysOld days (before: $($CutoffDate.ToString('yyyy-MM-dd HH:mm:ss')))"
    
    # Get all files older than 3 weeks
    $OldFiles = Get-ChildItem -Path $CachePath -File | Where-Object {
        $_.LastWriteTime -lt $CutoffDate
    }
    
    if ($OldFiles.Count -eq 0) {
        Write-Log "No files to delete. All files are newer than $DaysOld days."
    } else {
        Write-Log "Found $($OldFiles.Count) file(s) to delete:"
        
        foreach ($File in $OldFiles) {
            $Age = ((Get-Date) - $File.LastWriteTime).Days
            Write-Log "  - $($File.Name) (Age: $Age days, Size: $([Math]::Round($File.Length / 1MB, 2)) MB)"
            
            try {
                Remove-Item -Path $File.FullName -Force
                Write-Log "    ✓ Deleted"
            } catch {
                Write-Log "    ✗ ERROR: Failed to delete - $_"
            }
        }
    }
    
    # Log cache statistics
    $TotalSize = (Get-ChildItem -Path $CachePath -File | Measure-Object -Property Length -Sum).Sum
    $FileCount = (Get-ChildItem -Path $CachePath -File | Measure-Object).Count
    Write-Log "Cache statistics: $FileCount files, Total size: $([Math]::Round($TotalSize / 1GB, 2)) GB"
    
    Write-Log "=== Cache Cleanup Completed Successfully ==="
} catch {
    Write-Log "ERROR: Cleanup failed - $_"
    exit 1
}
