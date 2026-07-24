# FDV Cache Cleanup Setup

This directory contains scripts for automatically cleaning up old FDV cache files.

## Overview

- **cleanup_cache.ps1** - PowerShell script that removes cache files older than 3 weeks
- **register_cache_cleanup_task.ps1** - Registers the cleanup script as a Windows scheduled task

## Setup Instructions

### Prerequisites
- Windows PowerShell 5.0 or later (or PowerShell Core)
- Administrator privileges to register scheduled task
- The directory `d:\fdv\fdv_chart_cache` must exist (or modify the path in cleanup_cache.ps1)

### Step 1: Run the Registration Script

Open PowerShell **as Administrator** and run:

```powershell
cd "d:\FDV\git\fdv_dashboard\dev\aitools"
.\register_cache_cleanup_task.ps1
```

This will:
1. Create a scheduled task named `FDV_Cache_Cleanup`
2. Configure it to run every Monday at 12:00 AM
3. Display confirmation with the task details

### Step 2: Verify the Task

To check if the task was created successfully:

```powershell
Get-ScheduledTask -TaskName "FDV_Cache_Cleanup" | Format-List
```

### Manual Cleanup

To manually run the cleanup script without waiting for Monday:

```powershell
.\cleanup_cache.ps1
```

## Configuration

### Change the Cleanup Schedule

Edit `register_cache_cleanup_task.ps1` and modify the `$Trigger` line:

```powershell
# Current: Every Monday at midnight
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 00:00:00

# Examples:
# Every day at 2 AM:
$Trigger = New-ScheduledTaskTrigger -Daily -At 02:00:00

# Every Friday at 11 PM:
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At 23:00:00
```

### Change the Age Threshold

Edit `cleanup_cache.ps1` and modify the `$DaysOld` variable:

```powershell
# Current: 3 weeks (21 days)
$DaysOld = 21

# Examples:
$DaysOld = 14  # 2 weeks
$DaysOld = 30  # 1 month
```

### Change the Cache Directory

Edit `cleanup_cache.ps1` and modify the `$CachePath` variable:

```powershell
# Current:
$CachePath = "d:\fdv\fdv_chart_cache"

# Example:
$CachePath = "e:\cache\fdv_charts"
```

## Logging

The cleanup script logs all operations to: `d:\fdv\logs\cache_cleanup.log`

Each run includes:
- Timestamp of when cleanup ran
- Files deleted and their age
- Total cache size and file count
- Any errors encountered

### View Recent Logs

```powershell
Get-Content "d:\fdv\logs\cache_cleanup.log" -Tail 50  # Last 50 lines
```

## Task Management

### View All FDV Tasks

```powershell
Get-ScheduledTask -TaskName "FDV*"
```

### Disable the Task (Without Deleting)

```powershell
Disable-ScheduledTask -TaskName "FDV_Cache_Cleanup"
```

### Re-enable the Task

```powershell
Enable-ScheduledTask -TaskName "FDV_Cache_Cleanup"
```

### Remove the Task

```powershell
Unregister-ScheduledTask -TaskName "FDV_Cache_Cleanup" -Confirm:$false
```

### View Last Run Results

```powershell
Get-ScheduledTaskInfo -TaskName "FDV_Cache_Cleanup"
```

## Troubleshooting

### Task Not Running

1. Check if the task is enabled:
   ```powershell
   Get-ScheduledTask -TaskName "FDV_Cache_Cleanup" | Select-Object State
   ```

2. Verify the script path exists:
   ```powershell
   Test-Path "d:\FDV\git\fdv_dashboard\dev\aitools\cleanup_cache.ps1"
   ```

3. Check the Windows Event Viewer for task errors:
   - Event Viewer → Windows Logs → System
   - Filter for "Task Scheduler" events

### Permission Denied Errors

Make sure you run the registration script with Administrator privileges. If you get permission errors when deleting files, ensure:
1. You have write permissions to `d:\fdv\fdv_chart_cache`
2. No other processes are locking the files
3. The files aren't read-only

### Log File Not Creating

Ensure the log directory exists:
```powershell
New-Item -ItemType Directory -Path "d:\fdv\logs" -Force
```

## Safety

- The script only deletes files older than 3 weeks - recent cache is always preserved
- A log entry is created for each deleted file
- The script won't delete files it can't access (they'll be logged as errors)
- To test without deleting, comment out the `Remove-Item` line in cleanup_cache.ps1

## Additional Notes

- Files are identified by `LastWriteTime`, not `CreationTime`
- If the cache directory doesn't exist, the script logs an error but doesn't fail
- Cache statistics are logged after each cleanup run
- The task runs with the system account, so it may have different permissions than manual runs
