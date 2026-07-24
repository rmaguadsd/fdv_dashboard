#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Register Windows Task Scheduler task for cache cleanup
    
.DESCRIPTION
    Creates a scheduled task that runs cleanup_cache.ps1 every Monday at 12 AM
    Requires admin privileges
#>

$TaskName = "FDV_Cache_Cleanup"
$TaskPath = "\FDV\Maintenance\"
$ScriptPath = "d:\FDV\git\fdv_dashboard\dev\aitools\cleanup_cache.ps1"

# Check if running as admin
$IsAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $IsAdmin) {
    Write-Host "ERROR: This script requires Administrator privileges!"
    Write-Host "Please run PowerShell as Administrator and try again."
    exit 1
}

# Verify script exists
if (-not (Test-Path $ScriptPath)) {
    Write-Host "ERROR: Cleanup script not found: $ScriptPath"
    exit 1
}

Write-Host "Creating scheduled task: $TaskName"
Write-Host "Script: $ScriptPath"
Write-Host "Schedule: Every Monday at 12:00 AM"

try {
    # Remove existing task if it exists
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Write-Host "Removing existing task..."
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
    
    # Create trigger for every Monday at 12 AM
    $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 00:00:00
    
    # Create action to run PowerShell script
    $Action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`""
    
    # Create task settings
    $Settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable:$false `
        -MultipleInstancesPolicy Queue
    
    # Register the task
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Trigger $Trigger `
        -Action $Action `
        -Settings $Settings `
        -Description "Removes FDV cache files older than 3 weeks. Runs every Monday at midnight." `
        -ErrorAction Stop
    
    Write-Host "`n✓ Task registered successfully!"
    Write-Host "`nTask Details:"
    Get-ScheduledTask -TaskName $TaskName | Format-Table -Property TaskName, @{N="NextRunTime";E={if($_.Triggers){$_.Triggers[0].StartBoundary}else{"N/A"}}}
    
} catch {
    Write-Host "ERROR: Failed to create scheduled task - $_"
    exit 1
}
