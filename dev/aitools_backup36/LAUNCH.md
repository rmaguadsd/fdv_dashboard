# FDV Apps Launcher

Use this to start both FDV apps (Report v2 and POLL) with one command.

## Options
- Report v2 runs on port 5057.
- POLL runs on port 5055.
- Ports already in use are skipped.
- Uses the workspace virtual environment at `dev/.venv` if present.

## Run from VS Code Task
Open the Command Palette and run: `Tasks: Run Task` → `Launch FDV apps`.

## Run from terminal
PowerShell (using workspace venv):

```
& "C:/Users/rmaguad/OneDrive - NANDPS/Documents/Work_mirror/dev/.venv/Scripts/python.exe" "c:/Users/rmaguad/OneDrive - NANDPS/Documents/Work_mirror/dev/aitools/launch_fdv_apps.py"
```

or using the wrapper script:

```
& "c:/Users/rmaguad/OneDrive - NANDPS/Documents/Work_mirror/dev/aitools/launch_fdv_apps.ps1"
```

## Stop
Press Ctrl+C in the launcher terminal; it will stop child processes.
