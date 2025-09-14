Drive D configuration

- VS Code (Copilot caches): use .vscode/launch_vscode_on_d.ps1 to start Code with user data and extensions on D:.
- FDV Report v2: honors env FDV_REPORT2_TMPDIR; defaults to D:\\fdv_tmp with fallbacks.
- FDV POLL: honors env FDV_POLL_TMPDIR; defaults to D:\\fdv_tmp. We set tempfile.tempdir accordingly at app startup.

One-time prep

- Ensure D:\\fdv_tmp exists, or let the apps create it on first run.

Launch VS Code on D:

PowerShell

  ./.vscode/launch_vscode_on_d.ps1

Override temp dirs (optional)

PowerShell

  $env:FDV_REPORT2_TMPDIR = 'D:\\fdv_tmp'
  $env:FDV_POLL_TMPDIR = 'D:\\fdv_tmp'
