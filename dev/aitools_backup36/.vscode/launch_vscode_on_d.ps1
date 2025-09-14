param(
  [string]$DataDir = 'D:\\vscode-user',
  [string]$ExtDir = 'D:\\vscode-extensions',
  [string]$TmpDir = 'D:\\vscode-tmp'
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
New-Item -ItemType Directory -Path $ExtDir -Force | Out-Null
New-Item -ItemType Directory -Path $TmpDir -Force | Out-Null

# Redirect temp for this session
$env:TMP = $TmpDir
$env:TEMP = $TmpDir

# Optionally steer Copilot/NVIM/Node temp if they honor TEMP
$code = 'code'
& $code --user-data-dir $DataDir --extensions-dir $ExtDir