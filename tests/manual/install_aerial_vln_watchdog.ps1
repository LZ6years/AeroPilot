# Install a current-user task that relaunches the hidden frozen seen watchdog.
# No credentials or administrator elevation are requested. The Python watchdog
# has its own OS lock and verifies the frozen campaign before spawning work.
[CmdletBinding()]
param(
    [string]$OutputRoot = 'H:\FlyAgent-Eval\aerialvln_s_20261007',
    [ValidateSet(1, 2)][int]$Workers = 2
)
$ErrorActionPreference = 'Stop'
$Interpreter = 'F:\conda_env\flyagent\python.exe'
if (-not (Test-Path -LiteralPath $Interpreter -PathType Leaf)) { throw 'flyagent interpreter missing' }
$ResolvedOutput = (Resolve-Path -LiteralPath $OutputRoot).Path
$Overlay = Get-Content -LiteralPath (Join-Path $ResolvedOutput 'parallel\execution_overlay.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$ControllerScript = (Resolve-Path -LiteralPath $Overlay.controller_script).Path
$SnapshotRoot = [IO.Path]::GetFullPath((Join-Path $ResolvedOutput 'parallel\controller_source')) + '\'
if (-not $ControllerScript.StartsWith($SnapshotRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The task must use a frozen parallel controller snapshot'
}
$WatchDirectory = Join-Path $ResolvedOutput 'parallel\watchdog'
New-Item -ItemType Directory -Path $WatchDirectory -Force | Out-Null
$TaskName = 'FlyAgent_AerialVLNS_seen_' + (Split-Path -Leaf $ResolvedOutput)
$Launcher = Join-Path $WatchDirectory 'start.ps1'
$PowerShellExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$TaskArguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $Launcher + '"'
function PSQuote([string]$Value) { return "'" + $Value.Replace("'", "''") + "'" }
$LaunchCode = @(
    "`$ErrorActionPreference = 'Stop'",
    "`$Progress = $(PSQuote (Join-Path $ResolvedOutput 'progress.json'))",
    "`$TaskName = $(PSQuote $TaskName)",
    "`$TaskArguments = $(PSQuote $TaskArguments)",
    'try { $State = Get-Content -LiteralPath $Progress -Raw -Encoding UTF8 | ConvertFrom-Json } catch { $State = $null }',
    "if (`$State -and `$State.status -eq 'seen_complete') {",
    '  $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue',
    '  if ($Task -and $Task.Actions.Arguments -eq $TaskArguments) { Disable-ScheduledTask -TaskName $TaskName | Out-Null }',
    '  return',
    '}',
    "Start-Process -FilePath $(PSQuote $Interpreter) -ArgumentList @('-u', '-X', 'utf8', $(PSQuote ('"' + $ControllerScript + '"')), 'watch', '--output', $(PSQuote ('"' + $ResolvedOutput + '"')), '--workers', '$Workers') -WindowStyle Hidden"
) -join [Environment]::NewLine
[IO.File]::WriteAllText($Launcher, $LaunchCode, [Text.UTF8Encoding]::new($true))
$Identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$Action = New-ScheduledTaskAction -Execute $PowerShellExe -Argument $TaskArguments
$Periodic = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5)
$Logon = New-ScheduledTaskTrigger -AtLogOn -User $Identity
$Settings = New-ScheduledTaskSettingsSet -Hidden -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$Principal = New-ScheduledTaskPrincipal -UserId $Identity -LogonType Interactive -RunLevel Limited
$Description = 'Frozen AerialVLN-S seen watchdog; GPUs 0/2; GPU1 forbidden; fingerprint ' + $Overlay.base_fingerprint
$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($Existing -and ($Existing.Description -ne $Description -or $Existing.Actions.Arguments -ne $TaskArguments)) {
    throw 'A different task uses this name; do not overwrite it'
}
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger @($Periodic, $Logon) -Settings $Settings -Principal $Principal -Description $Description -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
$Installed = Get-ScheduledTask -TaskName $TaskName
$Record = [ordered]@{ task_name = $TaskName; user = $Identity; state = [string]$Installed.State; interval_seconds = 300; interpreter = $Interpreter; controller_script = $ControllerScript; launcher = $Launcher; task_arguments = $TaskArguments; installed_at = (Get-Date).ToString('o') }
[IO.File]::WriteAllText((Join-Path $WatchDirectory 'scheduled_task.json'), ($Record | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
$Record | ConvertTo-Json -Depth 5
