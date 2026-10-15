# Install a current-user task for the common frozen benchmark watchdog.
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
$Manifest = Get-Content -LiteralPath (Join-Path $ResolvedOutput 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($Manifest.strict_workers -and $Workers -ne $Manifest.strict_workers) {
    throw 'Requested workers differ from the frozen campaign configuration'
}
if ($Manifest.experiment -eq 'image_budget') {
    if ($Workers -ne 2) { throw 'Image-budget experiment requires two lanes' }
    $TaskName = 'FlyAgent_ImageBudget_' + (Split-Path -Leaf $ResolvedOutput)
}
elseif ($Manifest.dataset) {
    $TaskName = 'FlyAgent_' + $Manifest.dataset + '_' + (Split-Path -Leaf $ResolvedOutput)
}
# A WinExe entry avoids allocating a console before Python starts.
$LauncherSource = Join-Path $PSScriptRoot 'campaign_windowless_launcher.cs'
$Hasher = [Security.Cryptography.SHA256]::Create()
try { $SourceHash = [BitConverter]::ToString($Hasher.ComputeHash([IO.File]::ReadAllBytes($LauncherSource))).Replace('-', '').ToLower() }
finally { $Hasher.Dispose() }
$Launcher = Join-Path $WatchDirectory ('launcher_' + $SourceHash.Substring(0, 16) + '.exe')
$Compiler = Join-Path $env:SystemRoot 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $Launcher)) {
    & $Compiler /nologo /target:winexe /reference:System.Web.Extensions.dll "/out:$Launcher" $LauncherSource
    if ($LASTEXITCODE -ne 0) { throw 'Windowless launcher compilation failed' }
}
$LauncherConfig = Join-Path $WatchDirectory 'launcher.json'
$PythonArguments = '-u -X utf8 "' + $ControllerScript + '" watch --output "' + $ResolvedOutput + '" --workers ' + $Workers
$LaunchConfig = [ordered]@{
    interpreter = $Interpreter; arguments = $PythonArguments
    working_directory = $ResolvedOutput
    log_directory = (Join-Path $WatchDirectory 'launcher_logs')
    progress = (Join-Path $ResolvedOutput 'progress.json')
    completion_status = [string]$Manifest.completion_status
}
[IO.File]::WriteAllText($LauncherConfig, ($LaunchConfig | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
$TaskArguments = '"' + $LauncherConfig + '"'
$Identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$Action = New-ScheduledTaskAction -Execute $Launcher -Argument $TaskArguments
$Periodic = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5)
$Logon = New-ScheduledTaskTrigger -AtLogOn -User $Identity
$Settings = New-ScheduledTaskSettingsSet -Hidden -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$Principal = New-ScheduledTaskPrincipal -UserId $Identity -LogonType Interactive -RunLevel Limited
$Description = 'Frozen AerialVLN-S seen watchdog; GPUs 0/2; GPU1 forbidden; fingerprint ' + $Overlay.base_fingerprint
if ($Manifest.experiment -eq 'image_budget') {
    $Description = 'Frozen image-budget watchdog; two lanes; GPUs 0/2; fingerprint ' + $Overlay.base_fingerprint
}
elseif ($Manifest.dataset) {
    $Description = 'Frozen ' + $Manifest.dataset + ' campaign watchdog; GPUs 0/2; fingerprint ' + $Overlay.base_fingerprint
}
$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($Existing) {
    $LegacyArguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + (Join-Path $WatchDirectory 'start.ps1') + '"'
    $OwnedAction = ($Existing.Actions.Arguments -eq $TaskArguments -or $Existing.Actions.Arguments -eq $LegacyArguments)
    if ($Existing.Description -ne $Description -or -not $OwnedAction) { throw 'A different task uses this name; do not overwrite it' }
    $Backup = Join-Path $WatchDirectory ('task_before_windowless_' + (Get-Date -Format 'yyyyMMdd_HHmmss_ffff') + '.xml')
    Export-ScheduledTask -TaskName $TaskName | Set-Content -LiteralPath $Backup -Encoding UTF8
}
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger @($Periodic, $Logon) -Settings $Settings -Principal $Principal -Description $Description -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
$Installed = Get-ScheduledTask -TaskName $TaskName
$Record = [ordered]@{ task_name = $TaskName; user = $Identity; state = [string]$Installed.State; interval_seconds = 300; interpreter = $Interpreter; controller_script = $ControllerScript; launcher = $Launcher; launcher_source_sha256 = $SourceHash; launcher_config = $LauncherConfig; task_arguments = $TaskArguments; installed_at = (Get-Date).ToString('o') }
[IO.File]::WriteAllText((Join-Path $WatchDirectory 'scheduled_task.json'), ($Record | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
$Record | ConvertTo-Json -Depth 5
