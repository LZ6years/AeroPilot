# Compatibility entry; all datasets share the same Windows installer.
[CmdletBinding()]
param(
    [string]$OutputRoot = 'H:\FlyAgent-Eval\aerialvln_s_20261007',
    [ValidateSet(1, 2)][int]$Workers = 2
)
& (Join-Path $PSScriptRoot 'install_campaign_watchdog.ps1') -OutputRoot $OutputRoot -Workers $Workers
