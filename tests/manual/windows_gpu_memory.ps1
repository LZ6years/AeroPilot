# Read-only WDDM counters. Never changes GPU policy or another process.
param([Parameter(Mandatory=$true)][string]$OutputPath)
$ErrorActionPreference = 'Stop'
$stream = [System.IO.StreamWriter]::new($OutputPath, $false, [System.Text.UTF8Encoding]::new($false))
try {
    Get-Counter -Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage','\GPU Process Memory(*)\Total Committed' -SampleInterval 1 -Continuous -ErrorAction Continue |
        ForEach-Object {
            $rows = @($_.CounterSamples | ForEach-Object {
                $value = if ([double]::IsNaN($_.CookedValue) -or [double]::IsInfinity($_.CookedValue)) { $null } else { $_.CookedValue }
                @{ instance=$_.InstanceName; path=$_.Path; bytes=$value; status=$_.Status }
            })
            $stream.WriteLine((@{ utc=$_.Timestamp.ToUniversalTime().ToString('o'); counters=$rows } | ConvertTo-Json -Depth 6 -Compress))
            $stream.Flush()
        }
} finally { $stream.Dispose() }
