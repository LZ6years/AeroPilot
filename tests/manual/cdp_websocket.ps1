# Native .NET WebSocket transport for the local review-only CDP probe.
# JSON-lines control avoids shell interpolation of CDP messages or page scripts.
param([Parameter(Mandatory = $true)][string]$Endpoint)
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$socket = [System.Net.WebSockets.ClientWebSocket]::new()
function Reply($value) {
    [Console]::WriteLine(($value | ConvertTo-Json -Compress -Depth 10))
    [Console]::Out.Flush()
}
try {
    $deadline = [System.Threading.CancellationTokenSource]::new(30000)
    try { $null = $socket.ConnectAsync([Uri]$Endpoint, $deadline.Token).GetAwaiter().GetResult() }
    finally { $deadline.Dispose() }
    Reply @{ ready = $true }
    while ($null -ne ($line = [Console]::ReadLine())) {
        $request = $line | ConvertFrom-Json
        if ($request.op -eq 'close') { Reply @{ ok = $true }; break }
        $deadline = [System.Threading.CancellationTokenSource]::new(30000)
        try {
            if ($request.op -eq 'send') {
                $bytes = [System.Text.Encoding]::UTF8.GetBytes([string]$request.payload)
                $null = $socket.SendAsync([ArraySegment[byte]]::new($bytes), [System.Net.WebSockets.WebSocketMessageType]::Text, $true, $deadline.Token).GetAwaiter().GetResult()
                Reply @{ ok = $true }
            } elseif ($request.op -eq 'recv') {
                $buffer = [byte[]]::new(16384)
                $message = [System.IO.MemoryStream]::new()
                try {
                    do {
                        $part = $socket.ReceiveAsync([ArraySegment[byte]]::new($buffer), $deadline.Token).GetAwaiter().GetResult()
                        if ($part.MessageType -eq [System.Net.WebSockets.WebSocketMessageType]::Close) { throw 'CDP socket closed' }
                        $message.Write($buffer, 0, $part.Count)
                    } while (-not $part.EndOfMessage)
                    Reply @{ data = [System.Text.Encoding]::UTF8.GetString($message.ToArray()) }
                } finally { $message.Dispose() }
            } else { throw 'Unknown transport operation' }
        } finally { $deadline.Dispose() }
    }
} catch {
    Reply @{ error = $_.Exception.Message }
    exit 1
} finally { $socket.Dispose() }
