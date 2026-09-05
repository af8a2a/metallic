param(
    [Parameter(Mandatory = $true)][int]$EnginePid,
    [Parameter(Mandatory = $true)][string]$CaptureDirectory,
    [string]$CtlPath = "$PSScriptRoot\..\..\build\Source\metallicctl.exe"
)

# A stable GPUDriven streaming instance must already be running with debug control.
# This test launches no engine and changes no scene/render settings.
$ErrorActionPreference = 'Stop'
$CtlPath = (Resolve-Path -LiteralPath $CtlPath).Path
function Invoke-DebugCli([string[]]$Arguments) {
    $text = & $CtlPath --json @Arguments
    if ($LASTEXITCODE -ne 0) { throw "metallicctl failed: $text" }
    $response = $text | ConvertFrom-Json -AsHashtable
    if ($response.status -ne 'ok') { throw "Debug request failed: $text" }
    return $response
}
$hello = Invoke-DebugCli @('--pid', "$EnginePid", 'hello')
$connection = @('--pid', "$EnginePid", '--session', $hello.result.session)
$specPath = (Resolve-Path -LiteralPath "$PSScriptRoot\..\..\Documentation\DebugProbe.example.json").Path
$response = Invoke-DebugCli ($connection + @('probe', '--spec', $specPath, '--wait'))
if ($response.result.state -ne 'Ready') { throw 'Probe was not completed' }
$job = $response.result.job
$checks = @(
    'probes.requestErrors.max == buffers["streaming.GPUDriven.requestHeader"][0].invalidPageCounter',
    'probes.pageStates.min == buffers["streaming.GPUDriven.pageTable"][0].state.value'
)
foreach ($expression in $checks) {
    $result = Invoke-DebugCli ($connection + @('eval', $expression, '--job', $job))
    if ($result.result.value -ne $true) { throw "GPU probe differs from raw same-boundary evidence: $expression" }
}
$expressions = @('probes.requestErrors', 'probes.pageStates', 'buffers["streaming.GPUDriven.requestHeader"][0]')
$online = foreach ($expression in $expressions) {
    Invoke-DebugCli ($connection + @('eval', $expression, '--job', $job))
}
$export = Invoke-DebugCli ($connection + @('capture', 'export', $job, '--out', $CaptureDirectory))
for ($index = 0; $index -lt $expressions.Count; ++$index) {
    $offline = Invoke-DebugCli @('--capture', $CaptureDirectory, 'eval', $expressions[$index])
    $expected = $online[$index].result.value | ConvertTo-Json -Depth 100 -Compress
    $actual = $offline.result.value | ConvertTo-Json -Depth 100 -Compress
    if ($expected -cne $actual) { throw "Online/offline mismatch: $($expressions[$index])" }
}
$onlineStats = Invoke-DebugCli ($connection + @('jobs', 'get', $job, '--stats'))
$offlineStats = Invoke-DebugCli @('--capture', $CaptureDirectory, 'stats')
if (($onlineStats.result.statistics | ConvertTo-Json -Depth 100 -Compress) -cne
    ($offlineStats.result | ConvertTo-Json -Depth 100 -Compress)) { throw 'Probe statistics differ offline' }
$watchSpec = [System.IO.Path]::GetTempFileName()
$watch = $null
try {
    # Deterministically retain the first completed sample, including a healthy one.
    # Anomaly-only production watches normally use gt 0 instead.
    @{
        probe = (Get-Content -LiteralPath $specPath -Raw | ConvertFrom-Json -AsHashtable)
        everyExecutions = 2; maxSamples = 2
        trigger = @{ probe = 'requestErrors'; field = 'matchedCount'; op = 'ge'; value = 0 }
    } | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $watchSpec -Encoding utf8
    $created = Invoke-DebugCli ($connection + @('watch', 'create', '--spec', $watchSpec))
    $watch = $created.result.watch
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    do {
        $status = Invoke-DebugCli ($connection + @('watch', 'get', $watch))
        if ($status.result.state -ne 'Active') { break }
        Start-Sleep -Milliseconds 50
    } while ([DateTime]::UtcNow -lt $deadline)
    if ($status.result.state -ne 'Triggered') { throw "Watch did not trigger: $($status | ConvertTo-Json -Depth 10 -Compress)" }
    $evidence = Invoke-DebugCli ($connection + @('jobs', 'get', $status.result.job))
    if ($evidence.result.evidence.execution.value -ne $status.result.result.evidence.execution.value) {
        throw 'Watch job and trigger refer to different executions'
    }
} finally {
    if ($null -ne $watch) { $null = Invoke-DebugCli ($connection + @('watch', 'delete', $watch)) }
    Remove-Item -LiteralPath $watchSpec -Force
}
@{
    status = 'passed'; execution = $response.result.evidence.execution.value
    expressionsCompared = $expressions.Count; rawComparisons = $checks.Count
    statisticsMatch = $true; watchTriggered = $true; capture = $export.result.directory
} | ConvertTo-Json -Depth 5
