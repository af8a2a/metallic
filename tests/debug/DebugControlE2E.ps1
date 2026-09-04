param(
    [Parameter(Mandatory = $true)][int]$EnginePid,
    [Parameter(Mandatory = $true)][string]$CaptureDirectory,
    [string]$CtlPath = "$PSScriptRoot\..\..\build\Source\metallicctl.exe"
)

# Run against a stable GPUDriven streaming instance. This script never launches
# or terminates the user's engine, changes settings, or overwrites an export.
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
$session = $hello.result.session
$connection = @('--pid', "$EnginePid", '--session', $session)
$before = Invoke-DebugCli ($connection + @('eval', 'streaming.instances[0].stats.pendingPageCount'))
$batches = foreach ($point in @('AfterEarlyCull', 'AfterLateCull', 'AfterPass')) {
    $resources = @(@{ id = 'gpuScene.GPUDriven.visibleInstanceCounter'; count = 1 })
    if ($point -eq 'AfterPass') {
        $resources += @(
            @{ id = 'gpuScene.GPUDriven.visibleInstanceIds'; count = 1 },
            @{ id = 'streaming.GPUDriven.activeHeader'; count = 1 },
            @{ id = 'streaming.GPUDriven.activeGroups'; count = 1 },
            @{ id = 'streaming.GPUDriven.pageTable'; count = 1 },
            @{ id = 'streaming.GPUDriven.requestHeader'; count = 1 },
            @{ id = 'streaming.GPUDriven.visibleClusters'; count = 1 },
            @{ id = 'GPUDriven.visibility'; roi = @{ x = 0; y = 0; width = 8; height = 8 } }
        )
    }
    @{ pass = 'GPUDriven'; checkpoint = $point; resources = $resources }
}
$spec = [System.IO.Path]::GetTempFileName()
try {
    @{ batches = @($batches) } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $spec -Encoding utf8
    $captured = Invoke-DebugCli ($connection + @('capture', 'batch', '--spec', $spec, '--wait'))
} finally {
    Remove-Item -LiteralPath $spec -Force
}
$executions = @($captured.result.jobs | ForEach-Object {
    if ($_.status -ne 'ok' -or $_.result.state -ne 'Ready') { throw 'Capture did not become Ready' }
    $_.result.evidence.execution.value
} | Select-Object -Unique)
if ($executions.Count -ne 1) { throw 'Checkpoint group was split across executions' }
$job = $captured.result.jobs[2].result.job
$expressions = @(
    'buffers["streaming.GPUDriven.activeHeader"][0]',
    'buffers["streaming.GPUDriven.pageTable"]',
    'links["streaming.GPUDriven.activeGroups"].items[0]',
    'count(buffers["gpuScene.GPUDriven.visibleInstanceIds"], x => x >= gpuScene.stats.instanceCount)'
)
$online = foreach ($expression in $expressions) {
    Invoke-DebugCli ($connection + @('eval', $expression, '--job', $job))
}
$onlineStats = Invoke-DebugCli ($connection + @('jobs', 'get', $job, '--stats'))
$export = Invoke-DebugCli ($connection + @('capture', 'export', $job, '--out', $CaptureDirectory))
for ($index = 0; $index -lt $expressions.Count; ++$index) {
    $offline = Invoke-DebugCli @('--capture', $CaptureDirectory, 'eval', $expressions[$index])
    $expected = $online[$index].result.value | ConvertTo-Json -Depth 100 -Compress
    $actual = $offline.result.value | ConvertTo-Json -Depth 100 -Compress
    if ($expected -cne $actual) { throw "Online/offline mismatch: $($expressions[$index])" }
}
$offlineStats = Invoke-DebugCli @('--capture', $CaptureDirectory, 'stats')
if (($onlineStats.result.statistics | ConvertTo-Json -Depth 100 -Compress) -cne
    ($offlineStats.result | ConvertTo-Json -Depth 100 -Compress)) { throw 'Online/offline statistics differ' }
@{
    status = 'passed'; execution = $executions[0]; checkpointCount = 3
    expressionsCompared = $expressions.Count; statisticsMatch = $true
    pendingPageCountBeforeCapture = $before.result.value
    capture = $export.result.directory
} | ConvertTo-Json -Depth 5
