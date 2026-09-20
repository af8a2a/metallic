param(
    [string]$OutputDirectory = 'build-release/zorah-z5/first-frame',
    [int]$Cycles = 2,
    [int]$TimeoutSeconds = 1800,
    [switch]$Validation = $true
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not [IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory = Join-Path $repo $OutputDirectory }
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$executable = Join-Path $repo 'build-release/tests/MetallicRhiTests.exe'
$cache = Join-Path $repo 'Asset/ZorahFull/zorah_textured_public.v1.gltf.meshstream.bin'
if (-not (Test-Path -LiteralPath $cache) -or (Test-Path -LiteralPath "$cache.partial")) {
    throw 'ZorahFull cook is missing or incomplete. Finish MetallicMeshletCook before the first-frame run.'
}
$oldEnabled = $env:METALLIC_TEST_ZORAH_FULL
$oldCycles = $env:METALLIC_ZORAH_FULL_CYCLES
$env:METALLIC_TEST_ZORAH_FULL = '1'
$env:METALLIC_ZORAH_FULL_CYCLES = [string]$Cycles
try {
    $arguments = @('--gtest_filter=RhiRendering.zorah_full_first_frame', '--output-dir',
        ('"' + $OutputDirectory + '"'), ('"--gtest_output=json:' + (Join-Path $OutputDirectory 'rhi.json') + '"'))
    if ($Validation) { $arguments += '--rhi-validation' } else { $arguments += '--rhi-no-validation' }
    $gpuQuery = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    $watch = [Diagnostics.Stopwatch]::StartNew()
    Remove-Item -LiteralPath (Join-Path $OutputDirectory 'ZorahFullFirstFrame.json') -ErrorAction SilentlyContinue
    $process = Start-Process -FilePath $executable -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $OutputDirectory 'stdout.log') -RedirectStandardError (Join-Path $OutputDirectory 'stderr.log')
    # Windows PowerShell can otherwise lose the exit code after Refresh/HasExited.
    # Retain the native process handle while collecting asynchronous telemetry.
    $null = $process.Handle
    $telemetry = [IO.StreamWriter]::new((Join-Path $OutputDirectory 'process.jsonl'), $false)
    try {
        while (-not $process.WaitForExit(1000)) {
            $process.Refresh()
            if ($process.HasExited) { break }
            $sample = [ordered]@{ elapsedSeconds=$watch.Elapsed.TotalSeconds; privateBytes=$process.PrivateMemorySize64;
                workingSetBytes=$process.WorkingSet64; cpuSeconds=$process.TotalProcessorTime.TotalSeconds }
            if ($gpuQuery) {
                # Adapter-wide occupancy includes the desktop and other processes.
                $sample.gpuAdapter = @(& $gpuQuery.Source --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
            }
            $telemetry.WriteLine(($sample | ConvertTo-Json -Compress)); $telemetry.Flush()
            if ($watch.Elapsed.TotalSeconds -gt $TimeoutSeconds) {
                Stop-Process -Id $process.Id
                throw "Full first-frame run exceeded $TimeoutSeconds seconds; see stdout.log and process.jsonl"
            }
        }
        $process.WaitForExit()
        if ($process.ExitCode -ne 0) { throw "Full first-frame test failed ($($process.ExitCode)); see $OutputDirectory" }
        $report = Get-Content -LiteralPath (Join-Path $OutputDirectory 'ZorahFullFirstFrame.json') -Raw | ConvertFrom-Json
        if ($report.status -ne 'passed') { throw 'Full first-frame evidence was not completed' }
        $report.runs | Select-Object cycle,firstReadyFrame,firstReadySeconds,totalSeconds,retired
    } finally { $telemetry.Dispose() }
} finally {
    $env:METALLIC_TEST_ZORAH_FULL = $oldEnabled
    $env:METALLIC_ZORAH_FULL_CYCLES = $oldCycles
}
