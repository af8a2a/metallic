param(
    [Parameter(Mandatory=$true)][string]$Replay,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [string]$Executable = "build-release/tests/MetallicRhiTests.exe",
    [string[]]$Cases = @("m1", "m2", "quality"),
    [switch]$Realtime,
    [switch]$QualityWithoutValidation,
    [int]$TimeoutSeconds = 900
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$replayPath = [IO.Path]::GetFullPath($Replay)
$outputPath = [IO.Path]::GetFullPath($OutputRoot)
$exePath = if ([IO.Path]::IsPathRooted($Executable)) { [IO.Path]::GetFullPath($Executable) } else { [IO.Path]::GetFullPath((Join-Path $repo $Executable)) }
if (Test-Path -LiteralPath $outputPath) { throw "Choose a new output directory" }
$route = Get-Content -LiteralPath $replayPath -Raw | ConvertFrom-Json
if ($route.protocol -ne "minizorah-cfg-roam-v1") { throw "Unexpected replay protocol" }
New-Item -ItemType Directory -Path $outputPath | Out-Null
$keys = @("METALLIC_TEST_MINIZORAH", "METALLIC_MINIZORAH_BENCH_CLAS", "METALLIC_MINIZORAH_BENCH_QUALITY", "METALLIC_MINIZORAH_REPLAY", "METALLIC_MINIZORAH_BENCH_REALTIME")
$previous = @{}
foreach ($key in $keys) { $previous[$key] = [Environment]::GetEnvironmentVariable($key, "Process") }
function Get-ShaderTreeDigest {
    $records = foreach ($relative in (& rg --files (Join-Path $repo 'Shaders') -g '*.slang' | Sort-Object)) {
        $name = $relative.Substring($repo.Length).Replace('\', '/')
        "$name $((Get-FileHash -LiteralPath $relative -Algorithm SHA256).Hash)"
    }
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes(($records -join "`n"))))).Replace('-', '')
    } finally { $hasher.Dispose() }
}
$manifest = @{
    protocol = $route.protocol
    executableSha256 = (Get-FileHash -LiteralPath $exePath -Algorithm SHA256).Hash
    replaySha256 = (Get-FileHash -LiteralPath $replayPath -Algorithm SHA256).Hash
    sourceSha256 = (Get-FileHash -LiteralPath (Join-Path $repo 'tests/rhi/MiniZorahRoamingTests.cpp') -Algorithm SHA256).Hash
    shaderTreeSha256 = Get-ShaderTreeDigest
    gitHead = (& git -C $repo rev-parse HEAD)
    realtime = [bool]$Realtime
    qualityWithoutValidation = [bool]$QualityWithoutValidation
    start = (Get-Date).ToString('o')
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath 'Manifest.json') -Encoding UTF8
try {
    foreach ($case in $Cases) {
        if ($case -notin @('m1', 'm2', 'quality')) { throw "Unknown case: $case" }
        $casePath = Join-Path $outputPath $case
        New-Item -ItemType Directory -Path $casePath | Out-Null
        $env:METALLIC_TEST_MINIZORAH = '1'
        $env:METALLIC_MINIZORAH_BENCH_CLAS = '1'
        $env:METALLIC_MINIZORAH_BENCH_QUALITY = if ($case -eq 'quality') { '1' } else { '0' }
        $env:METALLIC_MINIZORAH_REPLAY = $replayPath
        $env:METALLIC_MINIZORAH_BENCH_REALTIME = if ($Realtime) { '1' } else { '0' }
        $validation = if ($case -eq 'quality' -and -not $QualityWithoutValidation) { '--rhi-validation' } else { '--rhi-no-validation' }
        Write-Output "Starting Metallic $case on the reference camera replay"
        $monitor = Start-Process nvidia-smi.exe -ArgumentList @('--query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,clocks.gr,temperature.gpu,power.draw', '--format=csv', '-l', '1') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $casePath 'Gpu.csv') -RedirectStandardError (Join-Path $casePath 'Gpu.stderr.txt')
        $competitionMonitor = $null
        try {
            $competitionMonitor = Start-Process powershell.exe -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass',
                '-File', ('"' + (Join-Path $PSScriptRoot 'MeasureGpuCompetition.ps1') + '"')) -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput (Join-Path $casePath 'GpuProcesses.csv') -RedirectStandardError (Join-Path $casePath 'GpuProcesses.stderr.txt')
            $arguments = @('--gtest_filter=RhiRendering.minizorah_fixed_baseline', $validation, '--rhi-async-compute', '--output-dir', ('"'+$casePath+'"'))
            if ($Realtime) { $arguments += '--rhi-realtime' }
            $process = Start-Process -FilePath $exePath -WorkingDirectory $repo -WindowStyle Hidden -PassThru -ArgumentList $arguments -RedirectStandardOutput (Join-Path $casePath 'stdout.log') -RedirectStandardError (Join-Path $casePath 'stderr.log')
            $started = Get-Date
            $captureCompleted = $null
            $forcedCleanup = $false
            while (-not $process.WaitForExit(1000)) {
                if (((Get-Date) - $started).TotalSeconds -gt $TimeoutSeconds) {
                    Stop-Process -Id $process.Id -Force
                    throw "Metallic replay timed out before normal exit"
                }
                # Streamline may stall at process teardown. Only reclaim our own
                # process after the complete report and terminal test result are flushed.
                $reportPath = Join-Path $casePath 'Baseline.json'
                if ($null -eq $captureCompleted -and (Test-Path -LiteralPath $reportPath)) {
                    try {
                        $candidateReport = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
                        $testFinished = Select-String -LiteralPath (Join-Path $casePath 'stdout.log') -SimpleMatch 'Global test environment tear-down' -Quiet
                        if ($candidateReport.status -in @('passed', 'failed') -and $testFinished) { $captureCompleted = Get-Date }
                    } catch { } # The report may still be being written.
                }
                if ($null -ne $captureCompleted -and ((Get-Date) - $captureCompleted).TotalSeconds -ge 8) {
                    $forcedCleanup = $true
                    Stop-Process -Id $process.Id -Force
                    $process.WaitForExit()
                    break
                }
            }
            $process.Refresh()
            $exitCode = $process.ExitCode
            # Some Windows PowerShell hosts do not retain ExitCode after WaitForExit.
            @{ exitCode = $exitCode; forcedCleanupAfterCapture = $forcedCleanup; arguments = $arguments } |
                ConvertTo-Json | Set-Content -LiteralPath (Join-Path $casePath 'Process.json') -Encoding UTF8
            if (-not $forcedCleanup -and $null -ne $exitCode -and $exitCode -ne 0) { throw "$case exited with code $exitCode" }
            $report = Get-Content -LiteralPath (Join-Path $casePath 'Baseline.json') -Raw | ConvertFrom-Json
            if ($report.status -ne 'passed') { throw "$case failed: $($report.error)" }
            Write-Output "$case passed: $($report.frameCount) frames"
        } finally {
            if (-not $monitor.HasExited) { Stop-Process -Id $monitor.Id -Force }
            if ($null -ne $competitionMonitor -and -not $competitionMonitor.HasExited) { Stop-Process -Id $competitionMonitor.Id -Force }
        }
    }
    if ((Get-FileHash -LiteralPath $exePath -Algorithm SHA256).Hash -ne $manifest.executableSha256) { throw "Binary changed during replay" }
    if ((Get-ShaderTreeDigest) -ne $manifest.shaderTreeSha256) { throw "Shaders changed during replay" }
    if ((Get-FileHash -LiteralPath (Join-Path $repo 'tests/rhi/MiniZorahRoamingTests.cpp') -Algorithm SHA256).Hash -ne $manifest.sourceSha256) { throw "Harness source changed during replay" }
} finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process') }
}
