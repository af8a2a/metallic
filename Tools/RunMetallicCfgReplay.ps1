param(
    [Parameter(Mandatory=$true)][string]$Replay,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [string]$Executable = "build-release/tests/MetallicRhiTests.exe",
    [string[]]$Cases = @("m1", "m2", "quality")
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
$keys = @("METALLIC_TEST_MINIZORAH", "METALLIC_MINIZORAH_BENCH_CLAS", "METALLIC_MINIZORAH_BENCH_QUALITY", "METALLIC_MINIZORAH_REPLAY")
$previous = @{}
foreach ($key in $keys) { $previous[$key] = [Environment]::GetEnvironmentVariable($key, "Process") }
$manifest = @{
    protocol = $route.protocol
    executableSha256 = (Get-FileHash -LiteralPath $exePath -Algorithm SHA256).Hash
    replaySha256 = (Get-FileHash -LiteralPath $replayPath -Algorithm SHA256).Hash
    sourceSha256 = (Get-FileHash -LiteralPath (Join-Path $repo 'tests/rhi/MiniZorahRoamingTests.cpp') -Algorithm SHA256).Hash
    gitHead = (& git -C $repo rev-parse HEAD)
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
        $validation = if ($case -eq 'quality') { '--rhi-validation' } else { '--rhi-no-validation' }
        Write-Output "Starting Metallic $case on the reference camera replay"
        $monitor = Start-Process nvidia-smi.exe -ArgumentList @('--query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,clocks.gr,temperature.gpu,power.draw', '--format=csv', '-l', '1') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $casePath 'Gpu.csv') -RedirectStandardError (Join-Path $casePath 'Gpu.stderr.txt')
        try {
            $process = Start-Process -FilePath $exePath -WorkingDirectory $repo -WindowStyle Hidden -PassThru -ArgumentList @('--gtest_filter=RhiRendering.minizorah_fixed_baseline', $validation, '--rhi-async-compute', '--output-dir', ('"'+$casePath+'"')) -RedirectStandardOutput (Join-Path $casePath 'stdout.log') -RedirectStandardError (Join-Path $casePath 'stderr.log')
            if (-not $process.WaitForExit(600000)) { Stop-Process -Id $process.Id -Force; throw "Metallic replay timed out" }
            $process.Refresh()
            $exitCode = $process.ExitCode
            # Some Windows PowerShell hosts do not retain ExitCode after WaitForExit.
            if ($null -ne $exitCode -and $exitCode -ne 0) { throw "$case exited with code $exitCode" }
            $report = Get-Content -LiteralPath (Join-Path $casePath 'Baseline.json') -Raw | ConvertFrom-Json
            if ($report.status -ne 'passed') { throw "$case failed: $($report.error)" }
            Write-Output "$case passed: $($report.frameCount) frames"
        } finally {
            if (-not $monitor.HasExited) { Stop-Process -Id $monitor.Id -Force }
        }
    }
    if ((Get-FileHash -LiteralPath $exePath -Algorithm SHA256).Hash -ne $manifest.executableSha256) { throw "Binary changed during replay" }
} finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process') }
}
