param(
    [string]$OutputRoot = "build-release/minizorah-baseline/runs",
    [string]$Executable = "build-release/tests/MetallicRhiTests.exe",
    [ValidateSet("warmup", "a1", "b1", "b2", "a2", "quality-a", "quality-b")]
    [string[]]$Cases = @("warmup", "a1", "b1", "b2", "a2", "quality-a", "quality-b")
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$outputPath = [IO.Path]::GetFullPath((Join-Path $repo $OutputRoot))
$exePath = [IO.Path]::GetFullPath((Join-Path $repo $Executable))
if (-not (Test-Path -LiteralPath $exePath)) { throw "Build MetallicRhiTests first: $exePath" }
if (Test-Path -LiteralPath (Join-Path $outputPath "Manifest.json")) {
    throw "Choose a new OutputRoot; refusing to overwrite an existing baseline manifest"
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$keys = @("METALLIC_TEST_MINIZORAH", "METALLIC_MINIZORAH_BENCH_CLAS", "METALLIC_MINIZORAH_BENCH_QUALITY", "METALLIC_MINIZORAH_BENCH_FRAMES")
$previous = @{}
foreach ($key in $keys) { $previous[$key] = [Environment]::GetEnvironmentVariable($key, "Process") }
$gpuFields = "timestamp,name,driver_version,utilization.gpu,memory.used,memory.total,clocks.gr,clocks.mem,temperature.gpu,power.draw"
$runManifest = [ordered]@{
    protocol = "minizorah-fixed-v1"
    startTime = (Get-Date).ToString("o")
    gitHead = (& git -C $repo rev-parse HEAD)
    executable = $exePath
    executableSha256 = (Get-FileHash -LiteralPath $exePath -Algorithm SHA256).Hash
    cases = $Cases
    sourceHashes = @()
}
$asset = Get-Item -LiteralPath (Join-Path $repo "Asset/MeshletCache/MiniZorahCook/MiniZorah.meshstream.bin")
$runManifest.asset = @{ path=$asset.FullName; bytes=$asset.Length; lastWriteUtc=$asset.LastWriteTimeUtc.ToString("o") }
$runManifest.hostInfo = @{
    cpu = (Get-ItemPropertyValue "HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0" -Name ProcessorNameString).Trim()
    logicalProcessors = [Environment]::ProcessorCount
    osVersion = [Environment]::OSVersion.Version.ToString()
}
foreach ($relative in @("tests/rhi/MiniZorahRoamingTests.cpp", "Pipelines/Samples/gpu_driven_minizorah_vbuffer.metallic_graph.json", "Source/Runtime/Render/Streamer/MeshletStreamCompactClasPool.cpp", "Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang", "Tools/RunMiniZorahBaseline.ps1", "Tools/AnalyzeMiniZorahBaseline.py")) {
    $runManifest.sourceHashes += @{ path = $relative; sha256 = (Get-FileHash -LiteralPath (Join-Path $repo $relative) -Algorithm SHA256).Hash }
}
$runManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $outputPath "Manifest.json") -Encoding UTF8
try {
    foreach ($case in $Cases) {
        $casePath = Join-Path $outputPath $case
        if (Test-Path -LiteralPath $casePath) { throw "Choose a new OutputRoot; refusing to overwrite $casePath" }
        New-Item -ItemType Directory -Path $casePath | Out-Null
        $env:METALLIC_TEST_MINIZORAH = "1"
        $env:METALLIC_MINIZORAH_BENCH_CLAS = if ($case -in @("a1", "a2", "quality-a")) { "0" } else { "1" }
        $env:METALLIC_MINIZORAH_BENCH_QUALITY = if ($case.StartsWith("quality")) { "1" } else { "0" }
        $env:METALLIC_MINIZORAH_BENCH_FRAMES = if ($case -eq "warmup") { "600" } else { "8400" }
        & nvidia-smi "--query-gpu=$gpuFields" --format=csv | Set-Content -LiteralPath (Join-Path $casePath "GpuBefore.csv")
        $monitor = Start-Process -FilePath "nvidia-smi.exe" -ArgumentList @("--query-gpu=$gpuFields", "--format=csv", "-l", "1") -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $casePath "GpuDuring.csv") -RedirectStandardError (Join-Path $casePath "GpuMonitor.stderr.txt")
        try {
            $validation = if ($case.StartsWith("quality") -or $case -eq "warmup") { "--rhi-validation" } else { "--rhi-no-validation" }
            $arguments = @("--gtest_filter=RhiRendering.minizorah_fixed_baseline", $validation, "--rhi-async-compute", "--output-dir", ('"' + $casePath + '"'))
            $started = Get-Date
            Write-Output "Starting $case (CLAS=$env:METALLIC_MINIZORAH_BENCH_CLAS, quality=$env:METALLIC_MINIZORAH_BENCH_QUALITY)"
            $process = Start-Process -FilePath $exePath -WorkingDirectory $repo -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $casePath "stdout.log") -RedirectStandardError (Join-Path $casePath "stderr.log")
            if (-not $process.WaitForExit(900000)) {
                Stop-Process -Id $process.Id -Force
                throw "Owned baseline process timed out: $case"
            }
            $process.Refresh()
            $exitCode = $process.ExitCode
            @{ start = $started.ToString("o"); end = (Get-Date).ToString("o"); exitCode = $exitCode } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $casePath "Process.json")
            $result = Get-Content -LiteralPath (Join-Path $casePath "Baseline.json") -Raw | ConvertFrom-Json
            if ($result.status -ne "passed" -or ($null -ne $exitCode -and $exitCode -ne 0)) { throw "$case failed: $($result.error), exit=$exitCode" }
            Write-Output "$case passed: $($result.frameCount) frames, $([math]::Round($result.runWallSeconds, 2)) seconds"
        } finally {
            if (-not $monitor.HasExited) { Stop-Process -Id $monitor.Id -Force }
            & nvidia-smi "--query-gpu=$gpuFields" --format=csv | Set-Content -LiteralPath (Join-Path $casePath "GpuAfter.csv")
        }
    }
    if ((Get-FileHash -LiteralPath $exePath -Algorithm SHA256).Hash -ne $runManifest.executableSha256) {
        throw "Executable changed during the baseline"
    }
    foreach ($source in $runManifest.sourceHashes) {
        if ((Get-FileHash -LiteralPath (Join-Path $repo $source.path) -Algorithm SHA256).Hash -ne $source.sha256) {
            throw "Source changed during the baseline: $($source.path)"
        }
    }
    $finalAsset = Get-Item -LiteralPath $asset.FullName
    if ($finalAsset.Length -ne $asset.Length -or $finalAsset.LastWriteTimeUtc -ne $asset.LastWriteTimeUtc) {
        throw "Cook changed during the baseline"
    }
} finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $previous[$key], "Process") }
}
