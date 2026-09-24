param(
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [ValidateRange(8,10000)][int]$Frames=360,
    [ValidateRange(0,120)][double]$WarmupSeconds=5,
    [int]$Width=1280,
    [int]$Height=720,
    [double]$Distance=6,
    [string[]]$Sequence=@('p0','p1','p1','p0'),
    [switch]$IncludeDiagnostics,
    [int]$TimeoutSeconds=600
)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$output=[IO.Path]::GetFullPath($OutputRoot)
$exe=Join-Path $repo 'build-release/Source/MetallicGPUDrivenSample.exe'
if (Test-Path -LiteralPath $output) { throw 'Choose a new output directory' }
if ($Sequence | Where-Object { $_ -notin @('p0','p1') }) { throw 'Sequence accepts only p0 and p1' }
New-Item -ItemType Directory -Path $output | Out-Null
$keys=@('METALLIC_FULL_ROAM_OUTPUT','METALLIC_FULL_ROAM_CONFIG','METALLIC_FULL_ROAM_HIDDEN',
    'METALLIC_NSIGHT_GRAPHICS_CAPTURE','METALLIC_DEBUG_CONTROL','METALLIC_DEBUG_VALIDATION')
$previous=@{}
foreach ($key in $keys) { $previous[$key]=[Environment]::GetEnvironmentVariable($key,'Process') }
$manifest=@{protocol='minizorah-classify-p1-roam-v1'; started=(Get-Date).ToString('o');
    gitHead=(& git -C $repo rev-parse HEAD); dirty=(& git -C $repo status --short);
    executableSha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash;
    frames=$Frames; width=$Width; height=$Height; distance=$Distance; sequence=$Sequence;
    gpu=(& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader);
    shaders=@{}; runs=@()}
foreach ($name in @('GPUDrivenStreamAsset.slang','StreamClusterClassify.slang')) {
    $manifest.shaders[$name]=(Get-FileHash -LiteralPath (Join-Path $repo "Shaders/Features/GPUDriven/$name") -Algorithm SHA256).Hash
}
$jobs=@($Sequence | ForEach-Object { @{variant=$_; diagnostic=$false} })
if ($IncludeDiagnostics) { $jobs+=@(@{variant='p0'; diagnostic=$true},@{variant='p1'; diagnostic=$true}) }
try {
    $env:METALLIC_FULL_ROAM_HIDDEN='1'
    $env:METALLIC_NSIGHT_GRAPHICS_CAPTURE='0'
    $env:METALLIC_DEBUG_CONTROL=$null
    $env:METALLIC_DEBUG_VALIDATION=$null
    for ($index=0; $index -lt $jobs.Count; ++$index) {
        $job=$jobs[$index]
        $name=('run{0}-{1}{2}' -f ($index+1),$job.variant,$(if ($job.diagnostic) {'-diagnostic'} else {''}))
        $directory=Join-Path $output $name
        New-Item -ItemType Directory -Path $directory | Out-Null
        $config=@{sample='gpu-driven-sample'; durationSeconds=30; routeFrames=$Frames; distance=$Distance;
            warmupSeconds=$WarmupSeconds; width=$Width; height=$Height;
            cullHardwareClassification=($job.variant -eq 'p1'); metadataFastClassification=$true; temporalJitter=$false;
            workloadEvery=$(if ($job.diagnostic) {30} else {0}); classifyCounters=$true; softwareWorkloadCounters=$false}
        $configPath=Join-Path $directory 'Config.json'
        $config | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $configPath -Encoding utf8
        $env:METALLIC_FULL_ROAM_CONFIG=$configPath
        $env:METALLIC_FULL_ROAM_OUTPUT=$directory
        $monitor=Start-Process nvidia-smi.exe -ArgumentList @('--query-gpu=timestamp,utilization.gpu,memory.used,clocks.gr,temperature.gpu,power.draw','--format=csv','-l','1') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $directory 'Gpu.csv') -RedirectStandardError (Join-Path $directory 'Gpu.stderr.txt')
        $process=$null
        try {
            $process=Start-Process -FilePath $exe -WorkingDirectory $repo -ArgumentList @('--sample','gpu-driven-sample') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $directory 'stdout.log') -RedirectStandardError (Join-Path $directory 'stderr.log')
            $started=Get-Date
            $completeAt=$null
            while (-not $process.WaitForExit(1000)) {
                if (((Get-Date)-$started).TotalSeconds -gt $TimeoutSeconds) { throw 'MiniZorah profile timed out' }
                if ((Test-Path -LiteralPath (Join-Path $directory 'Capture.json')) -and $null -eq $completeAt) { $completeAt=Get-Date }
                if ($completeAt -and ((Get-Date)-$completeAt).TotalSeconds -gt 20) {
                    Stop-Process -Id $process.Id -Force
                    'Process reclaimed after report export (teardown stalled)' | Set-Content -LiteralPath (Join-Path $directory 'Teardown.txt')
                    break
                }
            }
            $capture=Get-Content -LiteralPath (Join-Path $directory 'Capture.json') -Raw | ConvertFrom-Json
            if ($capture.status -ne 'capture_complete') { throw "Capture failed: $($capture.error)" }
            if ($capture.sample -ne 'gpu-driven-sample' -or $capture.frames -ne $Frames -or $capture.missingGpuFrames -ne 0) { throw 'Wrong scene or incomplete GPU profile' }
            $manifest.runs+=@{directory=$name; variant=$job.variant; diagnostic=$job.diagnostic; frames=$capture.frames}
            $manifest | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath (Join-Path $output 'Manifest.json') -Encoding utf8
            Write-Output "$name complete: $($capture.frames) frames; render extent $($capture.renderExtent)"
        } finally {
            if ($null -ne $process -and -not $process.HasExited) { Stop-Process -Id $process.Id -Force }
            if (-not $monitor.HasExited) { Stop-Process -Id $monitor.Id -Force }
        }
    }
    if ((Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash -ne $manifest.executableSha256) { throw 'Executable changed during A/B' }
    foreach ($name in $manifest.shaders.Keys) {
        if ((Get-FileHash -LiteralPath (Join-Path $repo "Shaders/Features/GPUDriven/$name") -Algorithm SHA256).Hash -ne $manifest.shaders[$name]) { throw 'Shader changed during A/B' }
    }
} finally { foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') } }
