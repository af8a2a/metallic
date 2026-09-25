param(
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [ValidateRange(1,1800)][double]$DurationSeconds=180,
    [ValidateRange(0,120)][double]$WarmupSeconds=5,
    [ValidateRange(1,10)][int]$Runs=3,
    [int]$Width=0,
    [int]$Height=0,
    [string]$RouteConfig='',
    [switch]$Validation,
    [switch]$NoVSync,
    [switch]$RasterComparison,
    [switch]$MetadataComparison,
    [switch]$SwComparison,
    [switch]$SwLoadComparison,
    [switch]$SwWorkComparison,
    [switch]$WorkloadCounters,
    [switch]$HistoryComparison,
    [ValidateRange(8,512)][int]$SampleFrames=64,
    [ValidateRange(2,120)][int]$SettleFrames=8,
    [ValidateRange(1,3)][int]$Rounds=3,
    [int]$TimeoutSeconds=900
)
$ErrorActionPreference='Stop'
if ($HistoryComparison) { $SwWorkComparison=$true }
if ($SwLoadComparison -or $SwWorkComparison) { $SwComparison=$true }
if ($SwLoadComparison -and $SwWorkComparison) { throw "Choose one SW comparison suite" }
if ($MetadataComparison -or $SwComparison) { $RasterComparison=$true }
if ($MetadataComparison -and $SwComparison) { throw "Choose one comparison suite" }
$repo=Split-Path -Parent $PSScriptRoot
$output=[IO.Path]::GetFullPath($OutputRoot)
$exe=Join-Path $repo 'build-release/Source/MetallicGPUDrivenSample.exe'
if (Test-Path -LiteralPath $output) { throw 'Choose a new output directory' }
if (($Width -eq 0) -ne ($Height -eq 0)) { throw 'Specify both Width and Height' }
$config=@{durationSeconds=$DurationSeconds; warmupSeconds=$WarmupSeconds}
if ($RouteConfig) {
    $inputConfig=Get-Content -LiteralPath $RouteConfig -Raw | ConvertFrom-Json
    foreach ($property in $inputConfig.PSObject.Properties) { $config[$property.Name]=$property.Value }
    $config.durationSeconds=$DurationSeconds
    $config.warmupSeconds=$WarmupSeconds
}
if ($Width) { $config.width=$Width; $config.height=$Height }
if ($RasterComparison) { $config.rasterComparison=$true; $config.sampleFrames=$SampleFrames; $config.settleFrames=$SettleFrames; $config.rounds=$Rounds }
if ($MetadataComparison) { $config.metadataComparison=$true }
if ($SwComparison) { $config.swComparison=$true }
if ($SwLoadComparison) { $config.swLoadComparison=$true }
if ($SwWorkComparison) { $config.swWorkComparison=$true }
if ($HistoryComparison) { $config.historyComparison=$true }
if ($WorkloadCounters) { $config.workloadCounters=$true; $config.workloadEvery=60 }
$analyzer=if ($config.rasterComparison) {"AnalyzeZorahFullRasterComparison.py"} else {"AnalyzeZorahFullRoam.py"}
New-Item -ItemType Directory -Path $output | Out-Null
$config | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath (Join-Path $output 'Config.json') -Encoding utf8
$keys=@('METALLIC_FULL_ROAM_OUTPUT','METALLIC_FULL_ROAM_CONFIG','METALLIC_FULL_ROAM_HIDDEN','METALLIC_FULL_ROAM_NO_VSYNC','METALLIC_NSIGHT_GRAPHICS_CAPTURE','METALLIC_DEBUG_CONTROL','METALLIC_DEBUG_VALIDATION')
$previous=@{}
foreach ($key in $keys) { $previous[$key]=[Environment]::GetEnvironmentVariable($key,'Process') }
function Get-ShaderDigest {
    $records=foreach ($file in (Get-ChildItem -LiteralPath (Join-Path $repo 'Shaders') -Filter '*.slang' -Recurse | Sort-Object FullName)) {
        "$($file.FullName.Substring($repo.Length)) $((Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash)"
    }
    $hash=[Security.Cryptography.SHA256]::Create()
    try { return [Convert]::ToHexString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes(($records -join "`n")))) }
    finally { $hash.Dispose() }
}
$manifest=@{protocol=$(if ($RasterComparison) {'zorah-full-raster-comparison-v1'} else {'zorah-full-editor-roam-v1'}); started=(Get-Date).ToString('o'); gitHead=(& git -C $repo rev-parse HEAD)
    executableSha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash; shaderSha256=(Get-ShaderDigest)
    config=$config; validation=[bool]$Validation; hidden=$true; vsync=(!$NoVSync); reflexMode=$env:METALLIC_REFLEX_MODE; streamMaintenanceBeforePacing=$env:METALLIC_STREAM_MAINTENANCE_BEFORE_PACING; runs=$Runs; gpu=(& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
    dirty=(& git -C $repo status --short)
}
$asset=Get-Item -LiteralPath (Join-Path $repo 'Asset/ZorahFull/zorah_textured_public.v1.gltf.meshstream.bin')
$manifest.asset=@{path=$asset.FullName; bytes=$asset.Length; modifiedUtc=$asset.LastWriteTimeUtc.ToString('o')}
$manifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $output 'Manifest.json') -Encoding utf8
try {
    $env:METALLIC_FULL_ROAM_CONFIG=Join-Path $output 'Config.json'
    $env:METALLIC_FULL_ROAM_HIDDEN='1'
    $env:METALLIC_FULL_ROAM_NO_VSYNC=if ($NoVSync) {'1'} else {$null}
    $env:METALLIC_NSIGHT_GRAPHICS_CAPTURE='0'
    $env:METALLIC_DEBUG_CONTROL=if ($Validation) {'1'} else {$null}
    $env:METALLIC_DEBUG_VALIDATION=if ($Validation) {'1'} else {$null}
    for ($run=1; $run -le $Runs; ++$run) {
        $directory=Join-Path $output "run$run"
        New-Item -ItemType Directory -Path $directory | Out-Null
        $env:METALLIC_FULL_ROAM_OUTPUT=$directory
        $monitor=Start-Process nvidia-smi.exe -ArgumentList @('--query-gpu=timestamp,utilization.gpu,memory.used,clocks.gr,temperature.gpu,power.draw','--format=csv','-l','1') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $directory 'Gpu.csv') -RedirectStandardError (Join-Path $directory 'Gpu.stderr.txt')
        $competition=Start-Process powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',('"'+(Join-Path $PSScriptRoot 'MeasureGpuCompetition.ps1')+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $directory 'GpuProcesses.csv') -RedirectStandardError (Join-Path $directory 'GpuProcesses.stderr.txt')
        $process=$null
        try {
            $process=Start-Process -FilePath $exe -WorkingDirectory $repo -ArgumentList @('--sample','gpu-driven-zorah-full') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $directory 'stdout.log') -RedirectStandardError (Join-Path $directory 'stderr.log')
            $started=Get-Date
            $completeAt=$null
            while (-not $process.WaitForExit(1000)) {
                if (((Get-Date)-$started).TotalSeconds -gt $TimeoutSeconds) { Stop-Process -Id $process.Id -Force; throw 'Full replay timed out' }
                if ((Test-Path -LiteralPath (Join-Path $directory 'Capture.json')) -and $null -eq $completeAt) { $completeAt=Get-Date }
                if ($completeAt -and ((Get-Date)-$completeAt).TotalSeconds -gt 20) {
                    Stop-Process -Id $process.Id -Force
                    'Process reclaimed after report export (teardown stalled)' | Set-Content -LiteralPath (Join-Path $directory 'Teardown.txt')
                    break
                }
            }
            $capture=Get-Content -LiteralPath (Join-Path $directory 'Capture.json') -Raw | ConvertFrom-Json
            if ($capture.status -ne 'capture_complete') { throw "Capture failed: $($capture.error)" }
            if (Select-String -LiteralPath (Join-Path $directory 'stdout.log'),(Join-Path $directory 'stderr.log') -Pattern 'Validation Error|VUID-|DeviceLost' -Quiet) { throw 'Validation/device error in capture' }
            python (Join-Path $PSScriptRoot $analyzer) $directory
            if ($LASTEXITCODE -ne 0) { throw 'Capture analysis failed' }
            $frameCount=if ($RasterComparison) { ($capture.cases | Measure-Object -Property frames -Sum).Sum } else { $capture.frames }
            Write-Output "Full capture $run/$Runs complete: $frameCount frames, output $($capture.outputExtent), render $($capture.renderExtent)"
        } finally {
            if ($null -ne $process -and -not $process.HasExited) { Stop-Process -Id $process.Id -Force }
            if (-not $competition.HasExited) { Stop-Process -Id $competition.Id -Force }
            if (-not $monitor.HasExited) { Stop-Process -Id $monitor.Id -Force }
        }
    }
    python (Join-Path $PSScriptRoot $analyzer) $output
    if ($LASTEXITCODE -ne 0) { throw 'Cross-run conditions differ' }
    if ((Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash -ne $manifest.executableSha256 -or (Get-ShaderDigest) -ne $manifest.shaderSha256) { throw 'Binary or shader changed during replay' }
} finally { foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') } }
