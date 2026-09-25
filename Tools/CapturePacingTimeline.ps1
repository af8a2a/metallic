param(
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [ValidateRange(1,4)][int]$Sessions=1,
    [ValidateRange(30,300)][int]$SessionTimeoutSeconds=180
)
# Run elevated. This helper controls only its uniquely named WPR session.
# The workload remains in the caller's normal user session.
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $root) { throw 'Choose a new output directory' }
New-Item -ItemType Directory -Path $root | Out-Null
$instance='MetallicPacing-'+[Guid]::NewGuid().ToString('N')
$profile=Join-Path $root 'PacingTrace.wprp'
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'PacingTrace.wprp') -Destination $profile
@{instance=$instance; pid=$PID; sessions=$Sessions; timeout=$SessionTimeoutSeconds; profileSha256=(Get-FileHash -LiteralPath $profile -Algorithm SHA256).Hash} | ConvertTo-Json | Set-Content (Join-Path $root 'Controller.json')
try {
    for ($i=1; $i -le $Sessions; ++$i) {
        if (Test-Path -LiteralPath (Join-Path $root 'Finish.request')) { break }
        $dir=Join-Path $root "trace$i"
        New-Item -ItemType Directory -Path $dir | Out-Null
        & wpr.exe -start ($profile+'!MetallicPacing') -instancename $instance *> (Join-Path $dir 'Start.log')
        if ($LASTEXITCODE -ne 0) { throw "WPR start failed: $LASTEXITCODE" }
        try {
            @{started=(Get-Date).ToString('o'); instance=$instance} | ConvertTo-Json | Set-Content (Join-Path $dir 'Started.json')
            $deadline=(Get-Date).AddSeconds($SessionTimeoutSeconds)
            while ((Get-Date) -lt $deadline -and !(Test-Path -LiteralPath (Join-Path $dir 'Stop.request')) -and !(Test-Path -LiteralPath (Join-Path $root 'Finish.request'))) {
                Start-Sleep -Milliseconds 250
            }
        } finally {
            & wpr.exe -stop (Join-Path $dir 'Pacing.etl') 'Metallic Full pacing timeline' -skipPdbGen -compress -instancename $instance *> (Join-Path $dir 'Stop.log')
            $stopCode=$LASTEXITCODE
            @{stopped=(Get-Date).ToString('o'); exitCode=$stopCode} | ConvertTo-Json | Set-Content (Join-Path $dir 'Stopped.json')
            if ($stopCode -ne 0) {
                # Only cancel the session this invocation successfully started.
                & wpr.exe -cancel -instancename $instance >> (Join-Path $dir 'Stop.log')
                throw "WPR save failed: $stopCode"
            }
        }
    }
    'complete' | Set-Content (Join-Path $root 'Complete.txt')
} catch {
    $_ | Out-String | Set-Content (Join-Path $root 'Error.txt')
    exit 1
}
