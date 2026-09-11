[CmdletBinding(DefaultParameterSetName = 'Connect')]
param(
    [Parameter(Position = 0, Mandatory, ParameterSetName = 'Capture')]
    [string]$Capture,

    [Parameter(ParameterSetName = 'Connect')]
    [string]$Address = '127.0.0.1',

    [Parameter(ParameterSetName = 'Connect')]
    [ValidateRange(1, 65535)]
    [int]$Port = 8086,

    [Parameter(ParameterSetName = 'Connect')]
    [switch]$WithMetallic,

    [Parameter(ParameterSetName = 'Connect')]
    [string]$MetallicBuild = 'build-dev'
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$viewer = Join-Path $repositoryRoot 'build-tracy-viewer/tracy-profiler.exe'
if (-not (Test-Path -LiteralPath $viewer -PathType Leaf)) {
    $viewer = Join-Path $repositoryRoot 'build-tracy-viewer/Release/tracy-profiler.exe'
}
if (-not (Test-Path -LiteralPath $viewer -PathType Leaf)) {
    throw 'Tracy Viewer is missing. See Documentation/TracyGpuProfiling.md for build instructions.'
}

if ($PSCmdlet.ParameterSetName -eq 'Capture') {
    $capturePath = (Resolve-Path -LiteralPath $Capture).Path
    if (-not (Test-Path -LiteralPath $capturePath -PathType Leaf)) {
        throw "Capture is not a file: $capturePath"
    }
    $viewerArguments = '"' + $capturePath + '"'
} else {
    if ($WithMetallic) {
        $metallic = Join-Path $repositoryRoot "$MetallicBuild/Source/Metallic.exe"
        if (-not (Test-Path -LiteralPath $metallic -PathType Leaf)) {
            throw "Metallic executable is missing: $metallic"
        }
        $alreadyRunning = Get-Process -Name Metallic -ErrorAction SilentlyContinue |
            Where-Object { $_.Path -eq $metallic }
        if (-not $alreadyRunning) {
            Start-Process -FilePath $metallic -WorkingDirectory $repositoryRoot
        }
    }
    $viewerArguments = @('-a', $Address, '-p', $Port.ToString())
}

# Both programs are interactive tools. Launch without a waiting console so the
# caller can keep working while Tracy connects or loads a capture.
Start-Process -FilePath $viewer -ArgumentList $viewerArguments -WorkingDirectory $repositoryRoot
