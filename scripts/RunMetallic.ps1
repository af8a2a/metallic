[CmdletBinding(DefaultParameterSetName = 'BuildAndRun')]
param(
    [ValidatePattern('^[A-Za-z][A-Za-z0-9_]*$')]
    [string]$Target = 'Metallic',

    [ValidateSet('metallic-dev', 'metallic-release', 'metallic-relwithdebinfo', 'metallic-ci', 'metallic-full')]
    [string]$Preset = 'metallic-relwithdebinfo',

    [Parameter(Mandatory, ParameterSetName = 'BuildOnly')]
    [switch]$BuildOnly,

    [Parameter(Mandatory, ParameterSetName = 'RunOnly')]
    [switch]$RunOnly,

    [Parameter(ParameterSetName = 'BuildAndRun')]
    [Parameter(ParameterSetName = 'RunOnly')]
    [string[]]$ProgramArguments = @(),

    [string]$ExecutablePath
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$buildDirectory = 'build-' + $Preset.Substring('metallic-'.Length)

Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    if (-not $RunOnly) {
        $cachePath = Join-Path $buildDirectory 'CMakeCache.txt'
        $hasCache = Test-Path -LiteralPath $cachePath -PathType Leaf
        $toolsetArgument = ''
        if ($hasCache) {
            $compilerEntry = Select-String -LiteralPath $cachePath -Pattern '^CMAKE_CXX_COMPILER:[^=]+=' | Select-Object -First 1
            if ($compilerEntry.Line -notmatch '^CMAKE_CXX_COMPILER:[^=]+=(.+)/VC/Tools/MSVC/([^/]+)/bin/Hostx64/x64/cl\.exe$') {
                throw "Expected a configured x64 MSVC toolchain in $cachePath."
            }
            $installation = $Matches[1]
            $toolsetArgument = "-vcvars_ver=$($Matches[2])"
        } else {
            $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
            $installation = & $vswhere -latest -prerelease -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            if ($LASTEXITCODE -ne 0 -or -not $installation) {
                throw 'No Visual Studio installation with x64 C++ build tools was found.'
            }
        }

        $vcvars = Join-Path $installation 'VC/Auxiliary/Build/vcvars64.bat'
        if (-not (Test-Path -LiteralPath $vcvars -PathType Leaf)) {
            throw "Missing Visual Studio environment script: $vcvars"
        }
        $buildCommand = "cmake --build $buildDirectory --target $Target --parallel 4"
        if (-not $hasCache) {
            $buildCommand = "cmake --preset $Preset && $buildCommand"
        }
        & $env:ComSpec /d /s /c "call `"$vcvars`" $toolsetArgument && $buildCommand"
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }

    if (-not $BuildOnly) {
        if (-not $ExecutablePath) {
            $ExecutablePath = switch ($Target) {
                'MetallicCtl' { 'Source/metallicctl.exe' }
                'MetallicTracy' { 'Tools/MetallicTracy.exe' }
                default { "Source/$Target.exe" }
            }
        }
        $executable = Join-Path $buildDirectory $ExecutablePath
        if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
            throw "Executable not found: $executable"
        }
        & (Resolve-Path -LiteralPath $executable).Path @ProgramArguments
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
