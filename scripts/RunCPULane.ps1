# Hosted candidate only: local CPU results do not validate this SDK-off configuration,
# first Slang download, or the Windows 2025 compiler/STL combination.
$ErrorActionPreference = 'Stop'
# Capture native exit codes explicitly, including commands piped into Tee-Object.
$PSNativeCommandUseErrorActionPreference = $false

$repositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repositoryRoot
$logDirectory = Join-Path $repositoryRoot 'build-ci/ci-logs'
$junitPath = Join-Path $repositoryRoot 'build-ci/CTestResults.xml'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

function Invoke-LoggedNative {
    param([string]$FilePath, [string[]]$Arguments, [string]$LogName)
    Write-Host ('=> {0} {1}' -f $FilePath, ($Arguments -join ' '))
    & $FilePath @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $logDirectory $LogName)
    $nativeExitCode = $LASTEXITCODE
    if ($nativeExitCode -ne 0) {
        exit $nativeExitCode
    }
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vswhereExitCode = $LASTEXITCODE
if ($vswhereExitCode -ne 0) {
    exit $vswhereExitCode
}
if (-not $installation) {
    throw 'No Visual Studio installation with the x64 C++ tools was found.'
}
$devCmd = Join-Path $installation 'Common7/Tools/VsDevCmd.bat'
# Capture, never print, the environment. The && preserves a failing VsDevCmd status.
$environment = & $env:ComSpec /d /s /c "call `"$devCmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
$devCmdExitCode = $LASTEXITCODE
if ($devCmdExitCode -ne 0) {
    exit $devCmdExitCode
}
foreach ($line in $environment) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
Remove-Variable environment

# Report only selected tool facts, not the full environment or credentials.
$compiler = Get-Command cl.exe -ErrorAction Stop
$compilerVersion = (Get-Item -LiteralPath $compiler.Source).VersionInfo.ProductVersion
@(
    "Runner image: $env:ImageOS / $env:ImageVersion"
    "cl: $($compiler.Source) / $compilerVersion"
    "VC tools: $env:VCToolsVersion"
    "Windows SDK: $env:WindowsSdkDir / $env:WindowsSDKVersion"
) | Tee-Object -FilePath (Join-Path $logDirectory 'toolchain.log')
Invoke-LoggedNative -FilePath cmake -Arguments @('--version') -LogName 'cmake-version.log'
Invoke-LoggedNative -FilePath ninja -Arguments @('--version') -LogName 'ninja-version.log'
# Resolve once after importing VS: pip and CMake/CTest must use this same interpreter.
$pythonExecutable = (Get-Command python -ErrorAction Stop).Source
Invoke-LoggedNative -FilePath $pythonExecutable -Arguments @(
    '-c', 'import sys; print(sys.executable); print(sys.version); sys.exit(0 if sys.version_info[:2] == (3, 12) else "CPU lane requires Python 3.12")'
) -LogName 'python-version.log'
Invoke-LoggedNative -FilePath $pythonExecutable -Arguments @('-m', 'pip', '--version') -LogName 'pip-version.log'
Invoke-LoggedNative -FilePath $pythonExecutable -Arguments @(
    '-m', 'pip', 'install', '--only-binary=:all:', '-r', 'requirements-cpu.txt'
) -LogName 'python-install.log'
Invoke-LoggedNative -FilePath $pythonExecutable -Arguments @('-m', 'pip', 'check') -LogName 'python-check.log'
Invoke-LoggedNative -FilePath $pythonExecutable -Arguments @('-m', 'pip', 'list', '--format=freeze') -LogName 'python-packages.log'

# The existing bootstrap fetches required submodules and hash-pinned Slang/GDeflate/zstd.
# No Vulkan SDK, optional NVIDIA SDK, ISPC, or SPIRV-Tools installation is requested.
Invoke-LoggedNative -FilePath cmake -Arguments @(
    '--preset', 'metallic-ci',
    "-DPython3_EXECUTABLE=$pythonExecutable",
    '-DMETALLIC_ENABLE_NVIDIA_SDKS=OFF',
    '-DMETALLIC_ENABLE_SPIRV_TOOLS=OFF',
    '-DMETALLIC_ENABLE_ISPC=OFF'
) -LogName 'configure.log'
Invoke-LoggedNative -FilePath ./External/slang/bin/slangc.exe -Arguments @('-version') -LogName 'slang-version.log'
Invoke-LoggedNative -FilePath cmake -Arguments @('--build', '--preset', 'metallic-ci', '--parallel', '2') -LogName 'build.log'
# JUnit describes outer CTest jobs, not native GoogleTest skip counts.
# Keep the existing native skip policy unchanged; do not infer zero native skips from JUnit.
Invoke-LoggedNative -FilePath ctest -Arguments @(
    '--preset', 'metallic-ci', '--output-on-failure', '--no-tests=error', '--output-junit', $junitPath
) -LogName 'ctest.log'
