[CmdletBinding()]
param(
    [string]$BuildDirectory = 'cmake-build-debug-visual-studio',
    [string]$OutputDirectory = '.cache/aftermath/captures',
    [switch]$ShaderDebugInfo,
    [ValidateRange(1, 600)][int]$TimeoutSeconds = 60
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$buildRoot = if ([IO.Path]::IsPathRooted($BuildDirectory)) { $BuildDirectory } else { Join-Path $repoRoot $BuildDirectory }
$executable = Join-Path $buildRoot 'tests/MetallicRhiTests.exe'
if (!(Test-Path -LiteralPath $executable)) { $executable = Join-Path $buildRoot 'tests/Debug/MetallicRhiTests.exe' }
$executable = (Resolve-Path -LiteralPath $executable).Path
$outputRoot = if ([IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else { Join-Path $repoRoot $OutputDirectory }
$captureDirectory = Join-Path $outputRoot ((Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $captureDirectory -Force | Out-Null
$captureDirectory = (Resolve-Path -LiteralPath $captureDirectory).Path
$metadata = [ordered]@{
    executable = $executable
    executableSha256 = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash
    shaderDebugInfo = [bool]$ShaderDebugInfo
    arguments = @('--filter', 'visibility_buffer_deferred_openpbr', '--rhi-validation', '--output-dir', $captureDirectory)
    startedAt = (Get-Date).ToUniversalTime().ToString('o')
}
$pdb = [IO.Path]::ChangeExtension($executable, '.pdb')
if (Test-Path -LiteralPath $pdb) { $metadata.pdbSha256 = (Get-FileHash -LiteralPath $pdb -Algorithm SHA256).Hash }
$previousTest = $env:METALLIC_TEST_AFTERMATH
$previousDebugInfo = $env:METALLIC_AFTERMATH_SHADER_DEBUG_INFO
try {
    $env:METALLIC_TEST_AFTERMATH = '1'
    $env:METALLIC_AFTERMATH_SHADER_DEBUG_INFO = if ($ShaderDebugInfo) { '1' } else { '0' }
    $captureProcess = Start-Process -FilePath $executable -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
        -ArgumentList @('--filter', 'visibility_buffer_deferred_openpbr', '--rhi-validation', '--output-dir', ('"' + $captureDirectory + '"')) `
        -RedirectStandardOutput (Join-Path $captureDirectory 'stdout.log') `
        -RedirectStandardError (Join-Path $captureDirectory 'stderr.log')
} finally {
    $env:METALLIC_TEST_AFTERMATH = $previousTest
    $env:METALLIC_AFTERMATH_SHADER_DEBUG_INFO = $previousDebugInfo
}
$metadata.pid = $captureProcess.Id
$moduleBase = 0L
for ($attempt = 0; $attempt -lt 100 -and $moduleBase -eq 0 -and !$captureProcess.HasExited; ++$attempt) {
    $captureProcess.Refresh()
    if ($null -ne $captureProcess.MainModule) { $moduleBase = $captureProcess.MainModule.BaseAddress.ToInt64() }
    if ($moduleBase -eq 0) { Start-Sleep -Milliseconds 20 }
}
$metadata.moduleBase = '0x{0:X}' -f $moduleBase
$metadata.timedOut = !$captureProcess.WaitForExit($TimeoutSeconds * 1000)
if ($metadata.timedOut) {
    # Only terminate the test process launched by this invocation.
    $captureProcess.Kill()
    $captureProcess.WaitForExit()
}
$metadata.exitCode = $captureProcess.ExitCode
$metadata.finishedAt = (Get-Date).ToUniversalTime().ToString('o')
$dumpRoot = Join-Path $repoRoot '.cache/aftermath'
$dumpFiles = @(if (Test-Path -LiteralPath $dumpRoot) {
    Get-ChildItem -LiteralPath $dumpRoot -File | Where-Object {
        $_.Name -match ('-' + $captureProcess.Id + '-[0-9]+\.(json|nv-gpudmp)$')
    }
})
$metadata.dumpFiles = @($dumpFiles.Name)
$symbolizer = Get-Command llvm-symbolizer -ErrorAction SilentlyContinue
$stacks = [Collections.Generic.List[string]]::new()
foreach ($dumpFile in $dumpFiles) {
    Copy-Item -LiteralPath $dumpFile.FullName -Destination (Join-Path $captureDirectory $dumpFile.Name)
    if ($dumpFile.Extension -ne '.json') { continue }
    $dump = Get-Content -LiteralPath $dumpFile.FullName -Raw | ConvertFrom-Json
    $metadata.pageFault = @($dump.'Page fault info' | Where-Object { $null -ne $_ })[0]
    $metadata.deviceInfo = @($dump.'Device info' | Where-Object { $null -ne $_ })[0]
    foreach ($event in $dump.'Aftermath markers'.Context.Events.Event) {
        $stacks.Add([string]$event.Status)
        foreach ($entry in ($event.Callstack.Stack.Entry | Where-Object { $_.'Module name' -eq [IO.Path]::GetFileName($executable) } | Select-Object -First 6)) {
            if ($moduleBase -ne 0 -and $symbolizer) {
                $relativeAddress = '0x{0:X}' -f ($entry.Pointer - $moduleBase)
                $stacks.Add((& $symbolizer.Source "--obj=$executable" --relative-address --no-inlines $relativeAddress | Out-String).TrimEnd())
            } else {
                $stacks.Add(('{0} 0x{1:X}' -f $entry.'Module name', $entry.Pointer))
            }
        }
    }
}
$metadata | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $captureDirectory 'capture.json') -Encoding utf8
$stacks | Set-Content -LiteralPath (Join-Path $captureDirectory 'checkpoints.txt') -Encoding utf8
Write-Host "Aftermath evidence: $captureDirectory"
Write-Host "GPU test exit code: $($metadata.exitCode); dumps: $($dumpFiles.Count); shader debug info: $ShaderDebugInfo"
if ($metadata.timedOut) { exit 124 }
exit $metadata.exitCode
