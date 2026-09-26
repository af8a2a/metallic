[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\External\slang')
)
$ErrorActionPreference = 'Stop'
$version = '2026.18.2'
$expectedHash = '747602aec6b3623658d55fea87492d71828e15d16802d7941204fde418ceee8e'
$destinationPath = [IO.Path]::GetFullPath($Destination)
$compilerPath = Join-Path $destinationPath 'bin\slangc.exe'
if (Test-Path -LiteralPath $destinationPath) {
    if (Test-Path -LiteralPath $compilerPath) {
        $installedVersion = (& $compilerPath -version 2>&1 | Out-String).Trim()
        if ($LASTEXITCODE -eq 0 -and $installedVersion -eq $version) {
            Write-Output "Slang $version is already installed at $destinationPath"
            exit 0
        }
    }
    throw "Destination already exists: $destinationPath. Keep it as a backup and choose an empty -Destination; this installer does not overwrite installations."
}
$cachePath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\.cache\slang'))
New-Item -ItemType Directory -Force -Path $cachePath | Out-Null
$archivePath = Join-Path $cachePath "slang-$version-windows-x86_64.zip"
if (-not (Test-Path -LiteralPath $archivePath)) {
    Invoke-WebRequest -Uri "https://github.com/shader-slang/slang/releases/download/v$version/slang-$version-windows-x86_64.zip" -OutFile $archivePath
}
$actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
if ($actualHash -ne $expectedHash) {
    throw "SHA256 mismatch for $archivePath. Expected $expectedHash; got $actualHash."
}
$stagingPath = Join-Path $cachePath ("staging-" + [Guid]::NewGuid().ToString('N'))
Expand-Archive -LiteralPath $archivePath -DestinationPath $stagingPath
$stagedCompiler = Join-Path $stagingPath 'bin\slangc.exe'
$stagedVersion = (& $stagedCompiler -version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $stagedVersion -ne $version) {
    throw "Unexpected compiler version in $stagingPath : $stagedVersion"
}
New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($destinationPath)) | Out-Null
Move-Item -LiteralPath $stagingPath -Destination $destinationPath
Write-Output "Installed Slang $version at $destinationPath (verified SHA256 $expectedHash)."
