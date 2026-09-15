# Companion to RunMetallicCfgReplay.ps1. Keep per-process evidence: whole-card
# utilization alone cannot distinguish this benchmark from another renderer.
$ErrorActionPreference = 'Stop'
Write-Output 'timestamp,pid,process,engine,utilization'
Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -Continuous | ForEach-Object {
    $sampleTime = $_.Timestamp.ToString('o')
    foreach ($sample in $_.CounterSamples) {
        if ($sample.CookedValue -lt 0.1) { continue }
        $match = [regex]::Match($sample.InstanceName, '^pid_(\d+)_')
        if (-not $match.Success) { continue }
        $processId = [int]$match.Groups[1].Value
        try { $processName = [Diagnostics.Process]::GetProcessById($processId).ProcessName }
        catch { $processName = 'exited' }
        [pscustomobject]@{ timestamp = $sampleTime; pid = $processId; process = $processName
            engine = $sample.InstanceName; utilization = $sample.CookedValue } |
            ConvertTo-Csv -NoTypeInformation | Select-Object -Skip 1 | Write-Output
    }
}
