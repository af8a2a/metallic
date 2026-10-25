# Companion to RunMetallicCfgReplay.ps1. Keep per-process evidence: whole-card
# utilization alone cannot distinguish this benchmark from another renderer.
$ErrorActionPreference = 'Stop'
Write-Output 'timestamp,pid,process,engine,utilization'
$queryFailures = 0
while ($true) {
    try {
        Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -Continuous -ErrorAction Stop | ForEach-Object {
            $sampleTime = $_.Timestamp.ToString('o')
            foreach ($sample in $_.CounterSamples) {
                if ($sample.Status -ne 0) { continue }
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
    } catch {
        # GPU Engine instance enumeration can fail transiently as processes
        # start/exit (PDH buffer resize). Preserve the gap; never emit zero data.
        ++$queryFailures
        [Console]::Error.WriteLine("{0} GPU counter query failed ({1}/5): {2}",
            [DateTime]::UtcNow.ToString('o'), $queryFailures, $_.Exception.Message)
        if ($queryFailures -ge 5) { exit 1 }
        Start-Sleep -Seconds 1
    }
}
