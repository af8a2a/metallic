param([Parameter(Mandatory=$true)][int]$TargetProcessId)
# Get-Counter wildcard expansion at query creation can miss a new GPU process.
# Wait for its engine instance, then use one continuous query during measurement.
$ErrorActionPreference = 'Stop'
Write-Output 'timestamp,pid,process,engine,utilization'
while ($true) {
    if (-not (Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue)) { exit 1 }
    $probe = Get-Counter '\GPU Engine(*)\Utilization Percentage' -MaxSamples 1 -ErrorAction Continue
    if (@($probe.CounterSamples | Where-Object { $_.InstanceName -like "pid_${TargetProcessId}_*" }).Count -gt 0) { break }
}
Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -Continuous -ErrorAction Continue | ForEach-Object {
    $sampleTime = $_.Timestamp.ToString('o')
    $names = @{}
    foreach ($sample in $_.CounterSamples) {
        if ($sample.Status -ne 0) { continue }
        $match = [regex]::Match($sample.InstanceName, '^pid_(\d+)_')
        if (-not $match.Success) { continue }
        $processId = [int]$match.Groups[1].Value
        # Keep target zero samples: absence and measured inactivity differ.
        if ($processId -ne $TargetProcessId -and $sample.CookedValue -lt 0.1) { continue }
        if (-not $names.ContainsKey($processId)) {
            try { $names[$processId] = [Diagnostics.Process]::GetProcessById($processId).ProcessName }
            catch { $names[$processId] = 'exited' }
        }
        $processName = $names[$processId]
        [pscustomobject]@{ timestamp = $sampleTime; pid = $processId; process = $processName
            engine = $sample.InstanceName; utilization = $sample.CookedValue } |
            ConvertTo-Csv -NoTypeInformation | Select-Object -Skip 1 | Write-Output
    }
}
