param([ValidateSet('native','wsl')][string]$Mode = 'wsl')
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location C:\tmp\dbv4-issue29
$ResultDir = if ($Mode -eq 'native') { Join-Path $PWD "results-native-gpu" } else { Join-Path $PWD "results-wsl-container" }
New-Item -ItemType Directory -Force $ResultDir | Out-Null
$EnvData = @{
  os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,TotalVisibleMemorySize,FreePhysicalMemory,TotalVirtualMemorySize,FreeVirtualMemory
  cpu = Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
  gpu = Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID
  pagefile = Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage
  power = powercfg /getactivescheme
  battery = Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining
  wslVersion = wsl --version | Out-String
  wslDistros = wsl --list --verbose | Out-String
  wslConfig = Get-Content "$env:USERPROFILE\.wslconfig" -ErrorAction SilentlyContinue | Out-String
  started = (Get-Date).ToString('o')
}
$EnvData | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 "$ResultDir\host-environment.json"
if ($Mode -eq 'native') {
  $Python = 'C:\Users\kouki\Documents\MyApp\wd14-tagger-xmp\venv_intel\Scripts\python.exe'
  $CommandArgs = @('benchmark_intel_memory.py','C:\Users\kouki\Documents\MyApp\wd14-tagger-xmp\.dbv4\models\53e96223459a3046\model.onnx','--baseline-batch-size','4','--iterations','5','--timeout','300','--output',"$ResultDir\comparison")
} else {
  $Python = 'wslc.exe'
  $CommandArgs = @('run','--rm','--name','dbv4-issue29-memory','--gpus','all','--shm-size','1g',
    '--mount','type=volume,source=dbv4-issue29-memory-workspace,target=/workspace',
    '--mount','type=bind,source=C:\tmp\dbv4-issue29,target=/experiment',
    '--mount','type=bind,source=C:\Users\kouki\Documents\MyApp\wd14-tagger-xmp\.dbv4,target=/model-cache,readonly',
    '--entrypoint','/bin/bash','dbv4-tagger:issue29','/experiment/run_issue29_container.sh')
}
$Handle = Start-Process -FilePath $Python -ArgumentList $CommandArgs -PassThru -WorkingDirectory $PWD -RedirectStandardOutput "$ResultDir\console.log" -RedirectStandardError "$ResultDir\errors.log"
$null = $Handle.Handle
$Writer = New-Object System.IO.StreamWriter("$ResultDir\host-samples.jsonl",$false,[System.Text.Encoding]::UTF8)
try {
  do {
    $Processes = @(Get-CimInstance Win32_Process)
    $TrackedIds = @($Handle.Id)
    do {
      $OldCount = $TrackedIds.Count
      $TrackedIds = @($TrackedIds + @($Processes | Where-Object { $_.ParentProcessId -in $TrackedIds } | Select-Object -ExpandProperty ProcessId) | Sort-Object -Unique)
    } while ($TrackedIds.Count -gt $OldCount)
    $Memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory | Select-Object AvailableMBytes,CommittedBytes,CommitLimit,PagesInputPersec,PageReadsPersec
    $PageFiles = @(Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage)
    $Vmmem = @(Get-Process -Name 'vmmem*' -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,WorkingSet64,PrivateMemorySize64)
    $GpuMemory = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -ErrorAction SilentlyContinue | Where-Object { $Name = $_.Name; @($TrackedIds | Where-Object { $Name -like "pid_$($_)_*" }).Count -gt 0 } | Select-Object Name,SharedUsage,DedicatedUsage,TotalCommitted)
    @{time=(Get-Date).ToString('o');memory=$Memory;pagefiles=$PageFiles;vmmem=$Vmmem;gpu_process_memory=$GpuMemory;tracked_pids=$TrackedIds} | ConvertTo-Json -Depth 6 -Compress | ForEach-Object { $Writer.WriteLine($_); $Writer.Flush() }
    Start-Sleep -Seconds 1
    $Handle.Refresh()
  } while (-not $Handle.HasExited)
} finally { $Writer.Close() }
$Handle.WaitForExit()
Write-Output "EXIT_CODE=$($Handle.ExitCode)"
Get-Content "$ResultDir\console.log"
Get-Content "$ResultDir\errors.log"
exit $Handle.ExitCode
