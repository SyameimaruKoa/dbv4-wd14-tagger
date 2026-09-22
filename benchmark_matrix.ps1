<#
.SYNOPSIS
    Runs comparable CPU and GPU benchmarks on Windows.
.DESCRIPTION
    Invokes benchmark_matrix.py for the selected GPU vendor. Every provider uses
    its own .venv_bench_<provider> environment; unavailable cases are recorded.
.PARAMETER Vendor
    nvidia, intel, or amd.
.PARAMETER DeviceName
    GPU name verified in Windows Device Manager. Must contain the vendor name.
.PARAMETER OutputDir
    New directory for raw results, manifest, and summary.
.PARAMETER GpuIndex
    CUDA, TensorRT, DirectML adapter index. Verify the adapter before measuring.
.PARAMETER DirectMlDeviceIndex
    DirectML adapter index, independent of CUDA/TensorRT.
.PARAMETER WebGpuDeviceIndex
    WebGPU adapter index. Verify the adapter before measuring.
.PARAMETER RetryProvider
    Provider to retry, such as tensorrt. Requires RetryFailed.
.PARAMETER RetryFailed
    Resume a completed matrix, preserving successful JSON and rerunning failures.
.PARAMETER TensorRtLibDir
    TensorRT 10 bin directory containing nvinfer_10.dll. If omitted, the
    benchmark discovers a single TensorRT installation under Downloads.
.PARAMETER OpenVinoDevice
    OpenVINO device such as GPU.0. Intel hardware name is checked automatically.
.EXAMPLE
    .\benchmark_matrix.ps1 -Vendor intel -DeviceName "Intel UHD Graphics 630" -OutputDir benchmarks\intel_windows_20260923
#>
#region Arguments
param(
    [string]$Vendor,
    [string]$DeviceName,
    [string]$OutputDir,
    [int]$GpuIndex = 0,
    [int]$DirectMlDeviceIndex = 0,
    [int]$WebGpuDeviceIndex = 0,
    [string]$OpenVinoDevice = "GPU.0",
    [string]$TensorRtLibDir,
    [switch]$RetryFailed,
    [string]$RetryProvider,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)
#endregion
#region Help
if ($args -contains "-h" -or $args -contains "--help" -or $ExtraArgs -contains "-h" -or $ExtraArgs -contains "--help" -or -not $Vendor -or -not $DeviceName -or -not $OutputDir) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}
#endregion
#region Run
$python = if (Test-Path -LiteralPath ".venv_bench_cpu\Scripts\python.exe") {
    ".venv_bench_cpu\Scripts\python.exe"
} else {
    "python"
}
$tensorRtArgs = if ($TensorRtLibDir) { @("--tensorrt-lib-dir", $TensorRtLibDir) } else { @() }
$retryArgs = if ($RetryFailed) { @("--retry-failed") } else { @() }
if ($RetryProvider) { $retryArgs += @("--retry-provider", $RetryProvider) }
& $python benchmark_matrix.py --vendor $Vendor --device-name $DeviceName --output-dir $OutputDir --gpu-index $GpuIndex --directml-device-index $DirectMlDeviceIndex --webgpu-device-index $WebGpuDeviceIndex --openvino-device $OpenVinoDevice @tensorRtArgs @retryArgs @ExtraArgs
exit $LASTEXITCODE
#endregion
