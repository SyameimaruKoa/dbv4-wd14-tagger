# 詳細な使い方は下のコメントベースヘルプと Get-Help を参照。
#region Parameters

<#
.SYNOPSIS
    WindowsからWSL Linuxコンテナ版DBV4 Taggerを起動する。
.DESCRIPTION
    WSL 2.9.3以降のwslcが必要。初回は -Action Build でイメージを作成する。
    コードはイメージから読み込み、Linux仮想環境、config.json、モデル、認証は
    名前付きボリュームに保持する。Windows版の仮想環境は共有しない。
    GPU provider指定時は --gpus all で全GPUを公開する。GPU番号は -GpuIndex。
    providerを固定するため、GPU初期化失敗時にCPUへ自動退避しない。
.PARAMETER Action
    Build: イメージ構築。Run: 起動。Help: ヘルプ。
.PARAMETER Mode
    Server / Standalone / Client / Login / Probe。Probeはモデルを実際に初期化する。
.PARAMETER Provider
    cpu / cuda / tensorrt / intel / ncnn / webgpu / rocm / migraphx。
    各GPUのWSL対応WindowsドライバーとLinuxユーザー空間ライブラリが必要。
    DirectMLはLinux版ONNX Runtimeで使用できないためWindows版で利用する。
.PARAMETER Path
    Standalone/Clientで処理するWindowsの画像ファイルまたはフォルダ。
    コンテナへ書き込み可能で公開するため、XMP・整理も元画像へ適用される。
.PARAMETER Image
    イメージ名。ROCmなどの追加ライブラリが必要な場合は派生イメージを指定。
.PARAMETER BaseImage
    Build時のベースイメージ。既定ubuntu:24.04。ROCm用には対応するROCmイメージを指定。
.PARAMETER DataPath
    既存の.dbv4ディレクトリを共有する場合のWindowsパス。モデル・HF認証を再利用する。
    未指定時は名前付きボリューム内に保存。Windows版と同時に更新しない。
.PARAMETER GpuRuntimePath
    WSL用ベンダーライブラリを含むWindowsディレクトリ。/gpu-runtimeへ読み取り専用で公開。
    AMDのWSL用libhsa-runtime64.so等を別途配置する場合に使用する。
.PARAMETER WorkspaceVolume
    永続ボリューム名。別環境を作る場合は別名を指定。同じボリュームで同時起動しない。
.PARAMETER ContainerName
    コンテナ名（既定dbv4-tagger-wsl）。停止は wslc stop dbv4-tagger-wsl。
.PARAMETER Port
    Server/Clientポート（既定5000）。ServerではWindowsへ同じ番号で公開する。
.PARAMETER PublishAddress
    ServerのWindows側待受アドレス。既定127.0.0.1。
    Linuxなど別PCから接続する場合は0.0.0.0を指定しWindowsファイアウォールを設定。
.PARAMETER HostName
    Clientの接続先。コンテナのlocalhostはWindowsホストとは異なる。
.PARAMETER ModelProfile
    モデルプロファイル名（既定balanced）。
.PARAMETER GpuIndex
    コンテナ内のGPU番号（既定0）。
.PARAMETER TaggerArgs
    Linux run_tagger.shへ渡す追加引数の配列。パスはコンテナ内のパスを指定。
.PARAMETER Help
    -h / -Helpで詳しいヘルプを表示する。引数なしでも表示する。
.PARAMETER Interactive
    stdinをコンテナへ接続する。Loginでは自動で有効。端末から実行する場合に使用する。
.EXAMPLE
    .\run_tagger_wsl.ps1 -Action Build
.EXAMPLE
    .\run_tagger_wsl.ps1 -Mode Server -Provider intel -ModelProfile ultra -PublishAddress 0.0.0.0
.EXAMPLE
    .\run_tagger_wsl.ps1 -Mode Standalone -Provider cuda -Path 'C:\Images' -TaggerArgs @('--recursive', '--no-report')
.EXAMPLE
    .\run_tagger_wsl.ps1 -Mode Probe -Provider intel -ModelProfile lightweight
.EXAMPLE
    .\run_tagger_wsl.ps1 -Mode Login
#>

[CmdletBinding()]
param(
    [ValidateSet('Build', 'Run', 'Help')]
    [string]$Action = 'Run',
    [ValidateSet('Server', 'Standalone', 'Client', 'Login', 'Probe')]
    [string]$Mode = 'Server',
    [ValidateSet('cpu', 'cuda', 'tensorrt', 'intel', 'ncnn', 'webgpu', 'rocm', 'migraphx')]
    [string]$Provider = 'cpu',
    [string]$Path,
    [string]$Image = 'dbv4-tagger-wsl:local',
    [string]$BaseImage = 'ubuntu:24.04',
    [string]$DataPath,
    [string]$GpuRuntimePath,
    [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_.-]*$')]
    [string]$WorkspaceVolume = 'dbv4-tagger-wsl-workspace',
    [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_.-]*$')]
    [string]$ContainerName = 'dbv4-tagger-wsl',
    [ValidateRange(1, 65535)]
    [int]$Port = 5000,
    [ValidateSet('127.0.0.1', '0.0.0.0')]
    [string]$PublishAddress = '127.0.0.1',
    [string]$HostName,
    [string]$ModelProfile = 'balanced',
    [ValidateRange(0, 2147483647)]
    [int]$GpuIndex = 0,
    [string[]]$TaggerArgs = @(),
    [Alias('h')]
    [switch]$Help,
    [switch]$Interactive
)
#endregion

#region Help and prerequisites
$ErrorActionPreference = 'Stop'
if ($Help -or $Action -eq 'Help' -or $PSBoundParameters.Count -eq 0) {
    Get-Help $PSCommandPath -Full
    exit 0
}
if (-not (Get-Command wslc -ErrorAction SilentlyContinue)) {
    throw 'wslcが見つかりません。WSL 2.9.3以降をインストールし、wsl --update を実行してください。'
}
#endregion

#region Build
if ($Action -eq 'Build') {
    & wslc build --file (Join-Path $PSScriptRoot 'containers/wsl/Containerfile') --build-arg "BASE_IMAGE=$BaseImage" --tag $Image $PSScriptRoot
    exit $LASTEXITCODE
}
#endregion

#region Container arguments
$ContainerArgs = @('run', '--rm', '--name', $ContainerName, '--shm-size', '1g',
    '--mount', "type=volume,source=$WorkspaceVolume,target=/workspace")
if ($Interactive -or $Mode -eq 'Login') {
    $ContainerArgs += '--interactive'
}
if ($DataPath) {
    $DataItem = Get-Item -LiteralPath $DataPath
    if (-not $DataItem.PSIsContainer -or $DataItem.FullName.Contains(',')) {
        throw '-DataPathにはカンマを含まない既存のディレクトリを指定してください。'
    }
    $ContainerArgs += @('--mount', "type=bind,source=$($DataItem.FullName),target=/workspace/.dbv4")
}
if ($GpuRuntimePath) {
    $RuntimeItem = Get-Item -LiteralPath $GpuRuntimePath
    if (-not $RuntimeItem.PSIsContainer -or $RuntimeItem.FullName.Contains(',')) {
        throw '-GpuRuntimePathにはカンマを含まない既存のディレクトリを指定してください。'
    }
    $ContainerArgs += @('--mount', "type=bind,source=$($RuntimeItem.FullName),target=/gpu-runtime,readonly")
}
if ($Provider -ne 'cpu' -and $Mode -notin @('Client', 'Login')) {
    $ContainerArgs += @('--gpus', 'all')
}
if ($Mode -eq 'Server') {
    $ContainerArgs += @('--publish', "${PublishAddress}:${Port}:${Port}")
}
$InputPath = $null
if ($Mode -in @('Standalone', 'Client')) {
    if (-not $Path) {
        throw '-Mode Standalone/Client では -Path が必要です。-h で詳しい使い方を確認できます。'
    }
    $Item = Get-Item -LiteralPath $Path
    if ($Item.PSIsContainer) {
        $MountPath = $Item.FullName
        $InputPath = '/images'
    } else {
        $MountPath = $Item.DirectoryName
        $InputPath = '/images/' + $Item.Name
    }
    if ($MountPath.Contains(',')) {
        throw '画像のマウント元パスにはカンマを使用できません。別のパスを指定してください。'
    }
    $ContainerArgs += @('--mount', "type=bind,source=$MountPath,target=/images")
}
$ContainerArgs += $Image
$LinuxArgs = @('--provider', $Provider, '--model-profile', $ModelProfile,
    '--gpu-index', "$GpuIndex")
switch ($Mode) {
    'Server' { $LinuxArgs += @('--server', '--port', "$Port") }
    'Client' {
        if (-not $HostName) {
            throw '-Mode Client では -HostName で接続先を指定してください。'
        }
        $LinuxArgs += @('--client', '--host', $HostName, '--port', "$Port", '--path', $InputPath)
    }
    'Standalone' { $LinuxArgs += @('--path', $InputPath) }
    'Login' { $LinuxArgs = @('--login') }
    'Probe' { $LinuxArgs += '--probe-provider' }
}
$ContainerArgs += $LinuxArgs + $TaggerArgs
#endregion

#region Run
& wslc @ContainerArgs
exit $LASTEXITCODE
#endregion
