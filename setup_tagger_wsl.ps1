# 詳細な使い方は下のコメントベースヘルプと Get-Help を参照。
#region Parameters

<#
.SYNOPSIS
    run_tagger.ps1のWSLモードから呼び出すコンテナセットアップ・起動処理。
.DESCRIPTION
    通常はrun_tagger.ps1 -Wslを使用する。ネイティブの仮想環境を作成せず、
    コンテナイメージとLinux用保存先を準備する。初回は自動でイメージを構築する。
    -Action Buildはイメージの更新のみ実行する。引数なしと-hはヘルプを表示する。
.PARAMETER ApplicationArgs
    共通ランチャーで構築したPython引数の配列。
.PARAMETER Action
    Run: イメージがなければ構築して起動。Build: イメージの構築のみ。Help: ヘルプ。
.PARAMETER Image
    起動・構築するコンテナイメージ名。
.PARAMETER BaseImage
    コンテナ構築時のベースイメージ。既定ubuntu:24.04。
.PARAMETER WorkspaceVolume
    Linux仮想環境・設定・キャッシュを保持する名前付きボリューム。
.PARAMETER ContainerName
    コンテナ名。停止後はコンテナ本体だけ削除する。
.PARAMETER PublishAddress
    ServerのWindows側待受アドレス。既定127.0.0.1。
.PARAMETER DataPath
    既存.dbv4のWindowsディレクトリ。未指定時はボリューム内へ保存する。
.PARAMETER GpuRuntimePath
    追加WSL GPUライブラリのWindowsディレクトリ。読み取り専用で公開する。
.PARAMETER Interactive
    標準入力を接続する。Loginでは自動接続する。
.PARAMETER Path
    Windowsの画像ファイル・フォルダ。/imagesへ公開する。
.PARAMETER Port
    Server公開ポート（既定5000）。
.PARAMETER Server
    Serverとしてポートを公開する。
.PARAMETER Client
    Clientとして接続し、GPU公開を省略する。
.PARAMETER HostIP
    Clientの接続先。ClientではPathとともに必須。
.PARAMETER Help
    -hで詳細ヘルプを表示する。引数なしでも表示する。
.EXAMPLE
    .\setup_tagger_wsl.ps1 -Action Build
#>
[CmdletBinding()]
param(
    [Alias('ac')][ValidateSet('Build', 'Run', 'Help')][string]$Action = 'Run',
    [Alias('im')][string]$Image = 'dbv4-tagger-wsl:local',
    [Alias('bi')][string]$BaseImage = 'ubuntu:24.04',
    [Alias('vol')][ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_.-]*$')][string]$WorkspaceVolume = 'dbv4-tagger-wsl-workspace',
    [Alias('cn')][ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_.-]*$')][string]$ContainerName = 'dbv4-tagger-wsl',
    [Alias('pa')][ValidateSet('127.0.0.1', '0.0.0.0')][string]$PublishAddress = '127.0.0.1',
    [Alias('dp')][string]$DataPath,
    [Alias('gr')][string]$GpuRuntimePath,
    [Alias('it')][switch]$Interactive,
    [Alias('p')][string]$Path,
    [Alias('u')][int]$Port = 5000,
    [Alias('s')][switch]$Server,
    [Alias('c')][switch]$Client,
    [Alias('j')][string]$HostIP,
    [Alias('aa')][string[]]$ApplicationArgs = @(),
    [Alias('h')][switch]$Help
)
#endregion
#region Setup
$ErrorActionPreference = 'Stop'
if ($Help -or $Action -eq 'Help' -or $PSBoundParameters.Count -eq 0) {
    Get-Help $PSCommandPath -Full
    exit 0
}
if (-not (Get-Command wslc -ErrorAction SilentlyContinue)) { throw 'WSL 2.9.3以降とwslcが必要です。wsl --updateで更新してください。' }
if ($Client -and (-not $HostIP -or -not $Path)) { throw 'Clientでは-HostIPと-Pathが必要です。' }
Write-Host '[INFO] 実行モード: WSL / Linuxコンテナ (-Wsl)' -ForegroundColor Cyan
Write-Host "[INFO] コンテナイメージ: $Image"
$NeedsBuild = $Action -eq 'Build'
if (-not $NeedsBuild) {
    & wslc image inspect $Image *> $null
    $NeedsBuild = $LASTEXITCODE -ne 0
}
if ($NeedsBuild) {
    Write-Host '[INFO] WSLコンテナイメージを構築します。' -ForegroundColor Cyan
    & wslc build --file (Join-Path $PSScriptRoot 'containers/wsl/Containerfile') --build-arg "BASE_IMAGE=$BaseImage" --tag $Image $PSScriptRoot
    if ($LASTEXITCODE -ne 0 -or $Action -eq 'Build') { exit $LASTEXITCODE }
}
$UseGpu = $ApplicationArgs -contains '--gpu'
$ProviderIndex = [Array]::IndexOf($ApplicationArgs, '--provider')
if ($ProviderIndex -ge 0) { $UseGpu = $ApplicationArgs[$ProviderIndex + 1] -ne 'cpu' }
if ($UseGpu -and $ProviderIndex -lt 0 -and -not $Client) {
    $Vendors = (Get-CimInstance Win32_VideoController).Name -join ' '
    if ($Vendors -match 'NVIDIA') { $ApplicationArgs += @('--provider', 'cuda') }
    elseif ($Vendors -match 'Intel') { $ApplicationArgs += @('--provider', 'intel') }
    elseif ($Vendors -match 'AMD|Radeon') { $ApplicationArgs += @('--provider', 'rocm') }
    else { throw 'GPUを識別できません。-Providerを指定してください。' }
}
#endregion
#region Container arguments
$ContainerArgs = @('run', '--rm', '--name', $ContainerName, '--shm-size', '1g',
    '--mount', "type=volume,source=$WorkspaceVolume,target=/workspace")
if ($Interactive -or $ApplicationArgs -contains '--login') {
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
if ($UseGpu -and -not $Client -and $ApplicationArgs -notcontains '--login') {
    $ContainerArgs += @('--gpus', 'all')
}
if ($Server) {
    $ContainerArgs += @('--publish', "${PublishAddress}:${Port}:${Port}")
}
$InputPath = $null
if ($Path) {
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
# Translate only the transport boundary; inference options are shared with native.
$LinuxArgs = @()
for ($Index = 0; $Index -lt $ApplicationArgs.Count; $Index++) {
    $Argument = $ApplicationArgs[$Index]
    if ($Argument -eq '--mode') {
        $Index++
        switch ($ApplicationArgs[$Index]) {
            'server' { $LinuxArgs += '--server' }
            'client' { $LinuxArgs += '--client' }
        }
    } elseif ($Path -and $Argument -eq $Path) {
        $LinuxArgs += $InputPath
    } else { $LinuxArgs += $Argument }
}
if ($Server -or $Client) {
    if ($LinuxArgs -notcontains '--port') { $LinuxArgs += @('--port', "$Port") }
}
if ($LinuxArgs.Count -eq 0) { $LinuxArgs = @('--gen-config') }
$ContainerArgs += $LinuxArgs
#endregion
#region Run
Write-Host "[INFO] WSLコンテナ名: $ContainerName"
Write-Host "[INFO] Linux環境・設定の保存先: $WorkspaceVolume (/workspace)"
if ($DataPath) {
    Write-Host "[INFO] Windows共有データ: $($DataItem.FullName) -> /workspace/.dbv4"
} else {
    Write-Host "[INFO] モデル・認証の保存先: コンテナ用ボリューム内の /workspace/.dbv4"
}
if ($Server) {
    Write-Host "[INFO] サーバー公開設定: http://${PublishAddress}:${Port} (モデル初期化後に接続可能)"
}
Write-Host '[INFO] WSL Linuxコンテナを起動します。以降はコンテナ内のログです。' -ForegroundColor Cyan
& wslc @ContainerArgs
exit $LASTEXITCODE
#endregion
