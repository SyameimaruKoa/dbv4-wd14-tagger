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
    モード未指定で-Pathを指定した場合は、既存版と同じStandaloneで実行する。
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
.PARAMETER Server
    推論サーバーとして起動する（-Mode Serverと同じ）。
.PARAMETER Client
    クライアントとして起動する（-Mode Clientと同じ）。
.PARAMETER Login
    Hugging Faceへログインする（-Mode Loginと同じ）。
.PARAMETER Organize
    フォルダ整理を行う。-Tag併用時はタグ付けも行う。
.PARAMETER Tag
    整理モードでタグ付けを併用する。
.PARAMETER WallgenUnsorted
    wallgen未整理の末端フォルダ単位で整理する。
.PARAMETER NoReport
    HTMLレポートを作成しない。
.PARAMETER Recursive
    サブフォルダも検索する。
.PARAMETER NoRecursive
    サブフォルダを検索しない。
.PARAMETER Force
    既存のスコアを使わず再解析する。
.PARAMETER RecordRatio
    RAWスコアをXMPへ保存する。
.PARAMETER NoRecordRatio
    RAWスコアをXMPへ保存しない。
.PARAMETER IgnoreSensitive
    SensitiveをGeneralとして扱う（旧CLI互換）。
.PARAMETER Gpu
    GPUを使用する。Provider未指定時はWindowsのGPUから選択する。
.PARAMETER WebGpu
    WebGPUを使用する。WSL対応の実GPUドライバーが必要。
.PARAMETER BatchSize
    推論バッチサイズ。未指定時はLinux版の既定値を使用する。
.PARAMETER IoWorkers
    読込ワーカー数。-1は自動、0は並列読込なし。
.PARAMETER Thresh
    タグ採用閾値。未指定時はモデルの推奨値を使用する。
.PARAMETER ModelRepo
    Hugging FaceモデルリポジトリID。
.PARAMETER ModelFile
    モデルファイル名またはコンテナ内のパス。
.PARAMETER TagsFile
    タグCSV名またはコンテナ内のパス。
.PARAMETER ClientUploadMode
    Client転送方式。p/preprocessedは前処理済み、o/originalは元画像。
.PARAMETER NcnnPrecision
    ncnnの演算精度（未指定時はfp32）。
.PARAMETER NcnnPartSizeMiB
    ncnn重み分割目安MiB。未指定時は自動、0は分割なし。
.PARAMETER DirectMlDeviceIndex
    DirectML番号。DirectMLはLinux非対応のため指定時は停止する。
.PARAMETER WebGpuDeviceIndex
    WebGPUデバイス番号。
.PARAMETER TargetVendor
    対象GPUベンダー。
.PARAMETER OpenVinoDevice
    OpenVINOデバイス名（例GPU.0）。
.PARAMETER TensorRtLibDir
    TensorRTライブラリのコンテナ内ディレクトリ。
.PARAMETER SensitiveSplitMode
    旧CLI互換のSensitive分割数。DBV4では5段階固定。
.PARAMETER RatingThresh
    旧CLI互換のrating閾値。
.PARAMETER WallgenMoveMinRating
    wallgen未整理の移動下限（例R-15_0）。
.PARAMETER RemainingArgs
    Linux版へ渡す追加引数。
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
    [Alias('ac')]
    [ValidateSet('Build', 'Run', 'Help')]
    [string]$Action = 'Run',
    [Alias('md')]
    [ValidateSet('Server', 'Standalone', 'Client', 'Login', 'Probe')]
    [string]$Mode = 'Server',
    [Alias('ep')]
    [ValidateSet('cpu', 'cuda', 'tensorrt', 'intel', 'directml', 'ncnn', 'webgpu', 'rocm', 'migraphx')]
    [string]$Provider = 'cpu',
    [Alias('p')]
    [string]$Path,
    [Alias('im')]
    [string]$Image = 'dbv4-tagger-wsl:local',
    [Alias('bi')]
    [string]$BaseImage = 'ubuntu:24.04',
    [Alias('dp')]
    [string]$DataPath,
    [Alias('gr')]
    [string]$GpuRuntimePath,
    [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_.-]*$')]
    [Alias('vol')]
    [string]$WorkspaceVolume = 'dbv4-tagger-wsl-workspace',
    [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_.-]*$')]
    [Alias('cn')]
    [string]$ContainerName = 'dbv4-tagger-wsl',
    [Alias('u')]
    [ValidateRange(1, 65535)]
    [int]$Port = 5000,
    [Alias('pa')]
    [ValidateSet('127.0.0.1', '0.0.0.0')]
    [string]$PublishAddress = '127.0.0.1',
    [Alias('j', 'HostIP')]
    [string]$HostName,
    [Alias('m')]
    [string]$ModelProfile = 'balanced',
    [ValidateRange(0, 2147483647)]
    [Alias('gi')]
    [int]$GpuIndex = 0,
    [Alias('ta')]
    [string[]]$TaggerArgs = @(),
    [Alias('s')]
    [switch]$Server,
    [Alias('c')]
    [switch]$Client,
    [Alias('lo')]
    [switch]$Login,
    [Alias('o')]
    [switch]$Organize,
    [Alias('t')]
    [switch]$Tag,
    [Alias('x')]
    [switch]$WallgenUnsorted,
    [Alias('z')]
    [switch]$NoReport,
    [Alias('r')]
    [switch]$Recursive,
    [Alias('n')]
    [switch]$NoRecursive,
    [Alias('f')]
    [switch]$Force,
    [Alias('a')]
    [switch]$RecordRatio,
    [Alias('k')]
    [switch]$NoRecordRatio,
    [Alias('i')]
    [switch]$IgnoreSensitive,
    [Alias('g')]
    [switch]$Gpu,
    [Alias('wg')]
    [switch]$WebGpu,
    [Alias('b')]
    [int]$BatchSize,
    [Alias('w')]
    [int]$IoWorkers,
    [Alias('q')]
    [Nullable[float]]$Thresh,
    [Alias('e')]
    [string]$ModelRepo,
    [Alias('l')]
    [string]$ModelFile,
    [Alias('y')]
    [string]$TagsFile,
    [Alias('um')]
    [ValidateSet('original', 'preprocessed', 'o', 'p')]
    [string]$ClientUploadMode,
    [Alias('np')]
    [ValidateSet('fp32', 'fp16-storage', 'fp16-packed', 'fp16-arithmetic')]
    [string]$NcnnPrecision,
    [Alias('ns')]
    [ValidateRange(0, 2147483647)]
    [int]$NcnnPartSizeMiB,
    [Alias('di')]
    [ValidateRange(0, 2147483647)]
    [int]$DirectMlDeviceIndex,
    [Alias('wi')]
    [ValidateRange(0, 2147483647)]
    [int]$WebGpuDeviceIndex,
    [Alias('tv')]
    [ValidateSet('nvidia', 'intel', 'amd')]
    [string]$TargetVendor,
    [Alias('od')]
    [string]$OpenVinoDevice,
    [Alias('td')]
    [string]$TensorRtLibDir,
    [Alias('v')]
    [ValidateSet(2, 4, 6)]
    [int]$SensitiveSplitMode,
    [Alias('d')]
    [float]$RatingThresh,
    [Alias('wmm')]
    [string]$WallgenMoveMinRating,
    [Alias('ra')]
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs = @(),
    [Alias('h')]
    [switch]$Help,
    [Alias('it')]
    [switch]$Interactive
)
#endregion

#region Help and prerequisites
$ErrorActionPreference = 'Stop'
function Show-Help {
    $Details = Get-Help $PSCommandPath -Full
    Write-Host 'DBV4 Tagger Universal (日本語ヘルプ)' -ForegroundColor Cyan
    Write-Host ''
    Write-Host '使い方: .\run_tagger_wsl.ps1 [スイッチ] [値付きオプション] -Path <画像パス>' -ForegroundColor Yellow
    Write-Host ''
    Write-Host '  WSL Linuxコンテナ版。引数なしではヘルプを表示します。' -ForegroundColor Gray
    Write-Host '  初回は -Action Build でイメージを構築します。'
    Write-Host ''
    Write-Host '処理時に必要な入力:' -ForegroundColor Yellow
    Write-Host '    -Path (-p) <画像パス>          Standalone/Clientで処理するWindowsのファイル/フォルダ'
    Write-Host '    -HostIP / -HostName (-j) <ホスト名/IP>  Client接続先（Clientでは必須）'
    Write-Host ''
    Write-Host '値を指定するオプション（<>内の値が必要）:' -ForegroundColor Yellow
    $ValueOptions = [ordered]@{
        Action = 'Build|Run|Help'
        Mode = 'Server|Standalone|Client|Login|Probe'
        Provider = 'プロバイダ名'
        Path = '画像パス'
        HostName = 'ホスト名/IP'
        Port = 'ポート番号'
        PublishAddress = '127.0.0.1|0.0.0.0'
        ModelProfile = 'プロファイル名'
        GpuIndex = '0以上の整数'
        Image = 'イメージ名'
        BaseImage = 'ベースイメージ名'
        WorkspaceVolume = 'ボリューム名'
        ContainerName = 'コンテナ名'
        DataPath = 'ディレクトリパス'
        GpuRuntimePath = 'ディレクトリパス'
        TaggerArgs = '追加引数の配列'
        BatchSize = '数値'
        IoWorkers = '数値'
        Thresh = '数値'
        ModelRepo = '値'
        ModelFile = '値'
        TagsFile = '値'
        ClientUploadMode = '値'
        NcnnPrecision = '値'
        NcnnPartSizeMiB = '数値'
        DirectMlDeviceIndex = '数値'
        WebGpuDeviceIndex = '数値'
        TargetVendor = '値'
        OpenVinoDevice = '値'
        TensorRtLibDir = '値'
        SensitiveSplitMode = '数値'
        RatingThresh = '数値'
        WallgenMoveMinRating = '値'
        RemainingArgs = '追加引数の配列'
    }
    $Defaults = @{
        Action = 'Run'; Mode = 'Server'; Provider = 'cpu'; Port = '5000'
        PublishAddress = '127.0.0.1'; ModelProfile = 'balanced'; GpuIndex = '0'
        Image = 'dbv4-tagger-wsl:local'; BaseImage = 'ubuntu:24.04'
        WorkspaceVolume = 'dbv4-tagger-wsl-workspace'; ContainerName = 'dbv4-tagger-wsl'
    }
    foreach ($Name in $ValueOptions.Keys) {
        $ParameterHelp = $Details.parameters.parameter | Where-Object { $_.name -eq $Name }
        $Description = ($ParameterHelp.description | ForEach-Object { $_.Text.Trim() }) -join ' '
        $Description = $Description -replace '\s*\r?\n\s*', ' '
        $Marker = ' '
        if ($Defaults.ContainsKey($Name)) {
            $Marker = '★'
            if ($Description -notmatch '既定') {
                $Description += "（既定 $($Defaults[$Name])）"
            }
        }
        $Aliases = ((Get-Command $PSCommandPath).Parameters[$Name].Aliases | ForEach-Object { "-$_" }) -join ', '
        Write-Host ('  {0} {1,-48} {2}' -f $Marker, "-$Name ($Aliases) <$($ValueOptions[$Name])>", $Description)
    }
    Write-Host ''
    Write-Host '値を指定しないスイッチ:' -ForegroundColor Yellow
    foreach ($Name in @('Server', 'Client', 'Login', 'Organize', 'Tag', 'WallgenUnsorted', 'NoReport', 'Recursive', 'NoRecursive', 'Force', 'RecordRatio', 'NoRecordRatio', 'IgnoreSensitive', 'Gpu', 'WebGpu', 'Interactive', 'Help')) {
        $ParameterHelp = $Details.parameters.parameter | Where-Object { $_.name -eq $Name }
        $Description = ($ParameterHelp.description | ForEach-Object { $_.Text.Trim() }) -join ' '
        $Aliases = ((Get-Command $PSCommandPath).Parameters[$Name].Aliases | ForEach-Object { "-$_" }) -join ', '
        Write-Host "    -$Name ($Aliases)  $Description"
    }
    Write-Host ''
    Write-Host '★ は初期設定。GPU指定時は全GPUを公開し、-GpuIndexで使用デバイスを選びます。'
    Write-Host '  追加引数のパスはコンテナ内のパスを指定してください。'
    Write-Host ''
    Write-Host '実行例:' -ForegroundColor Yellow
    foreach ($Example in $Details.examples.example) {
        Write-Host "    $($Example.code.Trim())"
    }
    Write-Host ''
}
if ($Help -or $Action -eq 'Help' -or ($RemainingArgs -contains '--help') -or $PSBoundParameters.Count -eq 0) {
    Show-Help
    exit 0
}
if (-not (Get-Command wslc -ErrorAction SilentlyContinue)) {
    throw 'wslcが見つかりません。WSL 2.9.3以降をインストールし、wsl --update を実行してください。'
}
if ($Provider -eq 'directml' -or $PSBoundParameters.ContainsKey('DirectMlDeviceIndex')) {
    throw 'DirectMLはLinuxコンテナ非対応です。run_tagger.ps1 -Provider directml を使用してください。'
}
$SelectedModes = @()
if ($Server) { $SelectedModes += 'Server' }
if ($Client) { $SelectedModes += 'Client' }
if ($Login) { $SelectedModes += 'Login' }
if ($SelectedModes.Count -gt 1 -or ($SelectedModes.Count -eq 1 -and $PSBoundParameters.ContainsKey('Mode') -and $Mode -ne $SelectedModes[0])) {
    throw '-Server / -Client / -Login / -Modeで矛盾するモードを指定できません。'
}
if ($SelectedModes.Count -eq 1) { $Mode = $SelectedModes[0] }
elseif ($Path -and -not $PSBoundParameters.ContainsKey('Mode')) { $Mode = 'Standalone' }
if ($WebGpu) {
    if ($PSBoundParameters.ContainsKey('Provider') -and $Provider -ne 'webgpu') {
        throw '-WebGpuと別のProviderは同時に指定できません。'
    }
    $Provider = 'webgpu'
} elseif ($Gpu -and -not $PSBoundParameters.ContainsKey('Provider') -and $Action -ne 'Build') {
    $Vendors = (Get-CimInstance Win32_VideoController).Name -join ' '
    if ($Vendors -match 'NVIDIA') { $Provider = 'cuda' }
    elseif ($Vendors -match 'Intel') { $Provider = 'intel' }
    elseif ($Vendors -match 'AMD|Radeon') { $Provider = 'rocm' }
    else { throw 'GPUを識別できません。-Providerで使用する経路を指定してください。' }
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
$ValueArguments = [ordered]@{
    BatchSize = '--batch-size'; IoWorkers = '--io-workers'; Thresh = '--thresh'
    ModelRepo = '--model-repo'; ModelFile = '--model-file'; TagsFile = '--tags-file'
    ClientUploadMode = '--client-upload-mode'; NcnnPrecision = '--ncnn-precision'
    NcnnPartSizeMiB = '--ncnn-part-size-mib'; WebGpuDeviceIndex = '--webgpu-device-index'
    TargetVendor = '--target-vendor'; OpenVinoDevice = '--openvino-device'
    TensorRtLibDir = '--tensorrt-lib-dir'; SensitiveSplitMode = '--sensitive-split-mode'
    RatingThresh = '--rating-thresh'; WallgenMoveMinRating = '--wallgen-move-min-rating'
}
foreach ($Name in $ValueArguments.Keys) {
    if ($PSBoundParameters.ContainsKey($Name)) {
        $LinuxArgs += @($ValueArguments[$Name], [string]$PSBoundParameters[$Name])
    }
}
$SwitchArguments = [ordered]@{
    Organize = '--organize'; Tag = '--tag'; WallgenUnsorted = '--wallgen-unsorted'
    NoReport = '--no-report'; Recursive = '--recursive'; NoRecursive = '--no-recursive'
    Force = '--force'; RecordRatio = '--record-ratio'; NoRecordRatio = '--no-record-ratio'
    IgnoreSensitive = '--ignore-sensitive'
}
foreach ($Name in $SwitchArguments.Keys) {
    if ($PSBoundParameters[$Name]) { $LinuxArgs += $SwitchArguments[$Name] }
}
$ContainerArgs += $LinuxArgs + $TaggerArgs + $RemainingArgs
#endregion

#region Run
& wslc @ContainerArgs
exit $LASTEXITCODE
#endregion
