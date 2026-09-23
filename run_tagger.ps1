<#
.SYNOPSIS
    DBV4 Tagger Universal Wrapper (日本語版)

.DESCRIPTION
    DBV4モデルを使用して画像のタグ付けや整理を行うスクリプトじゃ。
    引数なしで実行すると「環境構築モード」として動作し、セットアップのみを行って終了する。
    事故防止のため、処理を行いたい場合は必ず -Path などを指定するのじゃ。

    【主な実行モード】
    1. Standalone (通常): その場で画像を処理する。
    2. Server: GPU推論サーバーとして待機する。
    3. Client: サーバーに画像を投げる。

.PARAMETER Path
    【対象パス】 (文字列)
    処理対象の画像ファイル、またはフォルダパス。

.PARAMETER Organize
    【整理モード】 (スイッチ)
    タグ付けを行わず、フォルダ振り分けのみを行うモードじゃ。

.PARAMETER Tag
    【タグ付け有効化】 (スイッチ)
    -Organize と併用する際に、「整理もしつつタグ付けもしたい」場合に指定する。

.PARAMETER Pixiv
    【Pixiv整理モード】 (スイッチ)
    Pixiv専用の整理を行う。末端フォルダ単位で全画像をスキャンし、R17以上を含むフォルダは全画像を一括移動する。移動後に空になったフォルダは削除する。

.PARAMETER NoReport
    【レポートなし】 (スイッチ)
    HTMLレポートの作成をスキップする。

.PARAMETER Recursive
    【再帰検索】 (スイッチ)
    サブフォルダも検索対象にする。

.PARAMETER NoRecursive
    【再帰なし】 (スイッチ)
    サブフォルダを検索しない。

.PARAMETER Thresh
    【タグ採用閾値】 (数値)
    タグ付けを採用する確率の閾値。

.PARAMETER Gpu
    【GPU使用】 (スイッチ)
    GPUを使用して高速化する。

.PARAMETER BatchSize
    【バッチサイズ】 (数値)
    推論をまとめて行う枚数（デフォルト: 4）。モデル非対応時は自動で 1 になる。

.PARAMETER IoWorkers
    【前処理ワーカー数】 (数値)
    画像の読み込み・前処理を並列化するワーカー数（デフォルト: 自動）。

.PARAMETER ModelProfile
    【DBV4モデルプロファイル】 (文字列)
    compact_manual / lightweight / medium_manual / balanced / high / ultra から選択する。

.PARAMETER ModelRepo
    【モデルリポジトリ】 (文字列)
    HuggingFaceのモデル/タグのリポジトリID。

.PARAMETER ModelFile
    【モデルファイル】 (文字列)
    モデルファイル名またはローカルパス。

.PARAMETER TagsFile
    【タグCSV】 (文字列)
    タグCSVファイル名またはローカルパス。

.PARAMETER Force
    【強制実行】 (スイッチ)
    既存タグがあっても強制的に再解析・上書きする。

.PARAMETER Server
    【サーバーモード】 (スイッチ)
    推論サーバーとして起動する。

.PARAMETER Client
    【クライアントモード】 (スイッチ)
    クライアントとして動作し、指定したサーバーへ画像を送信する。

.PARAMETER HostIP
    【ホストIP】 (文字列)
    サーバーのIPアドレス。

.PARAMETER Port
    【ポート】 (数値)
    ポート番号。

.PARAMETER SensitiveSplitMode
    【Sensitive分割モード】 (数値: 2, 4, 6)
    旧CLI互換用。DBV4ではR-15/R-17は5段階固定のため、指定時は非推奨警告を表示する。
    未指定時はconfig.jsonの設定を使用。

.PARAMETER RecordRatio
    【スコア記録】 (スイッチ)
    メタデータにRAWスコアおよび割合スコアを記録する。

.PARAMETER NoRecordRatio
    【スコア記録無効化】 (スイッチ)
    メタデータへのスコア記録を無効化する。

.PARAMETER RatingThresh
    【R指定閾値】 (数値)
    [旧機能] R指定タグ合計値による閾値判定。

.PARAMETER IgnoreSensitive
    【Sensitive無視】 (スイッチ)
    [旧機能] SensitiveをGeneralとして扱う。

.PARAMETER Help
    【ヘルプ表示】 (スイッチ)
    このヘルプを表示する。

.PARAMETER Login
    【Hugging Faceログイン】 (スイッチ)
    認証のみ実行して終了する。

.PARAMETER RemainingArgs
    未定義の引数（--helpなど）を捕捉するための内部パラメータ。

.EXAMPLE
    # 初回セットアップ (何もしない)
    .\run_tagger.ps1

    # Hugging Faceへログイン
    .\run_tagger.ps1 -Login

    # 通常実行 (タグ付け＋レポート)
    .\run_tagger.ps1 -Path "C:\Images" -Gpu

    # フォルダ整理のみ (タグ付けなし)
    .\run_tagger.ps1 -Path "C:\Images" -Organize

    # 全部入り (タグ付け＋整理＋レポート)
    .\run_tagger.ps1 -Path "C:\Images" -Tag -Organize -Gpu

    # highプロファイル＋RAWスコア記録有効
    .\run_tagger.ps1 -Path "C:\Images" -Gpu -ModelProfile high -RecordRatio

    # 推論スキップ高速再整理（DBV4 RAWスコアを利用）
    .\run_tagger.ps1 -Path "C:\Images" -Organize -ModelProfile balanced
#>

[CmdletBinding()]
param (
    [Alias('p')]
    [string]$Path,
    [Alias('o')]
    [switch]$Organize,
    [Alias('t')]
    [switch]$Tag,
    [Alias('z')]
    [switch]$NoReport,
    [Alias('r')]
    [switch]$Recursive,
    [Alias('n')]
    [switch]$NoRecursive,
    [Alias('q')]
    [Nullable[float]]$Thresh,
    [Alias('g')]
    [switch]$Gpu,
    [Alias('b')]
    [int]$BatchSize,
    [Alias('w')]
    [int]$IoWorkers,
    [Alias('m')]
    [string]$ModelProfile,
    [Alias('e')]
    [string]$ModelRepo,
    [Alias('l')]
    [string]$ModelFile,
    [Alias('y')]
    [string]$TagsFile,
    [Alias('f')]
    [switch]$Force,
    [Alias('s')]
    [switch]$Server,
    [Alias('c')]
    [switch]$Client,
    [Alias('j')]
    [string]$HostIP,
    [Alias('u')]
    [int]$Port,
    [Alias('x')]
    [switch]$Pixiv,

    [ValidateSet(2, 4, 6)]
    [Alias('v')]
    [int]$SensitiveSplitMode,
    [Alias('a')]
    [switch]$RecordRatio,
    [Alias('k')]
    [switch]$NoRecordRatio,
    
    # Old params
    [Alias('d')]
    [float]$RatingThresh,
    [Alias('i')]
    [switch]$IgnoreSensitive,
    
    [Alias('h')]
    [switch]$Help,

    [switch]$Login,

    [ValidateSet('cpu','cuda','tensorrt','intel','directml','webgpu')]
    [string]$Provider,
    [switch]$WebGpu,
    [ValidateRange(0,2147483647)][int]$GpuIndex = 0,
    [ValidateRange(0,2147483647)][int]$DirectMlDeviceIndex = 0,
    [ValidateRange(0,2147483647)][int]$WebGpuDeviceIndex,
    [ValidateSet('nvidia','intel','amd')][string]$TargetVendor,
    [string]$OpenVinoDevice,
    [string]$TensorRtLibDir,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

#region Help Function
function Show-Help {
    Write-Host "DBV4 Tagger Universal (日本語ヘルプ)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "使い方: .\run_tagger.ps1 [オプション] [パス]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  引数なしで実行すると「環境構築モード」となり、セットアップのみを行います。" -ForegroundColor Gray
    Write-Host ""
    Write-Host "主なオプション:" -ForegroundColor Yellow
    Write-Host "    -Path (-p) <path>     処理対象ファイル/フォルダ"
    Write-Host "    -Gpu (-g)             GPUを使用する（Windows: DirectML）"
    Write-Host "    -Organize (-o)        フォルダ整理のみ行う（タグ付けOFF）"
    Write-Host "    -Tag (-t)             タグ付けも行う（-Organize併用時）"
    Write-Host "    -Pixiv (-x)           Pixiv整理モード（末端フォルダ単位で全画像を一括移動）"
    Write-Host "    -NoReport (-z)        レポート作成なし"
    Write-Host "    -Recursive (-r)       再帰検索ON"
    Write-Host "    -NoRecursive (-n)     再帰検索OFF"
    Write-Host "    -Thresh (-q) <0.0-1.0> DBV4のtag best_thresholdを一括上書き（省略時はタグ固有値）"
    Write-Host "    -BatchSize (-b) <n>   推論バッチサイズ"
    Write-Host "    -IoWorkers (-w) <n>   前処理の並列ワーカー数"
    Write-Host "    -ModelProfile (-m) <name> モデルプロファイル (compact_manual/lightweight/medium_manual/balanced/high/ultra/wd14_v3/future_1b)"
    Write-Host "    -ModelRepo (-e) <repo> DBV4モデル/タグのHFリポジトリIDを明示指定"
    Write-Host "    -ModelFile (-l) <file> モデルファイル名またはパス"
    Write-Host "    -TagsFile (-y) <file> タグCSVファイル名またはパス"
    Write-Host "    -Force (-f)           既存タグがあっても強制的に再解析・上書き"
    Write-Host "    -Server (-s)          サーバーモード（推論待機）"
    Write-Host "    -Client (-c)          クライアントモード"
    Write-Host "    -HostIP (-j) <ip>     サーバーのIPアドレス"
    Write-Host "    -Port (-u) <port>     ポート番号"
    Write-Host "    -SensitiveSplitMode (-v) <2|4|6> 旧CLI互換（DBV4では5段階固定・非推奨）"
    Write-Host "    -RecordRatio (-a)     メタデータにRAWスコア・割合スコアを記録"
    Write-Host "    -NoRecordRatio (-k)   メタデータへのスコア記録を無効化"
    Write-Host "    -Help (-h, --help)    このヘルプを表示"
    Write-Host "    -Login               Hugging Faceへログインして終了"
    Write-Host "    -Provider NAME       cpu/cuda/tensorrt/intel/directml/webgpu"
    Write-Host "    -GpuIndex N / -DirectMlDeviceIndex N / -WebGpuDeviceIndex N"
    Write-Host "    -TargetVendor NAME / -OpenVinoDevice GPU.N / -TensorRtLibDir DIR"
    Write-Host "    -RatingThresh (-d)    旧CLI互換: 非General rating判定閾値"
    Write-Host "    -IgnoreSensitive (-i) 旧CLI互換: SensitiveをGeneralとして扱う"
    Write-Host ""
    Write-Host "実行例:" -ForegroundColor Yellow
    Write-Host "    # 初回セットアップ（何もしない）"
    Write-Host "    .\run_tagger.ps1"
    Write-Host ""
    Write-Host "    # Hugging Faceへログイン"
    Write-Host "    .\run_tagger.ps1 -Login"
    Write-Host ""
    Write-Host "    # 通常実行（タグ付け＋レポート＋GPU）"
    Write-Host "    .\run_tagger.ps1 -Path C:\Images -Gpu"
    Write-Host ""
    Write-Host "    # フォルダ整理のみ（タグ付けなし）"
    Write-Host "    .\run_tagger.ps1 -Path C:\Images -Organize"
    Write-Host ""
    Write-Host "    # 全部入り（タグ付け＋整理＋レポート＋GPU）"
    Write-Host "    .\run_tagger.ps1 -Path C:\Images -Organize -Tag -Gpu"
    Write-Host ""
    Write-Host "    # highプロファイル＋RAWスコア記録有効"
    Write-Host "    .\run_tagger.ps1 -Path C:\Images -Gpu -ModelProfile high -RecordRatio"
    Write-Host ""
    Write-Host "    # 推論スキップ高速再整理（DBV4 RAWスコアを利用）"
    Write-Host "    .\run_tagger.ps1 -Path C:\Images -Organize -ModelProfile balanced"
    Write-Host ""
}

if ($Help -or ($RemainingArgs -contains '--help') -or ($RemainingArgs -contains '-h')) { Show-Help; exit }
#endregion

#region Environment Setup
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDir "embed_tags_universal.py"

$IsWindowsOS = $true
if ($PSVersionTable.PSVersion.Major -ge 6) {
    if ([System.OperatingSystem]::IsLinux()) { $IsWindowsOS = $false }
}

function Prepare-Environment {
    param ([bool]$UseGpu, [bool]$IsClient, [string]$SelectedProvider)
    $ExtraPackages = @()
    
    if ($IsClient) {
        if (Test-Path (Join-Path $ScriptDir "venv_gpu")) {
            $EnvName = "Client (流用 venv_gpu)"
            $TargetVenv = Join-Path $ScriptDir "venv_gpu"
            $OnnxPackage = "onnxruntime-directml"
        }
        elseif (Test-Path (Join-Path $ScriptDir "venv_std")) {
            $EnvName = "Client (流用 venv_std)"
            $TargetVenv = Join-Path $ScriptDir "venv_std"
            $OnnxPackage = "onnxruntime"
        }
        else {
            $EnvName = "Client (軽量)"
            $TargetVenv = Join-Path $ScriptDir "venv_client"
            $OnnxPackage = ""
        }
    }
    else {
        if ($SelectedProvider) {
            $EnvName = $SelectedProvider
            $TargetVenv = Join-Path $ScriptDir "venv_$SelectedProvider"
            switch ($SelectedProvider) {
                'cpu' { $OnnxPackage = 'onnxruntime' }
                'cuda' { $OnnxPackage = 'onnxruntime-gpu[cuda,cudnn]<1.27' }
                'tensorrt' { $OnnxPackage = 'onnxruntime-gpu[cuda,cudnn]<1.27' }
                'intel' { $OnnxPackage = 'onnxruntime-openvino==1.24.1'; $ExtraPackages = @('openvino==2025.4.1') }
                'webgpu' { $OnnxPackage = 'onnxruntime'; $ExtraPackages = @('onnxruntime-ep-webgpu') }
                'directml' { $OnnxPackage = 'onnxruntime-directml'; $TargetVenv = Join-Path $ScriptDir 'venv_gpu' }
            }
        }
        elseif ($IsWindowsOS -and $UseGpu) {
            $EnvName = "GPU (DirectML)"
            $TargetVenv = Join-Path $ScriptDir "venv_gpu"
            $OnnxPackage = "onnxruntime-directml"
        }
        elseif (-not $UseGpu -and (Test-Path (Join-Path $ScriptDir "venv_gpu"))) {
            $EnvName = "CPU (流用 venv_gpu)"
            $TargetVenv = Join-Path $ScriptDir "venv_gpu"
            $OnnxPackage = "onnxruntime-directml"
        }
        else {
            $EnvName = "Standard (CPU)"
            $TargetVenv = Join-Path $ScriptDir "venv_std"
            $OnnxPackage = "onnxruntime"
        }
    }
    
    Write-Host "[INFO] 環境確認: $EnvName" -ForegroundColor Cyan
    if (Test-Path $TargetVenv) {
        $VenvPy = if ($IsWindowsOS) { Join-Path $TargetVenv "Scripts/python.exe" } else { Join-Path $TargetVenv "bin/python" }
        try {
            $verMinor = & $VenvPy -c "import sys; print(sys.version_info.minor)" 2>$null
            if ([int]$verMinor -ge 14) {
                Write-Host "  -> 既存環境が PyPI非対応の Python 3.$verMinor です。再作成します..." -ForegroundColor Yellow
                Remove-Item -Recurse -Force $TargetVenv
            }
        }
        catch {}
    }
    if (-not (Test-Path $TargetVenv)) {
        Write-Host "  -> 仮想環境を作成中..." -ForegroundColor Yellow
        $PyCmd = if ($IsWindowsOS) { "python" } else { "python3" }
        try {
            $sysVer = & $PyCmd -c "import sys; print(sys.version_info.minor)" 2>$null
            if ([int]$sysVer -ge 14) {
                foreach ($candidate in @("python3.13", "python3.12", "python3.11", "python3.10")) {
                    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
                        $PyCmd = $candidate
                        break
                    }
                }
            }
        }
        catch {}
        & $PyCmd -m venv $TargetVenv
        if ($LASTEXITCODE -ne 0) { throw "仮想環境の作成に失敗しました。" }
    }
    
    if ($IsWindowsOS) {
        $Bin = Join-Path $TargetVenv "Scripts"
        $PyEx = Join-Path $Bin "python.exe"
        $PipEx = Join-Path $Bin "pip.exe"
    }
    else {
        $Bin = Join-Path $TargetVenv "bin"
        $PyEx = Join-Path $Bin "python"
        $PipEx = Join-Path $Bin "pip"
    }

    Write-Host "  -> ライブラリの確認・インストールを行います..." -ForegroundColor Yellow
    $ReqFile = Join-Path $ScriptDir "requirements.txt"
    if ($OnnxPackage) {
        & $PipEx install -r $ReqFile $OnnxPackage @ExtraPackages -q | Out-Null
    }
    else {
        & $PipEx install -r $ReqFile -q | Out-Null
    }
    
    if ($LASTEXITCODE -ne 0) { throw "依存パッケージのインストールに失敗しました。" }
    return $PyEx
}
#endregion

#region Main Logic

if ($Login -or ($RemainingArgs -contains '--login')) {
    $HfExecutable = $null
    $VenvPython = $null
    foreach ($VenvName in @('venv_gpu', 'venv_std', 'venv_client', 'venv_webgpu', 'venv_intel', 'venv_amd', 'venv_cuda', 'venv_tensorrt', 'venv_cpu')) {
        $Candidate = Join-Path $ScriptDir "$VenvName/Scripts/hf.exe"
        if (Test-Path $Candidate) {
            $HfExecutable = $Candidate
            Write-Host "[INFO] 既存の仮想環境を使用します: $VenvName" -ForegroundColor Cyan
            break
        }
    }
    if (-not $HfExecutable) {
        $VenvPython = Prepare-Environment -UseGpu $false -IsClient $false
        $HfExecutable = Join-Path (Split-Path -Parent $VenvPython) 'hf.exe'
    }
    if (-not (Test-Path $HfExecutable)) {
        Write-Error "Hugging Face CLIが見つかりません: $HfExecutable"
        exit 1
    }
    & $HfExecutable auth login
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "[INFO] Hugging Faceログインが完了しました。" -ForegroundColor Green
    exit 0
}

# 引数が一つもない場合はセットアップモード
if ($PSBoundParameters.Count -eq 0 -and (-not $RemainingArgs)) {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "   DBV4 Tagger Universal - Setup Mode" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "引数が指定されなかったため、環境構築のみを行いました。"
    Write-Host "画像処理を行うには -Path オプションなどを指定してください。"
    Write-Host "使い方がわからない場合は -Help を参照するのじゃ。"
    
    # Config生成のために一度CPU環境で実行
    $Py = Prepare-Environment -UseGpu $false -IsClient $false
    & $Py $PythonScript --gen-config
    exit
}

# Explicit selection must agree with the environment that is installed.
if ($WebGpu) {
    if ($Provider -and $Provider -ne 'webgpu') { throw '-WebGpu conflicts with -Provider' }
    $Provider = 'webgpu'
}
if ($Provider) { $Gpu = $Provider -ne 'cpu' }
if ($Gpu -and -not $Provider) { $Provider = 'directml' }
if (-not $IsWindowsOS -and $Provider) { throw 'Linuxではrun_tagger.shを使用してください。' }
if ($Provider -eq 'directml' -and $TargetVendor -and -not $Client) {
    $ProbePython = Prepare-Environment -UseGpu $true -IsClient $false -SelectedProvider 'webgpu'
}
$VenvPython = Prepare-Environment -UseGpu $Gpu -IsClient $Client -SelectedProvider $Provider

# Python引数構築
$PyArgs = @($PythonScript)

# モード設定
if ($Server) { $PyArgs += ("--mode", "server") }
elseif ($Client) { $PyArgs += ("--mode", "client") }
else { $PyArgs += ("--mode", "standalone") }

# アクション設定
# Organize指定時 -> デフォルトでNo-Tag扱いになる。Tag指定があればタグも有効。
if ($Organize) {
    $PyArgs += "--organize"
    if (-not $Tag -and -not $Pixiv) { $PyArgs += "--no-tag" }
}
if ($Pixiv) {
    $PyArgs += "--pixiv"
}
else {
    # 通常モード -> Tag指定は不要(デフォルトON)。No-Tag指定があれば...無いので実装不要
    # もし将来的に「タグなし・整理なし・レポートのみ」をするなら --no-tag 引数が必要だが
    # 今回のPSラッパーでは Organize がスイッチになっているため自動制御する
}

if ($NoReport) { $PyArgs += "--no-report" }

# 再帰設定
if ($Recursive) { $PyArgs += "--recursive" }
if ($NoRecursive) { $PyArgs += "--no-recursive" }

# その他パラメータ
if ($PSBoundParameters.ContainsKey("Thresh")) { $PyArgs += ("--thresh", $Thresh) }
if ($Gpu) { $PyArgs += "--gpu" }
if ($Provider) { $PyArgs += @('--provider', $Provider) }
$PyArgs += @('--gpu-index', "$GpuIndex", '--directml-device-index', "$DirectMlDeviceIndex")
if ($PSBoundParameters.ContainsKey('WebGpuDeviceIndex')) { $PyArgs += @('--webgpu-device-index', "$WebGpuDeviceIndex") }
if ($TargetVendor) { $PyArgs += @('--target-vendor', $TargetVendor) }
if ($OpenVinoDevice) { $PyArgs += @('--openvino-device', $OpenVinoDevice) }
if ($TensorRtLibDir) { $PyArgs += @('--tensorrt-lib-dir', $TensorRtLibDir) }
if ($PSBoundParameters.ContainsKey('BatchSize')) { $PyArgs += ("--batch-size", $BatchSize) }
if ($PSBoundParameters.ContainsKey('IoWorkers')) { $PyArgs += ("--io-workers", $IoWorkers) }
if ($PSBoundParameters.ContainsKey('ModelProfile')) { $PyArgs += ("--model-profile", $ModelProfile) }
if ($PSBoundParameters.ContainsKey('ModelRepo')) { $PyArgs += ("--model-repo", $ModelRepo) }
if ($PSBoundParameters.ContainsKey('ModelFile')) { $PyArgs += ("--model-file", $ModelFile) }
if ($PSBoundParameters.ContainsKey('TagsFile')) { $PyArgs += ("--tags-file", $TagsFile) }
if ($Force) { $PyArgs += "--force" }
if ($PSBoundParameters.ContainsKey('SensitiveSplitMode')) { $PyArgs += ("--sensitive-split-mode", $SensitiveSplitMode) }
if ($RecordRatio) { $PyArgs += "--record-ratio" }
if ($NoRecordRatio) { $PyArgs += "--no-record-ratio" }

if ($HostIP) { $PyArgs += ("--host", $HostIP) }
if ($Port) { $PyArgs += ("--port", $Port) }

# Old Params
if ($PSBoundParameters.ContainsKey('RatingThresh')) { $PyArgs += ("--rating-thresh", $RatingThresh) }
if ($IgnoreSensitive) { $PyArgs += "--ignore-sensitive" }

# 最後にパス
if ($Path) { $PyArgs += $Path }

# 実行
Write-Host "[INFO] Pythonスクリプトを実行..." -ForegroundColor Green
& $VenvPython @PyArgs
exit $LASTEXITCODE

#endregion
