# Windows＋NVIDIA 本体反映後の検証手順

対象: [GPU初期化修正](GPU_RUNTIME_PORT.md)を含むコミット。**この文書は実施手順であり、Windowsでの合格記録ではない。** このPCで実施した内容は[ローカル検証結果](LOCAL_VALIDATION.md)を参照。

ベンチマークだけでなく、`run_tagger.ps1`から通常タグ付けとServerを実行する。対象モデルは`balanced`と`wd14_v3`、batchは1と4、EPはCPU・CUDA・TensorRT・DirectML・WebGPU。

## 1. 環境とテスト画像

リポジトリ直下のPowerShellで実行する。Python 3.13、ExifTool（`exiftool`としてPATH上にあること）、CUDA 12用ランタイムを実行できるNVIDIAドライバーを用意する。TensorRTはCUDA 12用のTensorRT 10を展開し、`nvinfer_10.dll`のある`bin`を指定する。モデル取得で認証を求められる場合は`.\run_tagger.ps1 -Login`を実行する。

```powershell
$Work = Join-Path $env:TEMP ('dbv4-validation-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$Fixture = Join-Path $Work 'fixture'
New-Item -ItemType Directory -Path $Fixture -Force | Out-Null
Start-Transcript -Path (Join-Path $Work 'transcript.txt')
git rev-parse HEAD
$PSVersionTable
nvidia-smi
Get-Command exiftool
exiftool -ver
py -3.13 --version

# 検証用CPU環境。本体の各EP環境とは分離する。
py -3.13 -m venv .venv_verify
& .\.venv_verify\Scripts\python.exe -m pip install -r requirements.txt onnxruntime httpx
if ($LASTEXITCODE -ne 0) { throw '検証環境の準備に失敗' }
@'
import sys
from pathlib import Path
from PIL import Image
root = Path(sys.argv[1])
for i, color in enumerate([(127,63,191),(255,255,255),(0,0,0),(32,128,64)]):
    Image.new('RGB', (640,480), color).save(root / f'test-{i}.png')
'@ | & .\.venv_verify\Scripts\python.exe - $Fixture
if ($LASTEXITCODE -ne 0) { throw '画像生成に失敗' }

# PowerShell構文とヘルプ
$Tokens = $null; $ParseErrors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $PWD 'run_tagger.ps1'), [ref]$Tokens, [ref]$ParseErrors) | Out-Null
if ($ParseErrors.Count) { $ParseErrors; throw 'PowerShell構文エラー' }
.\run_tagger.ps1 -Help
& .\.venv_verify\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
if ($LASTEXITCODE -ne 0) { throw '回帰テスト失敗' }
```

WindowsではPOSIX用Bash実行テストとLinux共有ライブラリのテストがskipになる。Windows DLLに関する単体テストはモックであり、以下の実機試験も必要。

## 2. NVIDIAの番号を特定

CUDA/TensorRTの番号、DirectMLのDXGI番号、WebGPU番号は別体系。過去の測定値や同じ番号を流用しない。

```powershell
nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version --format=csv

# DirectMLの製造元照合にも利用する独立したWebGPU環境
py -3.13 -m venv venv_webgpu
& .\venv_webgpu\Scripts\python.exe -m pip install -r requirements.txt onnxruntime onnxruntime-ep-webgpu
if ($LASTEXITCODE -ne 0) { throw 'WebGPU環境の準備に失敗' }
& .\venv_webgpu\Scripts\python.exe probe_webgpu_adapters.py --json |
    Tee-Object -FilePath (Join-Path $Work 'adapters.json')
if ($LASTEXITCODE -ne 0) { throw 'アダプター列挙に失敗' }

# 以下は例。上記出力で確認したNVIDIAの値に置き換える。
$CudaIndex = 0
$DxgiIndex = 0
$WebGpuIndex = 2
$TrtBin = 'C:\TensorRT-10\bin'
if (-not (Test-Path (Join-Path $TrtBin 'nvinfer_10.dll'))) { throw 'TensorRTの場所を設定してください' }
```

可能ならタスクマネージャーのGPUエンジン、`nvidia-smi`、ドライバー診断も併用してNVIDIA実機の負荷を記録する。EP名・アダプター列挙だけを物理GPUの負荷の証明にはしない。特に複数GPU搭載PCのWebGPUで確認する。

## 3. 通常ランチャー: 全20条件

ケースごとに合成画像をコピーするため、利用者の画像を変更しない。`-Force`で既存scoreによる推論スキップを避ける。各ケースは新規プロセスで実行し、終了コード・ログ・XMPを保存する。1つの失敗で後続のケースを中断しない。

```powershell
$Results = @()
foreach ($Profile in @('balanced','wd14_v3')) {
    foreach ($Batch in @(1,4)) {
        foreach ($Ep in @('cpu','cuda','tensorrt','directml','webgpu')) {
            $CaseName = "$Profile-b$Batch-$Ep"
            $CaseDir = Join-Path $Work $CaseName
            New-Item -ItemType Directory -Path $CaseDir | Out-Null
            Copy-Item (Join-Path $Fixture '*.png') $CaseDir
            $Options = @{
                Provider = $Ep; ModelProfile = $Profile; BatchSize = $Batch
                Path = $CaseDir; Force = $true; NoReport = $true
            }
            switch ($Ep) {
                'cuda'     { $Options.GpuIndex = $CudaIndex; $Options.TargetVendor = 'nvidia' }
                'tensorrt' { $Options.GpuIndex = $CudaIndex; $Options.TargetVendor = 'nvidia'; $Options.TensorRtLibDir = $TrtBin }
                'directml' { $Options.DirectMlDeviceIndex = $DxgiIndex; $Options.TargetVendor = 'nvidia' }
                'webgpu'   { $Options.WebGpuDeviceIndex = $WebGpuIndex; $Options.TargetVendor = 'nvidia' }
            }
            try {
                & .\run_tagger.ps1 @Options *> (Join-Path $Work "$CaseName.log")
                $ExitCode = $LASTEXITCODE
            } catch {
                $_ | Out-String | Add-Content (Join-Path $Work "$CaseName.log")
                $ExitCode = 1
            }
            $Results += [pscustomobject]@{Case=$CaseName; ExitCode=$ExitCode; Review='未確認'}
            & exiftool -j -XMP:All $CaseDir > (Join-Path $Work "$CaseName-xmp.json")
        }
    }
}
$Results | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $Work 'cases.csv')
$Results | Format-Table
```

### 合格条件

- 全20条件の終了コードが0。各ログの推論実行4枚・スキップ0枚・タグ書き込み4枚を確認する。画像単位の失敗が最終終了コードに反映されない場合もあるため、コード0だけで合格にしない。
- `balanced`のラベル数12,476、`wd14_v3`は10,861。shape mismatch、DLLロード失敗、画像処理エラーがない。
- GPUの「起動検証で実行されたEP」に下表の要求EPがある。CPUノードとの混在は許容する。
- DirectMLの`balanced`で`logits`消失やCPUのみへのfallbackが再発しない。
- 起動検証を通過した後も、実画像処理まで完了し、ケース内PNGのXMPが保存されている。

| 指定 | 必須の実行EP | 特に確認する点 |
| --- | --- | --- |
| cpu | activeがCPUExecutionProvider | GPU不要の基準動作 |
| cuda | CUDAExecutionProvider | cuDNN tensor IR DLLのロード失敗なし |
| tensorrt | TensorrtExecutionProvider | CUDAだけでは不合格。初回構築と再実行を区別 |
| directml | DmlExecutionProvider | NVIDIA DXGI指定とCAFormer最適化回避 |
| webgpu | WebGpuExecutionProvider | NVIDIAの選択情報と実機負荷 |

`-Gpu`のみの従来起動も別途1ケース実行し、DirectML経路を維持していることを確認する。TensorRTは同じ条件を再実行し、初回後も正常に起動するか確認する。初回の環境構築・モデル取得・engine構築を通常推論時間と混ぜない。

## 4. Server / Client

ターミナルAで起動する。CUDAで実施後、停止してTensorRTでも繰り返す。`GPU.0`などのOpenVINO番号はこの試験では使わない。

```powershell
.\run_tagger.ps1 -Server -Provider cuda -GpuIndex $CudaIndex -ModelProfile balanced -Port 5051
# CUDA試験をCtrl+Cで終了してから:
.\run_tagger.ps1 -Server -Provider tensorrt -GpuIndex $CudaIndex -TensorRtLibDir $TrtBin -ModelProfile balanced -Port 5051
```

ターミナルBはリポジトリ直下で実行。`$Work`はターミナルAの作業フォルダーの実パスに設定する。

```powershell
$Work = 'C:\Users\YOUR_NAME\AppData\Local\Temp\dbv4-validation-YYYYMMDD-HHMMSS'
Invoke-RestMethod http://127.0.0.1:5051/metadata | ConvertTo-Json
.\run_tagger.ps1 -Client -HostIP 127.0.0.1 -Port 5051 -ModelProfile balanced -Path (Join-Path $Work 'fixture') -Force -NoReport
$LASTEXITCODE
```

合格条件: Server側で指定GPU EPのノード検証が成功し、`/metadata`のモデルがbalanced、Clientの4枚処理とXMP保存が成功する。ClientがローカルGPUを必要としないことも確認する。ログは各ターミナルで保存する。

## 5. 異常系: CPUや別GPUへ黙って切り替えない

成功ケースの環境を準備した後に行う。いずれも**非0終了・具体的な理由・利用者画像の処理開始前に停止**が合格。合成画像だけを使う。

```powershell
# 明示TensorRTパス欠落: PATH上に別TensorRTがあっても代用しない
.\run_tagger.ps1 -Provider tensorrt -TensorRtLibDir (Join-Path $Work 'missing-trt') -Path $Fixture -Force -NoReport
$LASTEXITCODE

# 範囲外WebGPU番号
.\run_tagger.ps1 -Provider webgpu -WebGpuDeviceIndex 99999 -TargetVendor nvidia -Path $Fixture -Force -NoReport
$LASTEXITCODE

# 選択したNVIDIAデバイスと製造元指定が不一致
.\run_tagger.ps1 -Provider webgpu -WebGpuDeviceIndex $WebGpuIndex -TargetVendor intel -Path $Fixture -Force -NoReport
$LASTEXITCODE
.\run_tagger.ps1 -Provider directml -DirectMlDeviceIndex $DxgiIndex -TargetVendor intel -Path $Fixture -Force -NoReport
$LASTEXITCODE

# CUDAがないCPU専用環境でCUDAを明示
& .\.venv_verify\Scripts\python.exe embed_tags_universal.py --provider cuda --model-profile balanced --no-tag --no-report $Fixture
$LASTEXITCODE
```

TensorRTの自動探索も確認する場合は、`TENSORRT_LIB_DIR`または`PATH`へ配置し、`-TensorRtLibDir`を省略する。候補が複数ある場合は停止して明示指定を要求すること。DLLの削除・リネームで既存環境を壊す試験は不要。

## 6. 数値・性能の確認と提出物

合成画像の成功はタグ品質の保証ではない。必要なら同じ複製画像をCPU/GPUで処理し、XMPのrating scoreとタグ差を確認する。差があれば入力・前処理・metadata version・モデル・batch・精度設定を揃えて調査し、速度が出たことだけで合格にしない。

性能を再測定する場合は[既存Windowsベンチマーク手順](WEBGPU_BENCHMARK_WINDOWS.md)のマトリクスを新規ディレクトリで実行する。今回の短い動作試験と既存の20回×3セット測定を同じ性能表に混ぜない。

保存するもの:

- コミットSHA、Windows/Python/PowerShell/ドライバー版、GPU名、電源設定。
- `adapters.json`と実際に指定した3種類の番号、TensorRTパス・版。
- `cases.csv`の各行へ合否を記入し、20ケースのログとXMP、Server/Client・異常系のログを添付。
- 各`venv_*\Scripts\python.exe -m pip freeze`の結果。認証トークンは記録しない。
- 再測定した場合のみ、新しい`manifest.json`・`summary.json`・個別結果JSON。

```powershell
Stop-Transcript
Write-Host "検証記録: $Work"
```
