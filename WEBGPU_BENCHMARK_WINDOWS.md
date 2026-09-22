# Windows ベンチマーク手順（NVIDIA・Intel iGPU・AMD・CPU）

測定・GPU 調査はユーザーが実施し、`summary.json` と失敗 JSON、環境メモを AI に渡して解析・結果文書を更新する。古い Linux ハンドオーバーは参照しない。既存の測定 JSON は残し、今回の結果は新しいディレクトリに保存する。

## 1. 条件と対象

すべての GPU 群で CPU を測定する。NVIDIA は CUDA、TensorRT、DirectML、WebGPU、Intel は OpenVINO GPU、DirectML、WebGPU、AMD は DirectML、WebGPU が対象。モデルは `wd14_v3` と `balanced`、batch は 1 と 4、固定 seed の 640×480 合成 RGB、warmup 3、20 回×3 セット。モデル取得、セッション作成、前処理は推論時間に含まない。各 EP は別の仮想環境で実行する。導入できない EP や実行に失敗した EP は JSON に残り、成功値にはしない。使用可能なものはすべて測るが、使えない EP の導入を無理に成功扱いしない。

先に `git status --short` と HEAD、Windows 版、CPU/GPU の正式名、ドライバー、Python、電源設定を記録する。この PC の Intel UHD Graphics 630 は QSV/D3D11VA、OpenCL、WebGPU から認識されており、画面出力は不要。OpenVINO 2025.4.1 だけは現在 `GPU` に NVIDIA を表示するため、Intel の OpenVINO 条件は失敗として記録し、DirectML/WebGPU の Intel 条件は測定できる。

## 2. 仮想環境

Python 3.13 用 wheel がなければ 3.12 を使用する。既存環境を再利用できるが、各環境の ONNX Runtime と GPU ランタイム版を確認する。追加パッケージは必ず仮想環境に入れる。

```powershell
$packages = @{
    cpu = @('onnxruntime')
    cuda = @('onnxruntime-gpu[cuda,cudnn]<1.27')
    tensorrt = @('onnxruntime-gpu[cuda,cudnn]<1.27')
    directml = @('onnxruntime-directml')
    webgpu = @('onnxruntime', 'onnxruntime-ep-webgpu')
    intel = @('onnxruntime-openvino==1.24.1', 'openvino==2025.4.1')
}
foreach ($provider in $packages.Keys) {
    py -3.13 -m venv ".venv_bench_$provider"
    $python = ".\.venv_bench_$provider\Scripts\python.exe"
    & $python -m pip install -r requirements.txt psutil
    & $python -m pip install $packages[$provider]
    & $python -c 'import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())'
}
```

CUDA/cuDNN DLL は測定コードで読み込む。TensorRT は別に TensorRT 10 の `nvinfer_10.dll` が必要で、対応する CUDA 12 版を導入し、実行する PowerShell の `PATH` に TensorRT の `bin` を追加する。例: `$env:PATH = "C:\Users\kouki\Downloads\TensorRT-10.16.1.11\bin;$env:PATH"`。DLL と ONNX Runtime の互換性を確認する。EP が列挙されても推論成功の保証にはならない。

Intel の OpenVINO DLL は同じ仮想環境の `openvino\libs` から測定コードが読み込む。まず GPU の実名を確認する。

```powershell
& .\.venv_bench_intel\Scripts\python.exe -c "import openvino as ov; c=ov.Core(); print([(d, c.get_property(d, 'FULL_DEVICE_NAME')) for d in c.available_devices])"
```

Intel 名の `GPU.N` が見えない場合は OpenVINO 条件を成功扱いしない。Intel の `OpenVINOExecutionProvider` は実行時にも製造元名を検証する。DirectML の `device_id` と WebGPU のデバイス番号は別体系。次のプローブで一覧を表示し、ランナーも製造元を検証する。DirectML は `DxgiAdapterNumber` を使う。

```powershell
& .\.venv_bench_webgpu\Scripts\python.exe probe_webgpu_adapters.py
```

この PC で実際に確認した番号は Intel UHD Graphics 630 が WebGPU 0/1/3、DirectML/DXGI 3/1/2、NVIDIA RTX 2070 が WebGPU 2、DirectML/DXGI 0。Intel の複数アダプターは同名なので、同じ実機か確認して選ぶ。`--device-name` は確認した名称を記録する項目であり、この文字列だけで DirectML/WebGPU の物理デバイスが自動検証されるわけではない。NVIDIA では `nvidia-smi` の VRAM 増分も補助的に確認する。GPU と EP の対応が不明ならその条件を比較に採用しない。

## 3. 測定

`benchmark_matrix.ps1 -h` で全オプションを表示する。以下は各 GPU があるホストで個別に実行する例。`--device-name` は実機名に置き換え、`GpuIndex` / `DirectMlDeviceIndex` / `WebGpuDeviceIndex` / `OpenVinoDevice` は確認した番号に変更する。同時に複数の測定を走らせない。モデル取得に認証が必要な場合は端末で行い、トークンを AI に渡さない。

```powershell
.\benchmark_matrix.ps1 -Vendor nvidia -DeviceName 'NVIDIA GeForce RTX 2070 with Max-Q Design' -OutputDir 'benchmarks\nvidia_windows_20260923' -GpuIndex 0 -DirectMlDeviceIndex 0 -WebGpuDeviceIndex 2
.\benchmark_matrix.ps1 -Vendor intel -DeviceName 'Intel UHD Graphics 630' -OutputDir 'benchmarks\intel_windows_20260923' -GpuIndex 0 -DirectMlDeviceIndex 1 -WebGpuDeviceIndex 1 -OpenVinoDevice 'GPU.0'
.\benchmark_matrix.ps1 -Vendor amd -DeviceName 'AMD Radeon Graphics' -OutputDir 'benchmarks\amd_windows_20260923' -GpuIndex 0 -DirectMlDeviceIndex 0 -WebGpuDeviceIndex 0
```

Intel の例の DirectML 1 と WebGPU 1 はこの PC で Intel として列挙された番号。OpenVINO `GPU.0` は現状 NVIDIA なので Intel 条件では失敗する。CPU 環境がない場合も結果は `skipped` になるため、最初に CPU 環境を準備する。特定 EP だけ再測定する場合は `--providers` を渡せるが、CPU も再測定される。結果を混ぜないため、新しい出力先を使う。

各条件の JSON に `status`、実行したノードの EP、各回の時間、RSS が残る。NVIDIA 指定時の VRAM 増分は GPU 選択の補助情報。`D3D12CreateDevice` 警告や CPU ノード割当警告だけでは成否を決めず、JSON の `status` と目的 EP の実行ノードを確認する。失敗した条件を修正後、同条件だけ再測定してもよい。

## 4. AI に渡すもの

各出力先の `summary.json`、`manifest.json`、`incomplete` に載った条件の JSON とコンソールエラー、OS/CPU/GPU/ドライバー/電源設定、各 EP が実際に選んだアダプターの確認メモを渡す。AI は同一ホスト・モデル・batch の CPU 比と GPU EP 間を解析する。Windows と Linux、異なる GPU の比率は環境差を明示する。AMD も同じ手順・同じ条件で測り直し、古い AMD 値を今回の比較に混ぜない。既存 PR の branch を継続する。
