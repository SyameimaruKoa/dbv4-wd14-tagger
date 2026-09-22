# Windows ベンチマーク：CUDA・TensorRT・Intel・DirectML・WebGPU

Windows と Linux は別セッションで測る。Windows を先に実行する。測定と環境調査はユーザーが行い、AI には集計 JSON と失敗条件だけ渡して解析・結果文書を更新させる。古い LINUX_HANDOFF.md は参照しない。

## 1. 準備

git status --short、branch、HEAD を記録する。Windows、NVIDIA/Intel GPU、ドライバー、CUDA/cuDNN/TensorRT、Python、電源モードをユーザーが記録する。NVIDIA は nvidia-smi、Intel はデバイスマネージャー等で実機を確認する。各 EP を独立した仮想環境へ入れる。Python 3.13 の wheel がなければ 3.12 を使い、以下の py -3.13 を読み替える。NVIDIA ドライバーの表示する CUDA バージョンは、CUDA ランタイム DLL の存在を保証しない。CUDA と TensorRT の仮想環境には CUDA 12 版の ONNX Runtime と CUDA/cuDNN DLL を揃える。TensorRT は対応する TensorRT 10 ランタイム（nvinfer_10.dll）も別途導入して PATH から読めるようにする。run_tagger.ps1 -Gpu は DirectML 経路なので CUDA 比較には使わない。

~~~powershell
$packages = @{
    cuda = @("onnxruntime-gpu[cuda,cudnn]<1.27")
    tensorrt = @("onnxruntime-gpu[cuda,cudnn]<1.27")
    webgpu = @("onnxruntime", "onnxruntime-ep-webgpu")
    intel = @("onnxruntime-openvino")
    directml = @("onnxruntime-directml")
}
foreach ($name in $packages.Keys) {
    py -3.13 -m venv ".venv_bench_$name"
    $python = ".\.venv_bench_$name\Scripts\python.exe"
    & $python -m pip install -r requirements.txt psutil
    & $python -m pip install $packages[$name]
    & $python -c "import onnxruntime as ort; print(ort.get_available_providers())"
}
~~~

既存の .venv_bench_cuda と .venv_bench_tensorrt で ONNX Runtime 1.30.0 が入っている場合も、上記の pip install で CUDA 12 版に入れ替える。CUDA/cuDNN の DLL は測定コードが ONNX Runtime の preload_dlls で読み込む。ONNX Runtime 1.26.0 が読み込み対象に含めていない cuDNN の cudnn_engines_tensor_ir64_9.dll は、仮想環境内にある場合に測定コードが追加で読み込む。TensorRT の DLL は別途確認する。各環境で目的の EP が列挙されることを確認する。ただし EP の列挙だけでは DLL 読み込み成功を意味しない。Hugging Face 認証が必要なら端末で行い、トークンを AI に渡さない。

### TensorRT の DLL がない場合

CUDA が成功しても、TensorRT は別のランタイムを必要とする。nvinfer_10.dll がない場合は NVIDIA の [TensorRT 10 Windows ZIP 導入手順](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/installing-tensorrt/install-zip.html)に従い、CUDA 12 対応の TensorRT 10 を取得・展開する。ONNX Runtime の仮想環境へ Python binding を入れるだけでは、DLL の検索パスが設定されたとは限らない。展開先を次の $trtRoot に指定し、**同じ PowerShell セッション**で確認してから TensorRT 条件を再実行する。

~~~powershell
$trtRoot = "C:\path\to\TensorRT-10.x.x.x"
$trtDll = Get-ChildItem -LiteralPath $trtRoot -Recurse -File -Filter "nvinfer_10.dll" | Select-Object -First 1
if (-not $trtDll) {
    throw "nvinfer_10.dll が TensorRT 展開先に見つかりません"
}
$env:PATH = "$($trtDll.DirectoryName);$env:PATH"
$env:TENSORRT_LIB_DIR = $trtDll.DirectoryName
& .\.venv_bench_tensorrt\Scripts\python.exe -c "import ctypes, os, onnxruntime as ort; ort.preload_dlls(); h=os.add_dll_directory(os.environ['TENSORRT_LIB_DIR']); ctypes.WinDLL(os.path.join(os.environ['TENSORRT_LIB_DIR'], 'nvinfer_10.dll')); print('TensorRT DLL OK')"
~~~

TensorRT の DLL と CUDA/cuDNN の版が ONNX Runtime と整合することを確認する。TensorRT が使えない間も CUDA と WebGPU の結果は保持し、TensorRT 条件だけ失敗として記録する。

## 2. ユーザーが測定する

NVIDIA の CUDA、TensorRT、WebGPU と、Intel の OpenVINO、DirectML、WebGPU を別ディレクトリに測る。Intel GPU がない場合は Intel 側を実行せず、その旨を記録する。wd14_v3 と balanced、batch size 1 と 4、warmup 3 回、20 回×3 セットを共通にする。画像は seed 固定の合成 640×480 RGB で、モデル取得・セッション作成・前処理は推論時間から除かれる。

~~~powershell
foreach ($profile in @("wd14_v3", "balanced")) {
    foreach ($batch in @(1, 4)) {
        foreach ($provider in @("cuda", "tensorrt")) {
            $python = ".\.venv_bench_$provider\Scripts\python.exe"
            & $python benchmark_nvidia_ep.py --provider $provider --profile $profile --batch-size $batch --gpu-index 0 --output "benchmarks/nvidia_windows/$($profile)-b$($batch)-$($provider).json"
        }
        & .\.venv_bench_webgpu\Scripts\python.exe benchmark_nvidia_ep.py --provider webgpu --profile $profile --batch-size $batch --gpu-index 0 --track-nvidia-memory --output "benchmarks/nvidia_windows/$($profile)-b$($batch)-webgpu.json"
    }
}
py -3.13 summarize_nvidia_ep.py benchmarks/nvidia_windows --providers cuda tensorrt webgpu --reference cuda --output benchmarks/nvidia_windows_summary.json
~~~

Intel の WebGPU が GPU 0 以外なら --webgpu-device-index で選ぶ。DirectML の --gpu-index は DirectML の device_id、OpenVINO の --openvino-device は GPU.0 などの対象デバイスを指定する。三つの EP が同じ Intel GPU を使ったか確認する。

~~~powershell
foreach ($profile in @("wd14_v3", "balanced")) {
    foreach ($batch in @(1, 4)) {
        foreach ($provider in @("intel", "directml", "webgpu")) {
            $python = ".\.venv_bench_$provider\Scripts\python.exe"
            & $python benchmark_nvidia_ep.py --provider $provider --profile $profile --batch-size $batch --output "benchmarks/intel_windows/$($profile)-b$($batch)-$($provider).json"
        }
    }
}
py -3.13 summarize_nvidia_ep.py benchmarks/intel_windows --providers intel directml webgpu --reference directml --output benchmarks/intel_windows_summary.json
~~~

CUDA または TensorRT が失敗したら、出力 JSON とコンソールの DLL エラーを確認し、依存ライブラリを修正してから同条件を再実行する。CPUExecutionProvider だけの結果を GPU 測定に採用しない。

出力 JSON に実行ノードの EP、各回の時間、プロセス RAM、NVIDIA 指定時のみ VRAM 増分が残る。目的 EP が実行ノードに現れない条件は失敗になる。WebGPU の NVIDIA VRAM 増分が観測されなければ --webgpu-device-index を変えて再測定し、GPU 名を記録する。Intel の VRAM は計測しない。D3D12CreateDevice 警告が出ても、WebGPU の JSON が status=ok で、executed_node_providers に WebGpuExecutionProvider があり、NVIDIA VRAM 増分も観測されれば測定値を保持し、警告を環境メモに残す。実際に使われた Dawn backend は別途確認する。失敗条件を成功値で埋めない。

## 3. AI に渡すもの

nvidia_windows_summary.json と intel_windows_summary.json、各 incomplete 条件の JSON またはコンソールエラー、GPU と EP の対応・ドライバー等の環境メモを渡す。AI は同一 GPU・同一 OS 内で CUDA 対 TensorRT/WebGPU、DirectML 対 OpenVINO/WebGPU を解析する。異なる GPU の比率を混ぜない。必要な生データだけ追加で確認し、Windows 結果を同じ branch と PR に commit・通常 push する。Linux セッションへ branch、SHA、実行条件を渡す。AMD 測定は別 GPU の参考値として扱う。
