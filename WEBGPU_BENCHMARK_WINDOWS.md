# Windows ベンチマーク：CUDA・TensorRT・Intel・DirectML・WebGPU

Windows と Linux は別セッションで測る。Windows を先に実行する。測定と環境調査はユーザーが行い、AI には集計 JSON と失敗条件だけ渡して解析・結果文書を更新させる。古い LINUX_HANDOFF.md は参照しない。

## 1. 準備

git status --short、branch、HEAD を記録する。Windows、NVIDIA/Intel GPU、ドライバー、CUDA/cuDNN/TensorRT、Python、電源モードをユーザーが記録する。NVIDIA は nvidia-smi、Intel はデバイスマネージャー等で実機を確認する。各 EP を独立した仮想環境へ入れる。Python 3.13 の wheel がなければ 3.12 を使い、以下の py -3.13 を読み替える。TensorRT は ONNX Runtime と整合する CUDA/cuDNN/TensorRT 10 ランタイム（nvinfer_10.dll）を別途導入する。run_tagger.ps1 -Gpu は DirectML 経路なので CUDA 比較には使わない。

~~~powershell
$packages = @{
    cuda = @("onnxruntime-gpu")
    tensorrt = @("onnxruntime-gpu")
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

各環境で目的の EP が列挙されることを確認する。Hugging Face 認証が必要なら端末で行い、トークンを AI に渡さない。

## 2. ユーザーが測定する

NVIDIA の CUDA、TensorRT、WebGPU と、Intel の OpenVINO、DirectML、WebGPU を別ディレクトリに測る。Intel GPU がない場合は Intel 側を実行せず、その旨を記録する。wd14_v3 と balanced、batch size 1 と 4、warmup 3 回、20 回×3 セットを共通にする。画像は seed 固定の合成 640×480 RGB で、モデル取得・セッション作成・前処理は推論時間から除かれる。

~~~powershell
foreach ($profile in @("wd14_v3", "balanced")) {
    foreach ($batch in @(1, 4)) {
        foreach ($provider in @("cuda", "tensorrt", "webgpu")) {
            $python = ".\.venv_bench_$provider\Scripts\python.exe"
            $extra = if ($provider -eq "webgpu") { @("--track-nvidia-memory") } else { @() }
            & $python benchmark_nvidia_ep.py --provider $provider --profile $profile --batch-size $batch --gpu-index 0 @extra --output "benchmarks/nvidia_windows/$($profile)-b$($batch)-$($provider).json"
        }
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

出力 JSON に実行ノードの EP、各回の時間、プロセス RAM、NVIDIA 指定時のみ VRAM 増分が残る。目的 EP が実行ノードに現れない条件は失敗になる。WebGPU の NVIDIA VRAM 増分が観測されなければ --webgpu-device-index を変えて再測定し、GPU 名を記録する。Intel の VRAM は計測しない。失敗条件を成功値で埋めない。

## 3. AI に渡すもの

nvidia_windows_summary.json と intel_windows_summary.json、各 incomplete 条件の JSON またはコンソールエラー、GPU と EP の対応・ドライバー等の環境メモを渡す。AI は同一 GPU・同一 OS 内で CUDA 対 TensorRT/WebGPU、DirectML 対 OpenVINO/WebGPU を解析する。異なる GPU の比率を混ぜない。必要な生データだけ追加で確認し、Windows 結果を同じ branch と PR に commit・通常 push する。Linux セッションへ branch、SHA、実行条件を渡す。AMD 測定は別 GPU の参考値として扱う。
