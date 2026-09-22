# Linux ベンチマーク：CUDA・TensorRT・Intel・WebGPU

Windows 結果が同じ branch に commit・push された後、別の Linux セッションで始める。古い LINUX_HANDOFF.md は参照しない。測定と環境調査はユーザーが行い、AI は集計結果を解析して結果文書を更新する。Windows の測定ファイルは変更しない。DirectML は Windows 用 EP なので Linux では測らず、Windows の DirectML 結果と Linux の Intel 結果を同一 OS 内の比率として扱わない。

## 1. 準備

git status --short、branch、HEAD と Windows から引き継いだ SHA を確認する。ディストリビューション、kernel、NVIDIA/Intel GPU、ドライバー、CUDA/cuDNN/TensorRT、Vulkan、Python、電源状態をユーザーが記録する。nvidia-smi、vulkaninfo --summary、lspci 等で対象 GPU を確認し、llvmpipe を対象にしない。Python 3.13 の wheel がなければ以下の python3.13 を 3.12 に読み替える。各 EP は独立した仮想環境に入れる。TensorRT は run_tagger.sh と同じく、ONNX Runtime に整合する CUDA/cuDNN と TensorRT 10 の共有ライブラリが必要で、必要なら LD_LIBRARY_PATH を設定する。

~~~bash
for provider in cuda tensorrt webgpu intel; do
    python3.13 -m venv ".venv_bench_$provider"
    ".venv_bench_$provider/bin/python" -m pip install -r requirements.txt psutil
done
.venv_bench_cuda/bin/python -m pip install 'onnxruntime-gpu[cuda,cudnn]<1.27'
.venv_bench_tensorrt/bin/python -m pip install 'onnxruntime-gpu[cuda,cudnn]<1.27' 'tensorrt-cu12<11'
.venv_bench_webgpu/bin/python -m pip install onnxruntime onnxruntime-ep-webgpu
.venv_bench_intel/bin/python -m pip install onnxruntime-openvino
for provider in cuda tensorrt webgpu intel; do
    ".venv_bench_$provider/bin/python" -c 'import onnxruntime as ort; print(ort.get_available_providers())'
done
~~~

CUDA 12 に対応する TensorRT 10 と共有ライブラリを揃える。ドライバーの CUDA 表示だけでは CUDA/cuDNN ランタイムが使えるとは限らない。目的 EP の列挙だけでなく、DLL/共有ライブラリを読み込んでノードが実行されることを結果 JSON で確認する。Hugging Face 認証は端末内で行い、トークンを AI に渡さない。

## 2. ユーザーが測定する

Windows と同じモデル、seed 固定の合成 640×480 RGB 入力、batch size 1 と 4、warmup 3 回、20 回×3 セットを使う。NVIDIA と Intel は別ディレクトリに記録する。Intel GPU がない場合は Intel 側を実行せず、その旨を記録する。

~~~bash
for profile in wd14_v3 balanced; do
    for batch in 1 4; do
        for provider in cuda tensorrt; do
            ".venv_bench_$provider/bin/python" benchmark_nvidia_ep.py --provider "$provider" --profile "$profile" --batch-size "$batch" --gpu-index 0 --output "benchmarks/nvidia_linux/$profile-b$batch-$provider.json"
        done
        .venv_bench_webgpu/bin/python benchmark_nvidia_ep.py --provider webgpu --profile "$profile" --batch-size "$batch" --gpu-index 0 --track-nvidia-memory --output "benchmarks/nvidia_linux/$profile-b$batch-webgpu.json"
    done
done
python3.13 summarize_nvidia_ep.py benchmarks/nvidia_linux --providers cuda tensorrt webgpu --reference cuda --output benchmarks/nvidia_linux_summary.json
~~~

Intel の WebGPU が別デバイスなら --webgpu-device-index を指定する。OpenVINO の GPU.0 と同じ Intel GPU が選ばれたことを確認する。

~~~bash
for profile in wd14_v3 balanced; do
    for batch in 1 4; do
        for provider in intel webgpu; do
            ".venv_bench_$provider/bin/python" benchmark_nvidia_ep.py --provider "$provider" --profile "$profile" --batch-size "$batch" --output "benchmarks/intel_linux/$profile-b$batch-$provider.json"
        done
    done
done
python3.13 summarize_nvidia_ep.py benchmarks/intel_linux --providers intel webgpu --reference intel --output benchmarks/intel_linux_summary.json
~~~

出力 JSON には実行ノードの EP、各回の時間、プロセス RAM、NVIDIA 指定時のみ VRAM 増分が残る。目的 EP のノードが実行されない条件は失敗になる。WebGPU で NVIDIA VRAM 増分が観測されなければ --webgpu-device-index を変えて再測定し、GPU 名を記録する。Intel の VRAM は計測しない。失敗条件を成功値で埋めない。

## 3. AI に渡すもの

nvidia_linux_summary.json と intel_linux_summary.json、各 incomplete 条件の JSON またはコンソールエラー、GPU と EP の対応・環境メモを渡す。AI は同一 OS・同一 GPU 内の比率を解析し、Windows と Linux の結果を環境差を明示して並べる。絶対速度の OS 間差を特定 EP 固有の損失とみなさない。必要な生データだけ追加で確認し、Linux 結果を同じ branch と PR に commit・通常 push する。
