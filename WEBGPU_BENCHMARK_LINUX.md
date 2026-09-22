# Linux 引継ぎ：NVIDIA の CUDA と WebGPU を測る

Windows の測定結果が commit・push された後、**別の Linux セッション**で開始する。AI は測定を逐次代行せず、ユーザーが [benchmark_nvidia_ep.py](benchmark_nvidia_ep.py) を実行して返した小さな集計結果を解析する。Windows の測定ファイルは変更しない。

## 1. 準備

`git status --short`、branch、HEAD を確認し、Windows から引き継いだ commit と一致させる。Linux distribution、kernel、NVIDIA GPU、ドライバー、CUDA、Vulkan、Python を記録する。`nvidia-smi` と `vulkaninfo --summary` で対象 GPU を確認し、llvmpipe を対象にしない。CUDA と WebGPU は別仮想環境へ導入する。Python 3.13 が無ければ、ONNX Runtime の wheel がある Python 3.12 を使う。

```bash
python3.13 -m venv .venv_bench_cuda
python3.13 -m venv .venv_bench_webgpu
.venv_bench_cuda/bin/python -m pip install -r requirements.txt onnxruntime-gpu psutil
.venv_bench_webgpu/bin/python -m pip install -r requirements.txt onnxruntime onnxruntime-ep-webgpu psutil
```

CUDA EP に必要な CUDA/cuDNN ライブラリが不足する場合は環境を修正してから測定する。CUDA EP が active でない結果、WebGPU が NVIDIA 以外の GPU を使った結果は比較から除く。Hugging Face のログインが必要なら端末内で行い、トークンをチャットへ貼らない。

## 2. ユーザーが実行するコマンド

Windows と同じモデル、seed 固定の合成 640×480 RGB 入力、batch size、warmup、反復回数を使う。出力先だけ Linux 用に分ける。

```bash
for profile in wd14_v3 balanced; do
    for batch in 1 4; do
        .venv_bench_cuda/bin/python benchmark_nvidia_ep.py --provider cuda --profile "$profile" --batch-size "$batch" --output "benchmarks/nvidia_linux/${profile}-b${batch}-cuda.json"
        .venv_bench_webgpu/bin/python benchmark_nvidia_ep.py --provider webgpu --profile "$profile" --batch-size "$batch" --output "benchmarks/nvidia_linux/${profile}-b${batch}-webgpu.json"
    done
done
python3 summarize_nvidia_ep.py benchmarks/nvidia_linux --output benchmarks/nvidia_linux_summary.json
```

WebGPU で NVIDIA の VRAM 増分が観測されなければ、`--webgpu-device-index 1` などで GPU を切り替えて同じ条件を再測定する。CPU fallback のみの成功扱いはしない。

## 3. AI に渡すもの

`benchmarks/nvidia_linux_summary.json` と、`incomplete` の条件の JSON またはコンソールエラーを渡す。AI は同一 OS 内の WebGPU/CUDA 比を解析し、Windows と Linux の結果を並べて、ドライバー・Dawn backend・温度・電源状態の違いを明示する。絶対速度の OS 間差を WebGPU 固有の損失とみなさない。必要な生データだけ追加で確認し、Linux 用結果を commit・通常 push する。
