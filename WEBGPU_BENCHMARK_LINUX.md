# Linux ベンチマーク手順（NVIDIA・Intel iGPU・AMD・CPU）

測定と GPU 調査はユーザーが行い、`summary.json` と失敗条件、環境メモを AI に渡して解析・結果文書を更新する。古い Linux ハンドオーバーは不要なので参照しない。Windows と同じ branch・PR を使い、今回の結果は新しいディレクトリに置く。

## 1. 条件と対象

CPU は全 GPU 群の基準。NVIDIA は CUDA、TensorRT、WebGPU、Intel は OpenVINO GPU、WebGPU、AMD は MIGraphX、WebGPU が対象。MIGraphX は ROCm 対応 GPU と一致する wheel/ランタイムがある場合に限る。AMD の古い iGPU など ROCm 非対応の機器では `skipped`/`failed` を保持し、WebGPU の結果だけ使う。DirectML は Windows 用。モデル・seed・画像・batch・warmup・反復回数は Windows と同一。

`git status --short` と HEAD、ディストリビューション、kernel、CPU/GPU 正式名、ドライバー、CUDA/ROCm/Vulkan、Python、電源設定を記録する。`lspci`、`nvidia-smi`、`vulkaninfo --summary`、OpenVINO のデバイス名などで対象 GPU を確認し、ソフトウェアレンダラーを GPU として記録しない。

## 2. 仮想環境

Python 3.13 用 wheel がなければ 3.12 を使う。依存パッケージは必ず各仮想環境に入れる。MIGraphX は対象 ROCm 版に一致する AMD 提供の ONNX Runtime wheel を選ぶ必要があるため、下記の共通 `pip install` には含めない。

```bash
for provider in cpu cuda tensorrt webgpu intel migraphx; do
    python3.13 -m venv ".venv_bench_$provider"
    ".venv_bench_$provider/bin/python" -m pip install -r requirements.txt psutil
done
.venv_bench_cpu/bin/python -m pip install onnxruntime
.venv_bench_cuda/bin/python -m pip install 'onnxruntime-gpu[cuda,cudnn]<1.27'
.venv_bench_tensorrt/bin/python -m pip install 'onnxruntime-gpu[cuda,cudnn]<1.27' 'tensorrt-cu12<11'
.venv_bench_webgpu/bin/python -m pip install onnxruntime onnxruntime-ep-webgpu
.venv_bench_intel/bin/python -m pip install 'onnxruntime-openvino==1.24.1' 'openvino==2025.4.1'
```

MIGraphX を使う場合は対象 GPU が ROCm に対応することを確認し、[AMD の ONNX Runtime/ROCm 手順](https://rocm.docs.amd.com/)で適合 wheel と ROCm ランタイムを `.venv_bench_migraphx` に入れる。実行前に `onnxruntime.get_available_providers()` に `MIGraphXExecutionProvider` があるか確認する。ROCm 版や GPU 世代が合わない場合はこの条件だけ失敗とする。TensorRT は対応する TensorRT 10 の共有ライブラリを用意する。ランナーは仮想環境内の `tensorrt*_libs`、`TENSORRT_LIB_DIR`、または `--tensorrt-lib-dir` から `libnvinfer.so.10` を検出し、TensorRT 条件の子プロセスへ `LD_LIBRARY_PATH` を渡す。開始時に TensorRT provider の読み込みも検証する。システム領域に導入した場合は通常の動的ローダーが解決できる状態にする。TensorRT がロードできなければ CUDA フォールバックを成功扱いしない。

OpenVINO は Intel GPU の実名を確認する。必要なら `openvino/libs` を `LD_LIBRARY_PATH` に含める。

```bash
openvino_libs=$(.venv_bench_intel/bin/python -c 'import openvino, pathlib; print(pathlib.Path(openvino.__file__).resolve().parent / "libs")')
export LD_LIBRARY_PATH="$openvino_libs:${LD_LIBRARY_PATH:-}"
.venv_bench_intel/bin/python -c 'import openvino as ov; c=ov.Core(); print([(d, c.get_property(d, "FULL_DEVICE_NAME")) for d in c.available_devices])'
```

Intel 名の GPU が列挙されないときは Intel 条件を実行しない。WebGPU のデバイス番号は OpenVINO/CUDA/MIGraphX と別体系。`.venv_bench_webgpu/bin/python probe_webgpu_adapters.py` で実名と番号を表示し、ランナーも製造元を照合する。`--device-name` は確認結果の記録であり、WebGPU の物理デバイスを文字列だけで自動確認はできない。

## 3. 測定

`./benchmark_matrix.sh --help` に全オプションがある。各 GPU の実名と確認したデバイス番号に置き換えて実行する。Intel の `GPU.N` と WebGPU の番号は独立している。下記の番号 0 は例であり、実機で確認してから使う。GPU ごとに別の出力先を使い、同時実行しない。

```bash
./benchmark_matrix.sh --vendor nvidia --device-name 'NVIDIA GeForce RTX 2070' --output-dir benchmarks/nvidia_linux_20260923 --gpu-index 0 --webgpu-device-index 0
./benchmark_matrix.sh --vendor intel --device-name 'Intel UHD Graphics' --output-dir benchmarks/intel_linux_20260923 --openvino-device GPU.0 --webgpu-device-index 0
./benchmark_matrix.sh --vendor amd --device-name 'AMD Radeon Graphics' --output-dir benchmarks/amd_linux_20260923 --gpu-index 0 --webgpu-device-index 0
```

EP が列挙されるだけでは成功ではない。JSON の `status=ok` と `executed_node_providers` で目的 EP のノード実行を確認する。GPU 依存ライブラリ不足、デバイス不一致、モデル非対応は条件別の失敗として残す。CPU のみへフォールバックした結果は GPU 成功として扱わない。

## 4. AI に渡すもの

各出力先の `summary.json` と `manifest.json`、`incomplete` の条件の JSON/コンソールエラー、OS・CPU・GPU・ドライバー・電源設定・EP と物理 GPU の対応を渡す。AI は同一ホスト内の CPU 比と同一 GPU の EP 間を解析する。Windows と Linux の差は環境差を明示し、異なる GPU の結果を同じデバイスとして比較しない。AMD の過去値は今回の一貫した比較には含めない。

## 5. 失敗条件の再測定

`manifest.json` が完成してから、元と同じ引数と出力先で `--retry-failed` を付ける。成功済み JSON は保持し、失敗条件だけ再実行する。TensorRT だけ再実行するときは `--retry-provider tensorrt` も付ける。TensorRT のライブラリ位置を明示する場合は `--tensorrt-lib-dir /path/to/TensorRT/lib` を追加する。
