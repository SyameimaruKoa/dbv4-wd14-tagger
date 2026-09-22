# Windows 引継ぎ：NVIDIA の CUDA と WebGPU を測る

Windows と Linux は別セッションにする。**Windows を先に実行する。** AI は測定を逐次代行せず、このリポジトリの [benchmark_nvidia_ep.py](benchmark_nvidia_ep.py) をユーザーへ渡す。ユーザーが実行し、[summarize_nvidia_ep.py](summarize_nvidia_ep.py) の小さな集計結果を渡したら AI が解析する。必要な生データだけ追加で確認する。トークンをチャットに貼らない。

## 1. 準備

`git status --short` と HEAD を確認し、既存変更を保護する。Windows のバージョン、NVIDIA GPU、ドライバー、CUDA、Python を記録する。`nvidia-smi` が対象 GPU を表示することを確認する。現行 `run_tagger.ps1 -Gpu` は DirectML 用なので、CUDA 比較には使わない。2つの独立した仮想環境を作る。Python 3.13 が無ければ、ONNX Runtime の wheel がある Python 3.12 を使う。

```powershell
py -3.13 -m venv .venv_bench_cuda
py -3.13 -m venv .venv_bench_webgpu
.\.venv_bench_cuda\Scripts\python.exe -m pip install -r requirements.txt onnxruntime-gpu psutil
.\.venv_bench_webgpu\Scripts\python.exe -m pip install -r requirements.txt onnxruntime onnxruntime-ep-webgpu psutil
```

必要なら Hugging Face にローカルでログインする。CUDA EP が初期化できない場合は CUDA/cuDNN 環境を修正し、CPU fallback の結果を CUDA 測定として扱わない。

## 2. ユーザーが実行するコマンド

同一 NVIDIA GPU で `wd14_v3` と `balanced`、batch size 1 と 4 を測る。既定で各条件 warmup 3 回、20 回×3 セット。モデル取得、セッション作成、画像前処理は推論時間から除かれる。出力 JSON には各回の時間、VRAM・プロセス RAM、active provider、環境バージョンが残る。

```powershell
$profiles = @("wd14_v3", "balanced")
foreach ($profile in $profiles) {
    foreach ($batch in @(1, 4)) {
        & .\.venv_bench_cuda\Scripts\python.exe benchmark_nvidia_ep.py --provider cuda --profile $profile --batch-size $batch --output "benchmarks/nvidia_windows/${profile}-b${batch}-cuda.json"
        & .\.venv_bench_webgpu\Scripts\python.exe benchmark_nvidia_ep.py --provider webgpu --profile $profile --batch-size $batch --output "benchmarks/nvidia_windows/${profile}-b${batch}-webgpu.json"
    }
}
py -3.13 summarize_nvidia_ep.py benchmarks/nvidia_windows --output benchmarks/nvidia_windows_summary.json
```

WebGPU 実行時に NVIDIA の VRAM 増分が観測されない場合は、別 GPU を選んだ可能性がある。`--webgpu-device-index 1` などを試し、同じ条件の結果を上書きする。失敗した条件は成功値で埋めず、失敗 JSON を残す。

## 3. AI に渡すもの

`benchmarks/nvidia_windows_summary.json` と、集計の `incomplete` に挙がった条件の JSON またはコンソールエラーを渡す。AI は WebGPU/CUDA の時間比、速度低下率、VRAM 差、GPU 選択、CPU fallback の可能性を解析する。必要な場合だけ生データを依頼する。結果を Windows 用ファイルへ記録して commit・通常 push し、その branch・SHA・実行条件を Linux セッションへ渡す。512 MiB／4 GiB の [AMD 測定](benchmarks/amd_barcelo_4gb.md)は別 GPU の参考値で、NVIDIA の比率には混ぜない。
