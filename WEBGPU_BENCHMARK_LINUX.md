# Linux セッションへの指示書：NVIDIA GPU で WebGPU と CUDA を比較

Windows セッションが終わり、そのコミットが push された後、このファイルを Linux 側の新しい AI セッションの最初の指示として渡す。Windows の成果物と同じ branch の最新コミットから開始する。

## 目的

同じ NVIDIA GPU、同じ ONNX モデル、同じ入力とバッチ条件で、Linux の WebGPU EP（Vulkan）と CUDA EP の推論速度を比較する。結果は `WebGPU / CUDA` の所要時間比と速度低下率で示す。両 OS とも CUDA を比較基準にするが、OS、ドライバー、Dawn backend、ONNX Runtime の差が残るので、Windows と Linux の絶対速度を直接比較して WebGPU 固有の低下率としない。

## 開始時に確認すること

1. `git status --short`、branch、HEAD、remote を確認する。Windows 側の push 済みコミットと一致させ、既存変更を保持する。`AGENTS.md` があれば読む。
   AMD Barcelo 512 MiB の [先行測定](benchmarks/amd_barcelo_512mb.md) は別 GPU の参考値として読み、NVIDIA の比率計算には混ぜない。
2. Linux distribution、kernel、CPU、NVIDIA GPU、ドライバー、CUDA、Vulkan GPU、Python、ONNX Runtime、CUDA EP、WebGPU EP のバージョンを記録する。`nvidia-smi` と `vulkaninfo --summary` で対象 NVIDIA GPU を確認し、llvmpipe 等の software device を測定対象にしない。
3. Windows セッションが追加した共通ベンチマークコードと `benchmarks/` の測定条件を読む。Windows の生データを変更しない。

## 測定

- 共通コードの同じ seed、合成 RGB 640×480 入力、前処理、モデル、batch size、warmup 回数、計測回数、セット数を使う。Windows で成功した `wd14_v3` と `balanced`、batch size 1 と 4 を優先する。アクセス制限や VRAM 不足で実行できない組み合わせは理由を残す。
- WebGPU は `venv_webgpu` と plugin EP を使い、Vulkan backend が NVIDIA GPU を選んだことを確認する。CUDA は `venv_gpu` の CUDA EP を明示し、TensorRT が有効になっていないことを確認する。CPU のみへ fallback した測定を成功として扱わない。
- 推論の壁時計時間だけを計時し、セッション作成、初回コンパイル、画像読み込み、前処理、XMP、レポート生成は別計測または対象外にする。セットごとの中央値と p95、全セットの中央値、ms/image、images/s、`低下率 = (WebGPU_ms / CUDA_ms - 1) × 100%` を出す。
- `nvidia-smi` で測定前・セッション常駐・推論中ピークの VRAM を記録する。WebGPU と CUDA の順番を交互に変え、電源モード、温度、他アプリの負荷をできるだけ一定にする。
- ONNX Runtime profiling で各 provider に割り当てられたノード数を記録する。CPU fallback の有無、同一入力での WebGPU と CUDA の最大絶対出力差も示す。

## 検証と報告

1. `./run_tagger.sh --webgpu --model-profile wd14_v3 --force --no-report <コピーした検証画像>` と CUDA 経路の実推論を確認する。原本画像は変更しない。WebGPU 用 CLI と共通ベンチマークの結果が矛盾した場合は原因を調べる。
2. Linux の環境、測定条件、生データ、集計、失敗と制約を `benchmarks/` 下の別ファイルへ保存する。Windows の表と並べて示してよいが、OS をまたいだ絶対速度差を WebGPU 固有の低下率として扱わない。
3. 必要な修正だけ行い、`bash -n run_tagger.sh`、全 unit test、Python の `py_compile`、`git diff --check`、変更した実行経路の再試験を行う。
4. README と changelog に Linux 結果の参照先を追記し、コミットして通常 push する。force push はしない。最終報告に環境、各 EP の実測 provider、profile 別の比率、出力差、CPU fallback、テスト結果、未検証項目、コミット SHA を含める。
