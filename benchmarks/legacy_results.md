# 過去の実測結果（READMEから移動）

[総合比較へ](../BENCHMARKS.md)

旧READMEの測定条件・数値・当時の未検証事項を保存した記録。以下の「未検証」「承認待ち」は各測定時点の状態。2026-09-23の統一測定とは別集計。

DirectML旧測定は [`benchmark_model_vram.py`](../benchmark_model_vram.py) の `predict_images`（前処理込み）3回の平均で、独立したwarmup除外はない。今回の `session.run` の中央値とは直接比較しない。

### DirectMLのVRAM目安

RTX 2070 Max-Q 8GB、DirectML、batch-size=4、合成640 × 480 RGB画像、3回推論での統一実測値。`nvidia-smi`を100ms間隔で監視し、測定前との差をモデル実行による増分とした。

| Profile | 入力解像度 | 測定前 | 常駐／ピーク | ピーク増分 | 推論時間 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lightweight | 448 × 448 | 843 MiB | 1,614／1,614 MiB | 771 MiB | 120.3 ms/img |
| balanced | 384 × 384 | 852 MiB | 2,051／2,051 MiB | 1,199 MiB | 458.8 ms/img |
| high | 448 × 448 | 847 MiB | 3,616／3,618 MiB | 2,771 MiB | 1,273.9 ms/img |
| ultra | 512 × 512 | 840 MiB | 6,570／6,583 MiB | 5,743 MiB | 1,735.3 ms/img |
| wd14_v3 | 448 × 448 | 1,087 MiB | 2,970／2,973 MiB | 1,886 MiB | 442.3 ms/img |

| 測定待ちProfile | 入力解像度 | 状態 | 暫定VRAM目安 |
| --- | ---: | --- | ---: |
| compact_manual | 384 × 384 | 管理者承認待ち | 約1～1.5GB（推定） |
| medium_manual | 448 × 448 | 管理者承認待ち | 約1.5～2.5GB（推定） |
| future_1b | 512 × 512 | ONNX未公開・RTX 2070では実行不可見込み | 未測定 |

> [!WARNING]
> VRAM使用量はGPU、DirectML／ONNX Runtimeのバージョン、ドライバー、batch-size、同時使用中のアプリによって変動する。ultraのピーク増分は5,743MiBだったため、8GB未満のGPUでは推奨しない。メモリ不足時は`-BatchSize 1`または`-BatchSize 2`を指定する。ただしbatch-sizeを下げてもモデル重み自体の常駐分は減らない。

承認待ち2プロファイルの値は、Hugging Faceで確認したFP32 ONNX容量、パラメータ数、公式入力解像度、および実測済みモデルから見積もった概算である。承認後は`python benchmark_model_vram.py compact_manual`と`python benchmark_model_vram.py medium_manual`で同じ条件を測定できる。


### Linux Intel GPU実機確認（2026-09-21）

Ubuntu 26.04.1、kernel 7.0.0-31-generic、Core i7-1355U内蔵 Iris Xe（8086:a7a1）、Python 3.13.15、onnxruntime-openvino 1.24.1で、`/dev/dri/renderD128`とIntel OpenCL platformを確認した。`openvino_gpu_device`は`GPU.0`。`--force-intel --model-profile wd14_v3`で合成PNGを処理し、active providerは`OpenVINOExecutionProvider`と`CPUExecutionProvider`、OpenVINO debugログはモデル全体の対応と推論成功を報告した。入力はNHWC `[batch_size, 448, 448, 3]`、ラベルは10,861件。ExifToolでXMP Subject書き込みを確認した。通常ログではOpenVINO内部診断は表示されなかった。

batch-size=4、初回warmup 2.89秒、1枚の通常推論176.0 ms/img（再試行178.1 ms/img）。合成画像1枚の測定値であり、タグ品質や安定した性能の評価ではない。GPU.0指定とOpenVINO EPの実行は確認したが、カーネル単位のGPU使用率は測定していないためCPU fallbackの完全な排除は未確認。`lightweight`はHugging Faceのモデル取得が401（gated repository、未認証）で推論前に停止した。`balanced`、保存scoreでの整理、実画像、Server/Clientの実機疎通は未検証。

### Linux AMD Barcelo WebGPU実機確認（2026-09-22）

専用 VRAM 512 MiB 時の全実行可能モデルの VRAM・共有 GPU メモリ・RAM・速度は [512 MiB測定記録](amd_barcelo_512mb.md) にまとめた。4 GiB 時の同条件測定と速度比較は [4 GiB測定記録](amd_barcelo_4gb.md) にまとめた。

Ubuntu 26.04.1、kernel 7.0.0-31-generic、Ryzen 5 7530U内蔵 Radeon Graphics（PCI 1002:15e7、Mesa RADV）で、Vulkan 1.4と公式ONNX Runtime WebGPU EP 0.3.0を確認した。`./run_tagger.sh --gpu --model-profile wd14_v3 --force --no-report /path/to/image.png` はBarceloを検出して`venv_webgpu`を構築し、Vulkan経由のWebGPUを自動選択する。他のGPUでも `--webgpu` で明示的に試せる。GPU EPが使えない場合はエラーで停止する。

16 × 16の単色合成PNGをWD14 V3で処理し、タグとXMPを書き込んだ。active providerは`WebGpuExecutionProvider`と`CPUExecutionProvider`。ONNX RuntimeのプロファイルではWebGPUで1,693ノード、CPUで885ノードが実行された。後者には形状処理などが含まれ、モデル全体がGPU専用になるわけではない。同じ入力に対するCPU出力との差は最大2.98e-6。単発の通常推論は945.7 ms/img。合成画像1枚の値であり、実画像での品質・性能比較やDBV4プロファイルの互換性は未検証。WebGPU EPは[ONNX Runtime公式手順](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html)に従って導入する。

