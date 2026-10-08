# ベンチマーク結果と分析

今後の複数OS・GPU・実機での再測定は、[AI向け統一測定・記録指示書](BENCHMARK_PROTOCOL.md)に従う。プログラムに帰属するメモリ、速度・精度、モデル変換時間の定義と記録テンプレートを共通化する。過去の結果は新仕様準拠の値へ読み替えない。

## 測定一覧の入口（2026-10-04更新）

[全測定178件の速度・RAM・VRAM一覧](benchmarks/all_measurements.md)に、保存済みのultra・balanced・wd14_v3・lightweight・highを集約した。実機・測定日ごとに速度とメモリの表を分け、各値の出典へリンクする。同じ測定組・実行方式・batchでのモデル間の時間比／RSS比／VRAM比も掲載。未測定・失敗・診断値は成功値に読み替えない。容量に対する使用率は分母が記録されている場合だけ算出する。

最終コード確認と修正・テスト結果は[最終確認記録](benchmarks/final_review_20261004/README.md)に保存した。

### ultraの実機結果早見表

時間はms/枚。`b1 / b4`を併記し、メモリはその2条件の最大値。VRAMの「時点値」は初期化・warmup後などのスナップショット最大で、連続監視ピークではない。機種・OS・測定回数・分割の有無が異なるため、保存結果の一覧として読む。

| 実機・OS | 実行方式 | b1 / b4 ms/枚 | RSS最大 MiB | GPUメモリ MiB・計測方式 | 出典 |
|---|---|---:|---:|---|---|
| AERO RTX 2070 Max-Q / Ubuntu | TensorRT | 295.60 / 296.39 | 7,237.22 | 4,238.00 時点値 | [b1](benchmarks/nvidia_linux_20261004/tensorrt-b1.json) / [b4](benchmarks/nvidia_linux_20261004/tensorrt-b4.json) |
| 同上 | CUDA | 402.75 / 397.56 | 1,103.75 | 5,244.00 時点値 | [b1](benchmarks/nvidia_linux_20261004/cuda-b1.json) / [b4](benchmarks/nvidia_linux_20261004/cuda-b4.json) |
| 同上 | ncnn FP32 | 1,015.81 / 1,019.59 | 372.70 | 3,025.00 時点値 | [b1](benchmarks/nvidia_linux_20261004/ncnn-b1.json) / [b4](benchmarks/nvidia_linux_20261004/ncnn-b4.json) |
| AERO RTX 2070 Max-Q / Windows | TensorRT | 313.30 / 307.09 | 未測定 | 4,117.00 時点値 | [b1](benchmarks/aero_regression_20261004/tensorrt-b1.json) / [b4](benchmarks/aero_regression_20261004/tensorrt-b4.json) |
| 同上 | CUDA | 403.69 / 397.79 | 未測定 | 5,323.00 時点値 | [b1](benchmarks/aero_regression_20261004/cuda-b1.json) / [b4](benchmarks/aero_regression_20261004/cuda-b4.json) |
| 同上 | ncnn FP32 | 1,012.66 / 1,014.69 | 未測定 | 未測定 | [b1](benchmarks/aero_regression_20261004/rtx-ncnn-b1.json) / [b4](benchmarks/aero_regression_20261004/rtx-ncnn-b4.json) |
| AERO UHD 630 / Windows | OpenVINO FP32 | 7,967.46 / 8,637.68 | 未測定 | 未測定 | [b1](benchmarks/aero_regression_20261004/intel-openvino-b1.json) / [b4](benchmarks/aero_regression_20261004/intel-openvino-b4.json) |
| a13m Iris Xe / Windows | OpenVINO FP32 | 1,599.69 / 1,542.60 | 未測定 | 未測定 | [b1](benchmarks/openvino_intel_a13m_20261004/fp32-b1.json) / [b4](benchmarks/openvino_intel_a13m_20261004/fp32-b4.json) |
| 同上 | ncnn FP32 | 32,900.08 / 33,299.30 | 未測定 | 未測定 | [b1](benchmarks/ncnn_intel_a13m_20261003/fp32-b1.json) / [b4](benchmarks/ncnn_intel_a13m_20261003/fp32-b4.json) |
| AMD Barcelo / Ubuntu | ncnn FP32同期あり | 6,505.56 / 6,659.12 | 219.16 | 500.02 VRAMピーク＋2,841.51 GTTピーク | [b1](benchmarks/ncnn_amd_20261003/sync-default-fp32-b1.json) / [b4](benchmarks/ncnn_amd_20261003/sync-fp32-b4.json) |
| 同じAMD PC / Bazzite | ncnn FP32同期あり | 7,509.21 / 7,558.89 | 198.97 | 493.02 VRAMピーク＋2,992.02 GTTピーク | [b1](benchmarks/ncnn_amd_bazzite_20261003/fp32-b1.json) / [b4](benchmarks/ncnn_amd_bazzite_20261003/fp32-b4.json) |
| Nintendo Switch / Linux | ncnn FP32・分割128 | 75,004.67 / 未測定 | 544.77 | 未測定 | [b1](benchmarks/ncnn_switch_20261003/ultra-stream128-b1.json) |
| PS4 Liverpool / Linux | ncnn FP32・分割128 | 87,825.23 / 80,302.54 | 274.14 | 690.33 VRAMピーク（33.71%）＋36.61 GTTピーク | [b1](benchmarks/ncnn_ps4_20261004/ultra-stream128-b1.json) / [b4](benchmarks/ncnn_ps4_20261004/ultra-stream128-b4.json) |

Switchの外部監視ではRSS最大660.27MiB。プロセス内監視と範囲が異なるため、早見表とは別に[全測定一覧の補助観測](benchmarks/all_measurements.md#補助観測と旧記録)へ掲載した。モデル間比較の例として、旧DirectML・batch4のultraはbalancedの時間3.78倍、VRAMピーク3.21倍、highは時間2.78倍、VRAMピーク1.76倍。同じ旧測定内での比較であり、最新のncnn/TensorRT値との直接比較には使わない。


## Issue #21: ncnn Vulkan 検証

2026-10-02、Intel Core i5-6500T / HD Graphics 530、RAM約15 GiBのLinux環境で検証した。`/dev/dri`が公開されず、Vulkanの実GPUを利用できない。pnnx 20260526で元PyTorchの `convnextv2_huge.dbv4-full` をbatch 1・入力512×512・FP32で変換し、`model.ncnn.param`（約71 KiB）と `.bin`（約2.58 GiB）を生成した。ncnn 1.0.20260526で両ファイルの読み込みとCPU推論が成功し、入出力名 `in0` / `out0` を確認した。元のGlobalResponseNormの `torch.addcmul` は同値の基本演算に変えてからtraceし、pnnx未変換演算が残らないことを確認した。Mesa llvmpipe上の合成Sigmoidグラフでは入力CHW `(3,4,4)` → 出力48値の推論が成功した。llvmpipeはVulkanのCPUデバイスなので自動GPU選択から除外する。

同じ合成640×480 RGB画像、同じDBV4前処理、metadata version `07cfcadc104847e8`、12,476ラベルを用いたCPU照合結果:

| 比較 | 最大確率絶対差 | 平均確率絶対差 | Rating 4値の最大絶対差 | best_threshold採用タグの差 |
| --- | ---: | ---: | ---: | --- |
| ncnn FP32 CPU / ONNX Runtime CPU | 0.000228107 | 0.000000253157 | 0.00000144460 | 0件 |
| 元PyTorch / ONNX Runtime CPU | 0.00000184774 | 0.00000000147093 | 0.0000000135042 | 0件 |

一次資料は [ONNX CPU](benchmarks/ncnn_local_20261002/cpu-b1.json)、[ncnn CPU変換検証](benchmarks/ncnn_local_20261002/ncnn-cpu-conversion-check.json)、[元PyTorch照合](benchmarks/ncnn_local_20261002/torch-source-check.json)。全確率配列とratingを保存している。FP32の検証は `atol=1e-4, rtol=1e-3` で全ラベル成功し、rating最大差は1e-4未満、採用タグは一致した。元モデル全体でGRNを書き換える前後の最大確率差は0.000000730157だった。ncnnの入力にはNumPyバッファの寿命を保持する必要があるため、実装は連続したfloat32配列をextract完了まで保持する。

ONNX CPU（onnxruntime-openvino 1.24.1のCPUExecutionProvider）のbatch 1はwarm-up 3回・20回の中央値 **8,063.17 ms/枚**、モデル取得・初期化を含むロード **128.50秒**、ピークRSS **3,578.36 MiB**。ncnn CPU照合は1回のコールド推論・4スレッドなので、速度比較に混ぜない。

AMD Barcelo実機（RAM約14 GiB、swap約14 GiBのzram）では元重み取得とTorchScript作成に成功したが、pnnxはカーネルのOOM killerで終了した（2026-10-02 18:19:39、pnnx pid 7741、ラッパー終了コード247）。trace用PyTorchプロセスを終了してからpnnxを実行し、保存するパラメーターの勾配も無効にするよう変換器を改善した。小さいモデルでこの変換経路と実ncnn推論の一致を確認し、勾配無効化の前後で生成したparam/binのSHA256も一致した。

以後のAMD変換はsystemd user serviceで `MemoryHigh=9G, MemoryMax=9G, MemorySwapMax=2G, OOMPolicy=stop` を設定し、変換失敗後もSSH応答が維持されることを確認した。いずれも2026-10-02、pnnx 20260526、FP32、ultraでの試行。

| 経路 | サービス実行時間 | メモリピーク / swap | 結果 |
| --- | --- | --- | --- |
| trace worker分離後のTorchScript | 6分48秒 | 9 GiB / 2 GiB | oom-kill |
| 上記＋パラメーターの勾配無効化 | 1分40秒 | 9 GiB / 2 GiB | oom-kill |
| 同じモデルのONNX＋外部重み | 21.866秒 | 9 GiB / 2 GiB | shape_inference中にoom-kill |

ONNX経路は既存の実機内ファイルをハードリンクして使用した。外部重みのシンボリックリンクはONNX Runtimeのパス検証で拒否された。ONNX経路ではpnnxが入力形状を再取得しており、形状と変換結果の妥当性は未確認。各経路とも有効なultraキャッシュは公開していない。上限付き試行の失敗は、より多いRAMでの変換不可能を示すものではない。

この時点ではultraのAMD変換と実GPU推論は未完了だった。後日の実GPU測定・精度差は次節に記録する。NVIDIA/Intel/PS4/Switchの実GPU比較も未測定。Windows用ランチャーはPowerShell 7.5.2の構文解析に成功したが、Windows実機動作は未確認。

AMD実機のncnnはGPU 0 `AMD Radeon Graphics (RADV RENOIR)` とGPU 1 `llvmpipe`を列挙した。通常ランチャーで `--provider ncnn --gpu-index 42 --probe-provider -m ultra` は「GPU 42を利用できません（検出数2）」で停止し、CPUへ変更されなかった。`--gpu-index 1` も「ソフトウェアデバイス」として停止した。これらはデバイス選択と失敗時の検証であり、DBV4の実GPU推論成功を示すものではない。

### 2026-10-03 AMD Vulkan実測と精度差

許可を受けてローカル変換済みモデル3ファイルをAMD実機へ転送し、全ファイルのSHA256一致を確認した。ncnn Vulkan GPU 0 `AMD Radeon Graphics (RADV RENOIR)` でultra FP32の実推論に成功した。以下は同期修正前の不合格の測定であり、修正後の合格結果は「RADV shader同期」の節に記録する。

| 条件 | warm-up / 反復 | 中央値 ms/枚 | ONNX CPUとの最大確率差 | 採用タグ差 |
| --- | --- | --- | --- | --- |
| ncnn Vulkan FP32 b1 | 3 / 20 | 6,614.24 | 0.122703 | 4件 |
| Winograd無効化 FP32 b1（診断） | 1 / 3 | 6,557.02 | 0.159602 | 4件 |
| packing無効化 FP32 b1（診断） | 1 / 3 | 6,617.08 | 0.0645395 | 0件 |
| WebGPU b1 | 3 / 20 | 4,964.40 | 0.00000292063 | 0件 |

最初の[FP32測定](benchmarks/ncnn_amd_20261003/fp32-b1.json)ではロード・起動検証11.43秒、RSSピーク225.63 MiB、VRAMピーク503.32 MiB、GTTピーク2,982.07 MiB、測定中GPU使用率平均98.0%・最大99%。GPUメモリ・負荷はデバイス全体の値であり、RSSにはGPU共有メモリの全量は含まれない。rating最大差は0.000444293で、精度条件を満たしていない。

同じAMD実機の[ncnn CPU照合](benchmarks/ncnn_amd_20261003/ncnn-cpu-check.json)は全12,476ラベルが `atol=1e-4, rtol=1e-3` 内、最大確率差0.000157028、rating最大差0.00000387430、採用タグ差0件で成功した。[初段の中間層比較](benchmarks/ncnn_amd_20261003/first-block-blobs.json)では畳み込みは一致し、LayerNorm後に差が出た。[packing無効化](benchmarks/ncnn_amd_20261003/no-packing-blobs.json)では調べた初段中間層が一致したが、[モデル全体](benchmarks/ncnn_amd_20261003/no-packing.json)の確率差は解消していない。subgroup無効化はshaderコンパイル失敗と終了コード139で停止したため採用しない。

[修正前の実GPU統合テスト](benchmarks/ncnn_amd_20261003/integration.log)は確率照合とXMPスコア照合の2件とも失敗した。当初のFP16・batch 4・統合テストの確認は未完了だった。WebGPU b1の同条件測定とb4の失敗は以下に記録する。通常テストは2026-10-03に107件実行して成功（実GPU用2件skip）。localhostソケット禁止のsandboxではServer/Clientテストが失敗したため、ループバック通信を許可して実行した。修正前の失敗を成功済みの精度・XMP検証として扱わない。同期設定後の統合テスト成功は後述する。

追加診断の[第3ステージ4ブロック目](benchmarks/ncnn_amd_20261003/block4-blobs.json)ではLayerNorm出力（blob 278）が一致し、その次のGemm（blob 280）で大きな差が出た。そこで同じLinearパラメーターを保持した別クラスをtraceし、空間Linearを同値の1×1 Convolutionへ変換する診断オプションを追加した。単にnn.Linear.forwardを変更するとpnnxがLinearとして扱い、Gemmが残ったため、その方式は採用しない。修正版ではGemm残存を検査し、76個のConvolutionを含むultraのncnnモデルを生成した。重みはFP32のまま、モデルとラベル数は変更していない。

修正版の[CPU照合](benchmarks/ncnn_local_20261002/conv1x1-cpu-check.json)は全12,476ラベルが許容差内、最大確率差0.00000546873、rating最大差0.0000000160653、採用タグ差0件で成功した（packing無効）。pnnxの中間ONNXはprotobufの2 GiB制限で書き出せなかったが、必要なncnn param/binは生成・読み込み・推論に成功した。修正版の[Vulkan FP32測定](benchmarks/ncnn_amd_20261003/conv1x1-fp32-b1.json)も最大確率差0.08614275、採用タグ差1件で不合格だった（warm-up 3回・20回、中央値4,882.48 ms/枚）。既定変換には採用していない。

[WebGPU b1](benchmarks/ncnn_amd_20261003/webgpu-b1.json)は同じ入力SHA256・metadataで20回測定した。ロード11.74秒、rating最大差0.000000119209、VRAMピーク501.94 MiB、GTTピーク3,507.48 MiB、GPU使用率平均98.45%。ログでAMD GPUの選択と起動検証のWebGpuExecutionProvider実行を確認した。[WebGPU b4](benchmarks/ncnn_amd_20261003/webgpu-b4.log)はwarm-up中に `Device is lost` で停止し、速度JSONは生成していない。batch 4の成功値として扱わない。

### RADV shader同期による精度一致

CPU中間テンソルをCPU/Vulkanの両方へ入力した[再生診断](benchmarks/ncnn_amd_20261003/replay278.json)ではReshapeが完全一致し、次のGemmも最大差0.00000143051で許容差内だった。一方、モデル全体でLayerNorm・Reduction・Reshapeの各演算種だけをCPUにしても精度差は解消しなかった。

AMD RADVの `RADV_DEBUG=syncshaders` で[同じultraの標準FP32](benchmarks/ncnn_amd_20261003/sync-default-fp32-b1.json)が全12,476ラベルで `atol=1e-4, rtol=1e-3` に一致した。最大確率差0.00000274181、rating最大差0.0000000135042、採用タグ差0件。warm-up 3回・20回の中央値 **6,505.56 ms/枚**、ロード10.30秒、RSSピーク219.16 MiB、VRAM500.02 MiB、GTT2,841.51 MiB、GPU使用率平均98.17%。通常のpacking/Winograd/subgroup設定を保持し、診断用CPU演算マスクは使用していない。

同期設定をAMD LinuxでVulkan初期化前に適用するよう実装した。[実GPU統合テスト](benchmarks/ncnn_amd_20261003/sync-integration.log)は確率・rating・閾値の照合と、単独実行／Server-Clientの原画像転送／前処理済み転送のXMP一致の2件とも成功した。変更後の通常テストは109件成功（実GPU用2件skip）。

[標準FP32 batch 4](benchmarks/ncnn_amd_20261003/sync-fp32-b4.json)もwarm-up 3回・20回で成功し、中央値 **6,659.12 ms/枚**、最大確率差0.00000274181、採用タグ差0件だった。Vulkan初期化前の同期設定はアプリが自動適用した。4枚はbatch 1グラフを順に実行する。VRAMピーク500.02 MiB、GTT2,841.51 MiB、GPU使用率平均98.75%。

同期設定後のFP16 storage・packed・arithmeticはbatch 1／4の6条件すべて起動検証で非有限出力を検出して停止した。[各条件の終了コード](benchmarks/ncnn_amd_20261003/sync-fp16-matrix.log)を保存し、速度JSONは生成していない。matrix用サービス自体の成功は6条件の成功を意味しない。今回のAMD ultraではFP32を推奨する。

実機のMesaは26.0.8。最初に試した `fullsync` は実機のdriver binaryに含まれず、その試行は同期の効果を示さない。`syncshaders` は実機で対応を確認した。[Mesa公式資料](https://docs.mesa3d.org/envvars.html#radv-driver-environment-variables)に記載されたdispatch間の同期を利用している。これは演算間同期に関係する問題を示すが、ncnnとdriverのどの処理が原因かまでは特定していない。WebGPU b1（4,964.40 ms/枚）の方が今回のncnn標準FP32より速く、性能優位は確認できていない。

### ネイティブncnnと中間テンソルの診断

upstream ncnn `9f9d4ec8150d840b570e735323499b12a354483a` をAMD実機上でVulkan有効・Python binding付きでビルドした。別のsource checkoutをPYTHONPATHで指定し、既存wheelを維持した。[未修正nativeのFP32](benchmarks/ncnn_amd_20261003/candidate-fp32.json)は最大確率差0.05907771で不合格だった。

subgroup無効化時にshaderの機能マクロが有効のまま残る不整合を[依存ライブラリ用パッチ](patches/README.md)で修正した。[パッチ後の測定](benchmarks/ncnn_amd_20261003/patched-no-subgroup.json)はshaderエラーなく終了したが、最大確率差0.19295597、採用タグ差3件で不合格。これはコンパイルエラーの修正であり、確率一致の修正には至っていない。JSONにはnative bindingのSHA256も記録する。

[FP16 storageの起動検証](benchmarks/ncnn_amd_20261003/conv1x1-fp16-storage-probe.log)は非有限の確率出力を検出して停止した。速度測定結果は生成していない。

### Windows実機確認（2026-10-03）

WindowsのPython 3.13.15、ncnn 1.0.20260526で、同じultraモデル3ファイルのSHA256一致を確認した。Vulkan GPU 2はRTX 2070 Max-Q、GPU 0はIntel UHD 630。PowerShell 5.1でランチャーの構文解析エラーは0件だった。

RTXの[FP32 batch 1](benchmarks/ncnn_windows_20261003/rtx-fp32-b1.json)はwarm-up 3回・20回の中央値1,131.62 ms/枚、最大確率差0.00000357628、rating最大差0.0000000200234、採用タグ差0件。[実GPU統合テスト](benchmarks/ncnn_windows_20261003/rtx-integration.log)2件は16.104秒で成功し、単独実行とServer/Clientの原画像・前処理済み転送でXMP一致を確認した。既存の[CUDA経路](benchmarks/ncnn_windows_20261003/cuda-probe.log)も起動検証でCUDAExecutionProviderの実行を確認した。Windows終了時はNetを解放してからVulkan instanceを破棄し、正常終了コード0を確認した。

[別のメモリ確認](benchmarks/ncnn_windows_20261003/rtx-memory-check.json)はVulkan index 2に対してNVIDIA SMI index 0を指定した。GPU全体の使用量は開始前716 MiB、ロード後とwarm-up後は3,732 MiB。時点ごとの値でありピーク測定ではない。WindowsのRSS・Intel共有メモリ・GPU使用率はこの測定で取得していない。

Intelは[実GPUでの全確率・rating・採用タグ比較](benchmarks/ncnn_windows_20261003/intel-integration.log)に成功した。ただし推論1回に約140秒かかり、Server/Clientは既定の15秒でタイムアウトした。[20回の速度測定](benchmarks/ncnn_windows_20261003/intel-fp32-b1.log)は途中で停止したため中央値を記録しない。`client_timeout`と`client_batch_timeout`を300秒にした[再確認](benchmarks/ncnn_windows_20261003/intel-integration-300s.log)は2件とも成功（771.317秒、終了コード0）。原画像・前処理済み転送と単独実行のXMP一致を確認した。アプリの既定タイムアウトは変更していない。

RTXの同じultra・入力SHA256・metadata・warm-up 3回／20回の[比較](benchmarks/ncnn_windows_20261003/comparison-summary.json)は以下のとおり。全条件の12,476確率をCPU参照に対して `atol=1e-4, rtol=1e-3` で照合し成功した。ncnnとCUDA/TensorRTの採用タグ差は0件、rating最大差は0.000000119209。最新修正を含む[RTX統合テスト](benchmarks/ncnn_windows_20261003/rtx-integration-latest.log)も2件成功（18.680秒）。

| RTX経路 | batch 1 ms/枚 | batch 4 ms/枚 | ncnn b1との最大確率差（b1 / b4） |
| --- | --- | --- | --- |
| ncnn FP32 | 1,131.62 | 1,128.64 | 0 / 0 |
| CUDA | 490.67 | 472.30 | 0.000000894070 / 0.00000120699 |
| TensorRT | 377.54 | 359.47 | 0.00000309944 / 0.00000309944 |

TensorRT/CUDAは起動検証の実行EPも確認した。Intelのncnn統合試験は別GPUで並行実行されており、この表は完全な無負荷時の測定ではない。ncnn b4は固定batch 1グラフの逐次実行である。PowerShell 5.1で実際の[候補生成関数](benchmarks/ncnn_windows_20261003/launcher-priorities.json)をGPU列挙の置き換えで確認し、NVIDIAではTensorRT/CUDA、IntelではOpenVINO、Windows AMDでは既存DirectMLをncnnより優先することを確認した。この検証はGPU列挙を模擬した候補順の確認であり、全候補の実機成功を示さない。

Intelの[OpenVINO比較](benchmarks/ncnn_windows_20261003/openvino-b1.log)はORT 1.24.1・OpenVINO 2025.4.1、実名確認済みのGPU.0 `Intel(R) UHD Graphics 630 (iGPU)`で実行した。GPU EPがactiveでも起動検証で非有限出力を検出し終了コード1で停止したため、確率差・速度JSONは生成していない。batch 4は同じ起動検証の失敗後には実行していない。GPU.1のNVIDIAに取り違えた結果ではない。OpenVINOの成功値として扱わず、同じIntel GPUで成功したncnnを代替候補として維持する。

### AMD Bazzite別OSでの再確認（2026-10-03）

同じBarcelo搭載PCのBazzite 44.20260929.0／kernel 7.2.7-ogc1.1.fc44／Mesa 26.2.2へrootで接続し、`/opt/dbv4-wd14-tagger`（実パス `/var/opt/dbv4-wd14-tagger`）をGit経由でPRブランチへ更新した。既存の差分156件は実行権限のみで、rootでGitの記録に合わせた。NVIDIA実機への接続・再測定は行っていない。

OS標準Python 3.14.7を変更せず、uvでリポジトリ内にPython 3.13.16と`venv_ncnn`を用意した。ncnnは1.0.20260526。ExifTool 13.59は公式Gitから`.dbv4/runtime/exiftool`へ配置し、PATHに追加して試験した。ultraのモデル3ファイルを転送し、既存のbin／param／manifestとのSHA256一致を確認した。[環境とハッシュ](benchmarks/ncnn_amd_bazzite_20261003/environment.log)、[依存版](benchmarks/ncnn_amd_bazzite_20261003/packages.log)を保存している。

[通常テスト](benchmarks/ncnn_amd_bazzite_20261003/unit-tests-complete.log)は111件成功（実GPU2件skip）。初回はテストが直接importするhttpxの不足で失敗し、venvへ追加後に通過した。[実GPU統合テスト](benchmarks/ncnn_amd_bazzite_20261003/integration.log)は2件成功（40.697秒、終了コード0）。GPU 0 `AMD Radeon Graphics (RADV RENOIR)`で、全12,476確率・rating・閾値、単独実行／原画像転送／前処理済み転送のXMP一致を確認した。ncnnだけの環境でもHFログインできるよう、Bash／PowerShellの既存venv探索に`venv_ncnn`を追加した。

[FP32 batch 1](benchmarks/ncnn_amd_bazzite_20261003/fp32-b1.json)はwarm-up 3回・20回の中央値 **7,509.21 ms/枚**、ロード9.74秒、最大確率差0.00000262260、rating最大差0.0000000108266、採用タグ差0件。アプリが `RADV_DEBUG=syncshaders` を適用した。GPU使用率平均97.86%・最大99%、RSSピーク198.77 MiB、VRAM493.02 MiB、GTT2,992.02 MiB。GPUカウンターはデバイス全体の値である。OS・driver・稼働条件が異なるため、旧Ubuntuとの時間差をOS単独の効果とは扱わない。

[FP32 batch 4](benchmarks/ncnn_amd_bazzite_20261003/fp32-b4.json)もwarm-up 3回・20回で成功し、中央値 **7,558.89 ms/枚**、ロード10.26秒、最大確率差0.00000262260、採用タグ差0件。RSSピーク198.97 MiB、VRAM493.02 MiB、GTT2,992.02 MiB、GPU使用率平均98.63%・最大99%。4枚はbatch 1グラフで逐次実行した。[全条件の照合](benchmarks/ncnn_amd_bazzite_20261003/comparison-summary.json)で、FP32 b1/b4の全12,476確率がCPU参照の許容差 `atol=1e-4, rtol=1e-3` 内であることを確認した。

FP16 storage／packed／arithmeticのb1/b4は6条件すべて非有限出力により起動検証で停止した。[個別終了コード](benchmarks/ncnn_amd_bazzite_20261003/matrix.log)はすべて1で、速度JSONは生成していない。新MesaでもFP16の成功・メモリ削減効果は確認できなかったため、この環境のultraはFP32を推奨する。matrixの外側の正常終了は全条件の成功を意味しない。

[通常Bashランチャー](benchmarks/ncnn_amd_bazzite_20261003/launcher-probe.log)の `--provider ncnn --gpu -m ultra --gpu-index 0 --probe-provider` も実GPUの起動検証に成功（終了コード0）。uvで作ったvenvへpipを追加して既存ランチャーで再利用した。[ログイン候補の確認](benchmarks/ncnn_amd_bazzite_20261003/login-environment-check.log)は、隔離したHF実行ファイルだけをstubにして実際のBashランチャーを実行し、唯一の`venv_ncnn`を選ぶことを確認した。これはHFへの実ログイン結果とは区別する。

### Nintendo Switch Linux・4GB共有メモリ（2026-10-03）

`ssh ns` の Ubuntu 24.04.4／aarch64／kernel 4.9.140-l4t、NVIDIA Tegra X1 (nvgpu)、RAM約3.9GiB、zram swap 3GiBで検証した。ncnnは1.0.20260526、Pythonは3.12.3。NVIDIA ICD `/etc/vulkan/icd.d/nvidia_icd.json` を `VK_DRIVER_FILES` と `VK_ICD_FILENAMES` に指定し、GPU 0の実名を確認した。ソフトウェアVulkanではない。[小さなSigmoidグラフ](benchmarks/ncnn_switch_20261003/small-vulkan-only.json)は全48値でNumPyの解析解と一致（最大差0.0000000596046、終了コード0）。最初のCPU側のプローブはsegfaultしたが、Vulkan側はncnn所有の入力を使って正常終了した。両者は条件が異なり、CPUプローブの原因は確定していない。

標準のultra FP32一括ロードは[メモリ監視](benchmarks/ncnn_switch_20261003/ultra-memory-watch.json)が約49.66秒で自プロセスを停止した。空きメモリ256MiBを残す方針に対し最低233.57MiB、RSSピーク2,226.46MiB、終了コード-15。起動検証・速度測定には到達しておらず、OOM killerの終了として扱わない。

同じ元モデルを `--ncnn-part-size-mib 128` で14区間に分け、残差分岐をまたがず1区間ずつVulkanで実行した。最大区間の重みは242.7MiB、重み・演算層・FP32精度・前処理・12,476ラベルを保持する。分割用の重み境界はncnn自身の層ローダーで求め、全binバイト数の一致を検査してからキャッシュを公開する。[実機分割テスト](benchmarks/ncnn_switch_20261003/partition-tests.log)は重みバイト保全・分岐・解析解との一致と非FP32の拒否の2件とも成功した。

[同一ultraの分割FP32測定](benchmarks/ncnn_switch_20261003/ultra-stream128-b1.json)はwarm-up 1回・測定1回で **75,004.67 ms/枚**、初回の分割生成と起動検証を含むロード217.71秒、終了コード0。全確率がCPU参照の `atol=1e-4, rtol=1e-3` 内、最大絶対差0.00000333786、rating最大差0.0000000111759、採用タグ差0件。[メモリ・GPU負荷記録](benchmarks/ncnn_switch_20261003/ultra-stream128-memory-watch.json)はRSSピーク660.27MiB、システム空きメモリ最小1,375.48MiB、Tegra GPU負荷最大99.7%。RSSにはGPU割り当て全量が含まれず、GPU負荷はデバイス全体の値。zramは共有RAMの圧縮領域であり、ディスクswapとは異なる。20回反復した他PCの中央値と同じ精度の速度推定としては扱わない。

[元重みと分割重みの全バイトSHA256照合](benchmarks/ncnn_switch_20261003/model-integrity.json)は一致し、binは `86c013447913351be6ab1688a5d801ed46e13a5a5b5a84d67c81fbcbb258f0e6` のまま。[実GPU統合テスト](benchmarks/ncnn_switch_20261003/integration.log)は2件成功（362.557秒、終了コード0）。全確率・rating・閾値と、単独実行／原画像転送／前処理済み転送のXMP一致を確認した。Client期限は600秒に設定した。[通常Bashランチャー](benchmarks/ncnn_switch_20261003/launcher-probe.log)の `--provider ncnn --gpu -m ultra --ncnn-part-size-mib 128 --gpu-index 0 --probe-provider` も終了コード0。再起動前の測定完了と再起動後の実GPU統合試験の双方を確認した。別モデルへの置き換えは行っていない。

### Intel Iris Xe／a13m追加調査（2026-10-03）

`ssh a13m`、指定フォルダ `C:\Users\kouki\Documents\MyApp\wd14-tagger-xmp` をGit経由で同じPRブランチへ更新した。[機種とドライバー](benchmarks/ncnn_intel_a13m_20261003/hardware.log)はCore i7-1355U、Intel Iris Xe、ドライバー32.0.101.7092、Windows 11 Home 10.0.26300、RAM約31.7GiB。ncnn 1.0.20260526はIris Xeの同名エントリーを4件列挙し、GPU 0を明示した。この列挙数を物理GPU 4台とは扱わない。

同じultra FP32・入力SHA256・metadataで、[batch 1](benchmarks/ncnn_intel_a13m_20261003/fp32-b1.json)はwarm-up 1回・測定3回、中央値 **32,900.08 ms/枚**、ロード40.62秒。[batch 4](benchmarks/ncnn_intel_a13m_20261003/fp32-b4.json)はwarm-up 1回・測定1回、**33,299.30 ms/枚**、ロード36.46秒。4枚は固定batch 1グラフを逐次実行した。両条件とも全12,476確率がCPU参照の `atol=1e-4, rtol=1e-3` 内、最大差0.00000327826、rating最大差0.0000000135042、採用タグ差0件、終了コード0。20回測定した他PCの中央値とは反復数が異なる。WindowsのRSS・GPU共有メモリ・使用率は未取得。

[実GPU統合テスト](benchmarks/ncnn_intel_a13m_20261003/integration.log)は2件成功（181.163秒、終了コード0）。全確率・rating・閾値、単独実行／原画像転送／前処理済み転送のXMP一致を確認し、Client期限は300秒に設定した。[PowerShell構文解析](benchmarks/ncnn_intel_a13m_20261003/powershell-parser.log)は0エラー。当日の[PowerShellランチャー記録](benchmarks/ncnn_intel_a13m_20261003/launcher-probe.log)は追加引数が未転送でPythonのヘルプ表示に終わっていたため、GPU起動検証としては扱わない。2026-10-04に転送を修正し、下記の実GPU検証を実施した。

[OpenVINO比較](benchmarks/ncnn_intel_a13m_20261003/openvino-b1.log)は実名 `Intel(R) Iris(R) Xe Graphics (iGPU)` のGPUを確認し、OpenVINOExecutionProviderがactiveでも起動検証で非有限出力を検出して終了コード1。測定JSONは生成せず、batch 4は未実行。成功速度として扱わない。最初の直接測定ではリポジトリ内の既存HFトークンが環境へ指定されていなかったため認証待ちになり、`HF_HOME`／`HF_TOKEN_PATH` を指定して再ログインなしで再開した。[環境・モデルSHA256](benchmarks/ncnn_intel_a13m_20261003/environment.json)も保存している。

IntelとSwitchのbin SHA256は同一。paramのSHA256は層名が異なるため一致しないが、[全層の型・接続・パラメーター照合](benchmarks/ncnn_intel_a13m_20261003/graph-integrity.json)は層名と空白を除いて一致した。[Intel全条件の精度照合](benchmarks/ncnn_intel_a13m_20261003/comparison-summary.json)と[Switchの精度照合](benchmarks/ncnn_switch_20261003/comparison-summary.json)を保存した。[最終通常テスト](benchmarks/ncnn_switch_20261003/unit-tests-final.log)は113件実行、実GPU用4件を除いて成功した。

### Intel OpenVINO停止の修正（2026-10-04）

同じa13m・ultra・ORT 1.24.1／OpenVINO 2025.4.1で精度設定だけを変えて調査した。[GPU既定値](benchmarks/openvino_intel_a13m_20261004/device-defaults.json)はFP16。[修正前と同じ設定](benchmarks/openvino_intel_a13m_20261004/default-control.json)と[明示FP16](benchmarks/openvino_intel_a13m_20261004/f16-control.json)は、いずれも起動検証の全12,476出力がNaNとなり停止した。出力バッファのdtypeはfloat32でも、内部のFP16計算は防げない。NaNが最初に生じる個別演算は未特定で、特定層のoverflowやdriverの不具合とは断定しない。

Intel providerへ `load_config={"GPU":{"INFERENCE_PRECISION_HINT":"f32"}}` を追加し、選択GPU・元モデル・入力・非有限値検査を保持した。設定方法は[ORT公式OpenVINO資料](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#load_config)に基づく。起動時の実行EPはOpenVINOExecutionProviderで、CPUに切り替えて成功させていない。

[修正後batch 1](benchmarks/openvino_intel_a13m_20261004/fp32-b1.json)はwarm-up 1回・測定3回、中央値 **1,599.69 ms/枚**、ロード90.22秒。[batch 4](benchmarks/openvino_intel_a13m_20261004/fp32-b4.json)もwarm-up 1回・測定3回で **1,542.60 ms/枚**、ロード16.97秒。全12,476確率がCPU参照の `atol=1e-4, rtol=1e-3` 内で、採用タグ差0件。[実GPU統合テスト](benchmarks/openvino_intel_a13m_20261004/integration.log)は2件成功（39.302秒）：全確率・rating・閾値、単独／原画像／前処理済み転送のXMP一致。WindowsのGPUメモリ・使用率は未測定で、3回反復の結果を20回測定と同じ精度の速度推定とは扱わない。

PowerShellの `RemainingArgs` がPythonへ渡っていなかった箇所も修正した。検証用引数を実際に渡し、[Intelランチャー](benchmarks/openvino_intel_a13m_20261004/launcher-probe.log)と[ncnnランチャー](benchmarks/openvino_intel_a13m_20261004/ncnn-launcher-probe.log)で実GPUの起動を検証した。以前のヘルプ表示・終了コード0を実GPU成功と読み替えない。

[通常のGPU自動選択](benchmarks/openvino_intel_a13m_20261004/auto-launcher-probe.log)もIntel OpenVINOを選び、GPU.0の実名とOpenVINO EPの実行を確認して終了コード0。明示指定しなくてもこのa13mでは修正後の経路を使う。[最終通常テスト](benchmarks/openvino_intel_a13m_20261004/unit-tests.log)は114件実行、実GPU用4件を除いて成功した。

### AERO RTX／UHD 630の再確認（2026-10-04）

AEROをGit経由で `9546b80` へ更新し、ultraのbatch 1／4を再検証した。[実機情報](benchmarks/aero_regression_20261004/hardware.log)はCore i7-8750H、Windows 11 Home 10.0.26200、RAM約31.9 GiB、RTX 2070 Max-Q（driver 32.0.16.1714）とIntel UHD 630（31.0.101.2145）。ncnnはRTXのVulkan index 2、OpenVINOは実名確認済みのIntel `GPU.0` を指定した。SwitchとAMDには接続していない。

[全8条件の照合](benchmarks/aero_regression_20261004/comparison-summary.json)で同じ入力SHA256・metadata version・全12,476確率をCPU参照と比較し、`atol=1e-4, rtol=1e-3` 内、採用タグ差0件、rating最大差1.2e-7未満を確認した。全試験・通常起動の終了コードは0。

| 実GPU経路 | batch 1 ms/枚 | batch 4 ms/枚 | CPUとの最大確率差（b1 / b4） |
| --- | ---: | ---: | --- |
| RTX ncnn FP32 | 1,012.66 | 1,014.69 | 0.00000357628 / 0.00000357628 |
| RTX CUDA | 403.69 | 397.79 | 0.00000303984 / 0.00000333786 |
| RTX TensorRT | 313.30 | 307.09 | 0.00000470877 / 0.00000470877 |
| UHD 630 OpenVINO FP32 | 7,967.46 | 8,637.68 | 0.00000545382 / 0.00000533462 |

warm-up 1回・測定3回。ncnn b4はbatch 1グラフの逐次処理。GPU試験は順番に実行したが、CUDA／TensorRT測定中に通常テストのCPU処理が短時間重なっている。旧20回測定との差を修正による速度改善とは扱わない。WindowsのGPUメモリ・負荷ピークは未測定。OpenVINOのロード・起動検証はb1で117.56秒、b4で115.12秒かかったが正常終了した。以前NaNで停止したUHD 630でも、今回のFP32指定で有効な出力を確認できた。

[RTX ncnn](benchmarks/aero_regression_20261004/rtx-ncnn-integration.log)、[CUDA](benchmarks/aero_regression_20261004/cuda-integration.log)、[TensorRT](benchmarks/aero_regression_20261004/tensorrt-integration.log)、[Intel OpenVINO](benchmarks/aero_regression_20261004/intel-openvino-integration.log)の統合テストは各2件成功。全確率・rating・閾値と、単独／原画像／前処理済みServer/Client転送のXMP一致を確認した。Client期限は300秒、Intelの2件全体は289.005秒。起動検証のログで各要求EPの実行も確認した。

[通常GPU自動起動](benchmarks/aero_regression_20261004/launcher-auto-rtx.log)はTensorRTを選択して実行EPを確認し、[Intel明示起動](benchmarks/aero_regression_20261004/launcher-intel.log)もUHD 630のOpenVINO実行を確認した。追加引数は実際に転送され、ヘルプ表示で終わっていない。ネイティブPowerShell子プロセスのログ取り込みでは日本語の一部が文字化けしているため、元の記録を保持し、判定には読み取れるデバイス名・EP名・終了コードを使った。

[通常テスト](benchmarks/aero_regression_20261004/unit-tests.log)は114件実行、8件skipで成功。4件はWindows対象外のBash試験、4件は別途有効化が必要なGPU試験。初回はテスト専用依存 `httpx` 不足で失敗し、`venv_intel` へ追加して全件を再実行した。今回の確認範囲では新たな推論・通信・XMPの異常を認めず、追加のアプリ修正は行っていない。

### AEROの応答停止・電源状態調査（2026-10-04）

上記試験後にSSH接続が途絶え、利用者から電源が落ちているとの報告を受けた。試験終了コードだけでホスト全体の正常性を判断した前の報告は不十分だった。復帰後の[電源イベント原文とXML](benchmarks/aero_regression_20261004/power-investigation/events.json)を取得し、次を確認した（日本時間）。

- 05:00:03：Kernel-Power 105の `AcOnline=false`。WindowsがAC電源なしを検出した。RemainingCapacity／FullChargeCapacityはどちらも94,240で、残量切れではない。
- 05:00:22：Kernel-Power 42、スリープ理由 `System Idle`。S3スリープへ移行。
- 05:35:48：Power-Troubleshooter 1による復帰記録。解除元は `Power Button`、元のスリープ時刻は05:00:22。OSの起動時刻は04:34:15のまま。

[電源ポリシー](benchmarks/aero_regression_20261004/power-investigation/sleep-policy.log)はSmartmanager Balanced、ACの自動スリープ0秒（無効）、DCは600秒（10分）。AC切断から19秒でスリープしているので、切断後に10分経過したとは解釈しない。無操作時間が既に蓄積していた可能性はあるが、その起点は未測定。今回の直接の応答停止原因はWindowsが記録した無操作S3スリープであり、AC検出の切り替わりが直前に起きている。ACが認識されなくなった物理的理由（抜線・接触・アダプター等）は特定できない。現在は `AcOnline=true`。直近の調査範囲ではKernel-Power 41、EventLog 6008、User32 1074、WHEAの障害記録はなく、新しいクラッシュダンプもない。Windowsの設定は変更していない。

[kernel.errors.txt](benchmarks/aero_regression_20261004/power-investigation/kernel.errors.txt)は04:39:54作成、04:53:42最終更新のCISA GPUカーネル検証診断で、Windowsカーネルのクラッシュダンプではない。2件の診断文は[OpenVINOの既存報告 #31511](https://github.com/openvinotoolkit/openvino/issues/31511)と一致する。同報告も出力が得られた状態での診断を扱っている。今回のFP32全確率・XMP検証は成功しているが、診断が解消済みとは主張しない。診断ファイル生成を今回のS3移行の原因と結び付ける証拠はない。S3では外観が電源OFFに見える場合があることは[Microsoftの電源状態資料](https://learn.microsoft.com/en-us/windows-hardware/drivers/kernel/system-power-states)にも記載されている。

利用者による追加確認で、充電器のSwitchBotに「毎朝05:00に1分間電源を切る」設定が残っていたことが判明した。05:00:03のAC切断と時刻が一致し、今回のスリープを引き起こした外部電源操作として説明できる。SwitchBot本体の履歴は未取得で、設定解除済みとは扱わない。追加確認でもスリープ前後のBootIdは266のまま、05:40時点のOS稼働時間は約66分、04:34:15の起動から継続していた。今回の接続停止調査は外部電源操作後の無操作S3スリープとして完了し、上記の推論・精度・通信・XMP試験結果は有効。IntelのCISA診断は別の依存ライブラリ側の記録として保持する。

### AERO UbuntuのNVIDIA確認（2026-10-04）

`ubuntu-komeiziaya` の `/opt/dbv4-wd14-tagger` をGit経由で更新し、RTX 2070 Max-Qのultraを確認した。[実機情報](benchmarks/nvidia_linux_20261004/hardware.log)はkernel 7.0.0-38-generic、NVIDIA driver 610.57.04、VRAM 8 GiB。Python 3.13.16、ORT GPU 1.26.0、TensorRT cu12 10.16.1.11。ncnn VulkanはRTXのindex 1、CUDA／TensorRTはindex 0を指定した。

| 経路 | batch 1 ms/枚 | batch 4 ms/枚 |
| --- | ---: | ---: |
| ncnn FP32 | 1,015.81 | 1,019.59 |
| CUDA | 402.75 | 397.56 |
| TensorRT | 295.60 | 296.39 |

warm-up 1回・反復3回、各GPU測定は順番に実行した。ncnn b4は固定batch 1グラフの逐次処理。[6条件の照合](benchmarks/nvidia_linux_20261004/comparison-summary.json)では同じ入力・metadataの全12,476確率がCPU参照の `atol=1e-4, rtol=1e-3` 内、採用タグ差0件、最大確率差4.65e-6未満、rating最大差1.2e-7未満。NVIDIAメモリ値はGPU全体の時点値でピークではない。Windowsとの速度差をOS単独の効果とは解釈しない。

初回の[TensorRT失敗](benchmarks/nvidia_linux_20261004/tensorrt-b1-initial.log)は `libnvonnxparser.so.10` の読み込みエラー。ライブラリはpip環境に存在したが、実装がnvinfer／pluginだけを先読みし、ORT bridgeが必要とするparserを先読みしていなかった。Linuxの読み込み処理へparserを追加し、ハンドルを保持するよう修正した。依存関係は[ORT公式のパッケージ定義](https://github.com/microsoft/onnxruntime/blob/main/setup.py)とも一致する。修正をGit経由で同期し、TensorRT b1／b4の実EP実行と数値一致を確認した。

[ncnn](benchmarks/nvidia_linux_20261004/ncnn-integration.log)、[CUDA](benchmarks/nvidia_linux_20261004/cuda-integration.log)、[TensorRT](benchmarks/nvidia_linux_20261004/tensorrt-integration.log)の実GPU統合テストは各2件成功（12.804／8.705／11.351秒）。単独／原画像／前処理済みServer/Client転送のXMP一致も確認した。[通常Bash自動起動](benchmarks/nvidia_linux_20261004/auto-launcher.log)はTensorRTを選び、実EP実行を確認して終了コード0。初回のCUDAへの切り替えは別ログとして保持している。修正後の全10終了コードは0、ローカルのGPU初期化関連テスト26件も成功。

### PS4 Linux・AMD Liverpool（2026-10-04 JST）

PS4 CUH-1200AのCachyOS、kernel `7.1.7-Strawberry-General-FullLTO+`、Mesa PS4 `26.0.4.217906.bbc23bed47f.2fc1477-1`、Python 3.13.15、ncnn 1.0.20260526で、ソース `6b518dc` を検証した。Vulkan GPU 0はAMD Liverpool（RADV）、RAM約5.8GiB、VRAM 2GiB。ホストのログ時刻はEDTのまま保存した。[実機情報](benchmarks/ncnn_ps4_20261004/environment.log)と[結果一覧](benchmarks/ncnn_ps4_20261004/comparison-summary.json)を参照。

既存のultra FP32モデルをGit経由で転送し、`--ncnn-part-size-mib 128` で14区間に分割した。最大区間は254,487,572 bytes（約242.7MiB）。binのSHA256は `86c013447913351be6ab1688a5d801ed46e13a5a5b5a84d67c81fbcbb258f0e6` で、分割binを連結した値も一致した。param・jsonも転送前と一致し、モデルの変更・再変換はしていない。[保全記録](benchmarks/ncnn_ps4_20261004/model-integrity.json)。元モデルと分割キャッシュで約5.2GiBのディスクを使用する。

| 条件 | 秒/枚 | RSS最大 MiB | VRAM最大 MiB | GTT最大 MiB |
|---|---:|---:|---:|---:|
| ultra FP32、分割128、batch 1 | 87.83 | 274.14 | 690.33 | 32.61 |
| ultra FP32、分割128、batch 4 | 80.30 | 229.41 | 690.33 | 36.61 |

warmup 1、測定1回。batch 4はbatch 1のグラフで4枚を順次実行し、JSONの精度比較は代表の先頭出力を使用する。全12,476確率がCPU参照と `atol=1e-4, rtol=1e-3` で一致し、最大絶対差2.742e-6、rating最大差1.397e-8、選択タグ差0。初回のキャッシュ生成・起動検証を含む初期化は296.66秒。測定回数が少ないため他機種との速度順位は判断しない。

メモリを0.2秒間隔で観測した。GPUメモリ値は画面表示などを含むデバイス全体の値で、GPU使用率・温度は取得できなかった。最小空きRAMはbatch 1で4825.54MiB、batch 4で4880.89MiB。実行はsystemd user serviceの `MemoryHigh=3G, MemoryMax=3500M, MemorySwapMax=1G` と外部タイムアウトで制限した。サービスのメモリ計上にはページキャッシュが含まれ、表のRSSとは異なる。OOMは観測していない。

通常テスト119件（skip 5）とネイティブ分割テスト2件が成功。実GPU統合テスト2件も成功し、単独処理・元画像アップロード・前処理済みアップロードのXMPが一致した。統合試験では `client_timeout` と `client_batch_timeout` を600秒に設定した（既定値は変更していない）。通常ランチャーの `--provider ncnn --gpu -m ultra --ncnn-part-size-mib 128 --gpu-index 0 --probe-provider` も実GPU起動検証に成功。自動選択は今回の試験対象外。保存した6件の終了コードはすべて0。

ExifToolは `.dbv4/runtime/exiftool` に配置し、実行時にそのディレクトリをPATHへ追加した。現在のGPU起動では既存の `RADV_DEBUG=syncshaders` 設定を使用する。各試験後のカーネルログにring timeout、GPU reset、GPU Recovery Failed、OOMはなかった。ただし[提供されたGist](https://gist.github.com/SyameimaruKoa/865677c33ddac427a406a754e20d6c82)のGeekbench Feature Matchingは実行しておらず、そのGPUハング原因や復旧失敗を修正したという結果ではない。

### 再測定

実機では[測定スクリプト](benchmark_ncnn.py)を各Providerの仮想環境で実行し、JSONを保存する。

```bash
bash benchmark_ncnn_matrix.sh benchmarks/ncnn_ultra
# 個別測定ではランチャーと同じ保存先を指定する
export DBV4_DATA_DIR="$PWD/.dbv4" HF_HOME="$PWD/.dbv4/huggingface"
venv_std/bin/python benchmark_ncnn.py --provider cpu --profile ultra --batch-size 1 --output benchmarks/ncnn_ultra_cpu_b1.json
venv_ncnn/bin/python benchmark_ncnn.py --provider ncnn --profile ultra --batch-size 1 --reference benchmarks/ncnn_ultra_cpu_b1.json --output benchmarks/ncnn_ultra_fp32_b1.json
venv_ncnn/bin/python benchmark_ncnn.py --provider ncnn --profile ultra --batch-size 4 --ncnn-precision fp16-storage --reference benchmarks/ncnn_ultra_cpu_b1.json --output benchmarks/ncnn_ultra_fp16_storage_b4.json
```

各JSONにはモデル取得・初期化・起動検証を含むロード時間、warm-up 3回後の20回のms/枚、生の反復時間、全ラベルの確率、rating 4値、best_threshold適用後のタグを記録する。取得可能ならNVIDIAのGPUメモリ、AMDのVRAM/GTT/GPU使用率、LinuxプロセスRSSを取得する。AMDカウンターは200 ms間隔、デバイス全体の値で他プロセスも含む。複数AMD GPUでは `--drm-device /sys/class/drm/cardN/device` を明示する。`--reference` は入力SHA256とmetadata versionも照合する。batch 4のncnnはbatch 1グラフを順に実行する。TensorRT/CUDA/OpenVINO/MIGraphX/WebGPUも対応する仮想環境で同じスクリプトを実行できる。

CPUでの変換照合は `python validate_ncnn_conversion.py --reference benchmarks/ncnn_ultra/cpu-b1.json --output benchmarks/ncnn_ultra/conversion-check.json`。実GPUの出力一致・XMP・Server/Client検証は `DBV4_NCNN_INTEGRATION=1 DBV4_NCNN_REFERENCE=benchmarks/ncnn_ultra/cpu-b1.json venv_ncnn/bin/python -m unittest discover -s tests -p test_ncnn_integration.py -v`。どちらも上記保存先の環境変数を指定する。XMP検証にはExifToolが必要。

集計日: 2026-09-23。[README](README.md) / [全100条件の詳細](benchmarks/matrix_details_20260923.md) / [READMEから移動した旧測定](benchmarks/legacy_results.md)

[速度比較](#速度比較) · [CPU比とbatch効果](#cpu比とbatch効果) · [メモリ](#メモリと起動時間) · [失敗条件](#失敗と追加確認が必要な条件) · [過去の結果](#過去の結果)

## 結果の要点

- NVIDIA RTX 2070 Max-Qでは、両OS・両モデル・両バッチでTensorRTが最速。CUDAより推論時間が約21～39%短い。メモリと初回構築時間も含めて選ぶ必要がある。
- Intel Iris Xe／HD 530のLinuxではOpenVINOが最速。HD 530のWebGPUはCPUより遅く、GPU経路なら必ず速くなるわけではない。
- AMD WindowsではDirectMLが全4条件で最速。WebGPUのCPU比は約1.03～1.42倍にとどまる。
- UHD 630のWebGPUは記録上最速だが、同OSのNVIDIA WebGPU値と約0.3～1.8%差しかない。選択アダプターの記録はIntelであるものの、物理GPUの実行先は追加確認が必要。この値だけでUHD 630の性能優位を断定しない。
- 全100条件のうち92成功、8失敗。失敗はWindowsのOpenVINO 4条件とDirectML 4条件。失敗・未測定を0 msとして扱わない。

## 対象と読み方

今回の7組の `manifest.json`・`summary.json`・個別結果100件、過去のWindows要約2件、AMD Linuxのメモリ測定2件、READMEの旧実測を対象とした。数値の一次資料は個別JSON。

- 時間は **ms/枚（小さいほど速い）**。今回の主表は前処理済み入力に対する `session.run` のみ。画像読み込み・前処理・モデル取得・セッション構築・XMP保存は含まない。
- seed `20260922` の640×480合成RGB画像を使用。同じ画像をbatch分複製し、warmup 3回、20回×3セット。各セットの中央値3個の中央値を採用。batch 4はバッチ時間を4で割った値であり、1件の応答待ち時間ではない。
- CPU比 = 同じ測定組・モデル・batchのCPU時間 ÷ 対象時間。1倍超が高速化。batch効果 = batch 1時間 ÷ batch 4時間。表の比較は丸め前の値で算出。
- ORTのプロファイリングを有効にした測定。成功行では要求EPのノード実行を確認しているが、EP名は物理GPUの実行証明やGPUのみで全演算を処理した証明ではない。出力shapeは確認済みだが、今回の100条件には出力数値の同等性・タグ品質の比較がない。
- ホストのCPU型番・ドライバー・電源設定・温度・モデルファイルhashは今回のJSONに揃っていない。OS間や異なるGPU間の差をハードウェア単独の差とは解釈しない。CPU基準も測定組ごとに異なる。

## 測定環境

全成功行のPythonは3.13.15。Linuxはkernel 7.0.0-31-generic / glibc 2.43、Windowsは11 / 10.0.26200。ORTはCPU・WebGPU 1.30.0、CUDA・TensorRT 1.26.0、OpenVINO 1.24.1、DirectML 1.24.4。EP比較にはORTのバージョン差も含まれる。


| 測定組 | 記録GPU / PCI device ID | WebGPU plugin | 成功 / 全条件 | 一次資料 |
| --- | --- | --- | --- | --- |
| RTX 2070 Max-Q / Linux | RTX 2070 Max-Q / 0x1f10 | 0.4.0 | 16 / 16 | [manifest](benchmarks/nvidia_linux_20260923/manifest.json) / [summary](benchmarks/nvidia_linux_20260923/summary.json) |
| RTX 2070 Max-Q / Windows | RTX 2070 Max-Q / 0x1f10 | 0.3.0 | 18 / 20 | [manifest](benchmarks/nvidia_windows_20260923/manifest.json) / [summary](benchmarks/nvidia_windows_20260923/summary.json) |
| UHD 630 / Linux | UHD 630 / 0x3e9b | 0.4.0 | 12 / 12 | [manifest](benchmarks/intel_linux_20260923/manifest.json) / [summary](benchmarks/intel_linux_20260923/summary.json) |
| UHD 630 / Windows | UHD 630 / 0x3e9b | 0.3.0 | 10 / 16 | [manifest](benchmarks/intel_windows_20260923/manifest.json) / [summary](benchmarks/intel_windows_20260923/summary.json) |
| Iris Xe / Linux | Iris Xe / 0xa7a1 | 0.4.0 | 12 / 12 | [manifest](benchmarks/intel2_linux_20260923/manifest.json) / [summary](benchmarks/intel2_linux_20260923/summary.json) |
| HD 530 / Linux | HD 530 / 0x1912 | 0.4.0 | 12 / 12 | [manifest](benchmarks/intel3_linux_20260923/manifest.json) / [summary](benchmarks/intel3_linux_20260923/summary.json) |
| Radeon Graphics / Windows | Radeon Graphics / 0x15e7 | 0.4.0 | 12 / 12 | [manifest](benchmarks/amd_windows_20260923/manifest.json) / [summary](benchmarks/amd_windows_20260923/summary.json) |

## 速度比較

太字は各行の成功した経路の最小値。UHD 630のWebGPUは「†」付きの参考値。`失敗` は実行失敗、`—` はその測定組の対象外。

### balanced

| 測定組 | batch | CPU | CUDA | TensorRT | OpenVINO | DirectML | WebGPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 2070 Max-Q / Linux | 1 | 593.10 | 58.16 | **46.06** | — | — | 120.08 |
| RTX 2070 Max-Q / Linux | 4 | 702.79 | 57.97 | **43.67** | — | — | 110.63 |
| RTX 2070 Max-Q / Windows | 1 | 944.55 | 76.75 | **60.90** | — | 失敗 | 197.99 |
| RTX 2070 Max-Q / Windows | 4 | 929.54 | 70.93 | **55.30** | — | 失敗 | 181.62 |
| UHD 630 / Linux | 1 | 853.91 | — | — | 800.19 | — | **118.26**† |
| UHD 630 / Linux | 4 | 938.49 | — | — | 780.84 | — | **109.04**† |
| UHD 630 / Windows | 1 | 980.37 | — | — | 失敗 | 失敗 | **197.02**† |
| UHD 630 / Windows | 4 | 930.74 | — | — | 失敗 | 失敗 | **181.36**† |
| Iris Xe / Linux | 1 | 524.27 | — | — | **222.12** | — | 430.66 |
| Iris Xe / Linux | 4 | 611.91 | — | — | **212.00** | — | 411.68 |
| HD 530 / Linux | 1 | 1015.37 | — | — | **689.83** | — | 2017.63 |
| HD 530 / Linux | 4 | 1189.94 | — | — | **658.96** | — | 1926.70 |
| Radeon Graphics / Windows | 1 | 808.58 | — | — | — | **562.46** | 736.16 |
| Radeon Graphics / Windows | 4 | 1063.75 | — | — | — | **576.66** | 749.72 |

### wd14_v3

| 測定組 | batch | CPU | CUDA | TensorRT | OpenVINO | DirectML | WebGPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 2070 Max-Q / Linux | 1 | 728.89 | 66.89 | **42.37** | — | — | 124.20 |
| RTX 2070 Max-Q / Linux | 4 | 767.28 | 62.31 | **39.49** | — | — | 115.29 |
| RTX 2070 Max-Q / Windows | 1 | 1052.67 | 87.53 | **53.72** | — | 208.45 | 214.92 |
| RTX 2070 Max-Q / Windows | 4 | 1032.19 | 74.39 | **46.88** | — | 128.72 | 194.55 |
| UHD 630 / Linux | 1 | 727.81 | — | — | 512.54 | — | **122.25**† |
| UHD 630 / Linux | 4 | 796.23 | — | — | 518.31 | — | **114.21**† |
| UHD 630 / Windows | 1 | 1121.18 | — | — | 失敗 | 2447.17 | **213.82**† |
| UHD 630 / Windows | 4 | 1044.63 | — | — | 失敗 | 2330.25 | **193.57**† |
| Iris Xe / Linux | 1 | 508.45 | — | — | **131.70** | — | 482.13 |
| Iris Xe / Linux | 4 | 648.49 | — | — | **131.43** | — | 470.14 |
| HD 530 / Linux | 1 | 1198.24 | — | — | **439.04** | — | 2202.89 |
| HD 530 / Linux | 4 | 1277.45 | — | — | **432.41** | — | 2088.64 |
| Radeon Graphics / Windows | 1 | 1015.19 | — | — | — | **635.64** | 984.13 |
| Radeon Graphics / Windows | 4 | 1145.20 | — | — | — | **626.38** | 937.04 |

## CPU比とbatch効果

各欄は `balanced / wd14_v3`。CPU比は同一batchのCPUが基準。CPU比の増加だけでbatch 4が速くなったとは判断せず、右端の実時間比も確認する。

| 測定組 | EP | CPU比 b1 | CPU比 b4 | batch効果 b1÷b4 |
| --- | --- | --- | --- | --- |
| RTX 2070 Max-Q / Linux | CUDA | 10.20× / 10.90× | 12.12× / 12.31× | 1.00× / 1.07× |
| RTX 2070 Max-Q / Linux | TensorRT | 12.88× / 17.20× | 16.09× / 19.43× | 1.05× / 1.07× |
| RTX 2070 Max-Q / Linux | WebGPU | 4.94× / 5.87× | 6.35× / 6.66× | 1.09× / 1.08× |
| RTX 2070 Max-Q / Windows | CUDA | 12.31× / 12.03× | 13.11× / 13.88× | 1.08× / 1.18× |
| RTX 2070 Max-Q / Windows | TensorRT | 15.51× / 19.60× | 16.81× / 22.02× | 1.10× / 1.15× |
| RTX 2070 Max-Q / Windows | DirectML | 失敗 / 5.05× | 失敗 / 8.02× | — / 1.62× |
| RTX 2070 Max-Q / Windows | WebGPU | 4.77× / 4.90× | 5.12× / 5.31× | 1.09× / 1.10× |
| UHD 630 / Linux | OpenVINO | 1.07× / 1.42× | 1.20× / 1.54× | 1.02× / 0.99× |
| UHD 630 / Linux | WebGPU† | 7.22× / 5.95× | 8.61× / 6.97× | 1.08× / 1.07× |
| UHD 630 / Windows | OpenVINO | 失敗 / 失敗 | 失敗 / 失敗 | — / — |
| UHD 630 / Windows | DirectML | 失敗 / 0.46× | 失敗 / 0.45× | — / 1.05× |
| UHD 630 / Windows | WebGPU† | 4.98× / 5.24× | 5.13× / 5.40× | 1.09× / 1.10× |
| Iris Xe / Linux | OpenVINO | 2.36× / 3.86× | 2.89× / 4.93× | 1.05× / 1.00× |
| Iris Xe / Linux | WebGPU | 1.22× / 1.05× | 1.49× / 1.38× | 1.05× / 1.03× |
| HD 530 / Linux | OpenVINO | 1.47× / 2.73× | 1.81× / 2.95× | 1.05× / 1.02× |
| HD 530 / Linux | WebGPU | 0.50× / 0.54× | 0.62× / 0.61× | 1.05× / 1.05× |
| Radeon Graphics / Windows | DirectML | 1.44× / 1.60× | 1.84× / 1.83× | 0.98× / 1.01× |
| Radeon Graphics / Windows | WebGPU | 1.10× / 1.03× | 1.42× / 1.22× | 0.98× / 1.05× |

### 解釈

- TensorRTの最小値はLinux `wd14_v3` b4の39.49 ms/枚（約25.32枚/秒）。Windows同条件は46.88 ms/枚。TensorRTはCUDA比でLinux約21～37%、Windows約21～39%短縮した。
- Iris XeのOpenVINOはCPU比2.36～4.93倍、HD 530は1.47～2.95倍。HD 530のWebGPUはCPU比0.50～0.62倍で、CPUより約1.6～2.0倍の時間を要する。
- AMD DirectMLはCPU比1.44～1.84倍。batch 4の1枚あたり時間はbalancedで約2.5%悪化、wd14_v3で約1.5%改善にとどまり、今回の条件では大きなbatchの効果は小さい。
- NVIDIA Windowsのwd14_v3 DirectMLはbatch 4で1.62倍の処理能力になるが、TensorRTより時間を要する。CPUは多くの組でbatch 4の1枚あたり時間が悪化した。同一画像の複製という条件も含め、batch 4を一律推奨する結果ではない。

## OS差と測定のばらつき

同じ記録GPU名のNVIDIAについて、Windows時間÷Linux時間を示す。1倍超は今回のLinux測定の方が速いことを表す。WebGPUはplugin 0.3.0と0.4.0の差もあるため、OS単独の効果ではない。

| EP | balanced b1 | balanced b4 | wd14_v3 b1 | wd14_v3 b4 |
| --- | --- | --- | --- | --- |
| CUDA | 1.32× | 1.22× | 1.31× | 1.19× |
| TensorRT | 1.32× | 1.27× | 1.27× | 1.19× |
| WebGPU | 1.65× | 1.64× | 1.73× | 1.69× |


セット間変動は `最大セット中央値÷最小セット中央値−1`。次表は各組で最も変動した条件で、統計的な信頼区間ではない。全条件のセット中央値範囲とp95は[詳細表](benchmarks/matrix_details_20260923.md)に記載。

| 測定組 | 最大変動の条件 | セット中央値範囲 ms/枚 | 変動 |
| --- | --- | --- | --- |
| RTX 2070 Max-Q / Linux | wd14_v3 b1 CUDA | 66.79～73.89 | 10.6% |
| RTX 2070 Max-Q / Windows | wd14_v3 b1 CUDA | 84.78～93.47 | 10.2% |
| UHD 630 / Linux | wd14_v3 b4 OpenVINO | 500.99～604.42 | 20.6% |
| UHD 630 / Windows | wd14_v3 b1 CPU | 1115.04～1138.62 | 2.1% |
| Iris Xe / Linux | wd14_v3 b1 CPU | 502.63～648.55 | 29.0% |
| HD 530 / Linux | balanced b4 CPU | 1163.33～1889.90 | 62.5% |
| Radeon Graphics / Windows | wd14_v3 b4 CPU | 1140.84～1313.78 | 15.2% |


HD 530のbalanced CPU b4は62.5%、Iris Xeのwd14_v3 CPU b1は29.0%の変動がある。これらを分母にしたCPU比は基準側の変動も受ける。小差を確定的な順位と解釈しない。

## メモリと起動時間

RSSはプロセスの常駐RAM、VRAM増分はGPU全体の測定前からピークまでの差。両者は異なる指標で、合算してモデル専有量とはしない。AMD・Intelの今回のVRAM欄は未測定（null）であり、0 MiBではない。メモリ監視はロードから推論までを含み、100 ms間隔の監視のため短いピークを捉えない可能性がある。

以下は各組・EPの成功条件を通した最小～最大値。詳細表には各条件の値とセッション構築秒数を掲載。初回モデル取得やTensorRTキャッシュ状態が統一されていないため、起動時間のランキングは作らない。

| 測定組 | EP | ピークRSS MiB | VRAM増分 MiB |
| --- | --- | --- | --- |
| RTX 2070 Max-Q / Linux | CPU | 1040.5～1863.1 | 未測定 |
| RTX 2070 Max-Q / Linux | CUDA | 1211.5～1260.8 | 1146.0～1147.0 |
| RTX 2070 Max-Q / Linux | TensorRT | 3150.2～3345.5 | 909.0～1323.0 |
| RTX 2070 Max-Q / Linux | WebGPU | 658.3～690.6 | 687.0～1832.0 |
| RTX 2070 Max-Q / Windows | CPU | 1045.9～1880.0 | 未測定 |
| RTX 2070 Max-Q / Windows | CUDA | 1017.3～1058.5 | 1131.0～1195.0 |
| RTX 2070 Max-Q / Windows | TensorRT | 2871.0～3096.5 | 939.0～1353.0 |
| RTX 2070 Max-Q / Windows | DirectML | 862.7～870.0 | 866.0～1968.0 |
| RTX 2070 Max-Q / Windows | WebGPU | 583.2～783.7 | 709.0～1863.0 |
| UHD 630 / Linux | CPU | 1071.2～1863.3 | 未測定 |
| UHD 630 / Linux | OpenVINO | 1550.2～2080.6 | 未測定 |
| UHD 630 / Linux | WebGPU | 653.6～699.2 | 未測定 |
| UHD 630 / Windows | CPU | 1045.8～1879.9 | 未測定 |
| UHD 630 / Windows | DirectML | 1498.7～2577.5 | 未測定 |
| UHD 630 / Windows | WebGPU | 603.3～781.9 | 未測定 |
| Iris Xe / Linux | CPU | 1094.4～1894.0 | 未測定 |
| Iris Xe / Linux | OpenVINO | 1517.6～1918.6 | 未測定 |
| Iris Xe / Linux | WebGPU | 595.8～625.6 | 未測定 |
| HD 530 / Linux | CPU | 1053.0～1847.3 | 未測定 |
| HD 530 / Linux | OpenVINO | 1538.0～1688.9 | 未測定 |
| HD 530 / Linux | WebGPU | 576.8～588.7 | 未測定 |
| Radeon Graphics / Windows | CPU | 1060.9～1878.4 | 未測定 |
| Radeon Graphics / Windows | DirectML | 601.1～745.7 | 未測定 |
| Radeon Graphics / Windows | WebGPU | 601.9～782.9 | 未測定 |


TensorRTは推論が最速だが、今回のピークRSSは約2.8～3.3 GiBでCUDAの約1.0～1.2 GiBより大きい。NVIDIA WebGPUはbatch 1ではVRAM増分が小さい一方、wd14_v3 b4では約1.8 GiBになりCUDA・TensorRTを上回る。省メモリ経路と一律には言えない。

## 失敗と追加確認が必要な条件

| 測定組 | 条件 | 件数 | 記録された原因 |
| --- | --- | --- | --- |
| Intel Windows | OpenVINO、両モデル×b1/b4 | 4 | `GPU.0` の実名がNVIDIA RTX 2070 Max-QのためIntel測定を拒否 |
| Intel Windows | DirectML、balanced×b1/b4 | 2 | 要求したDmlExecutionProviderがactiveにならずCPUのみ |
| NVIDIA Windows | DirectML、balanced×b1/b4 | 2 | 要求したDmlExecutionProviderがactiveにならずCPUのみ |

これらは当該環境・実行時点の失敗で、モデルやGPUの恒久的な非対応を意味しない。現行スクリプトにはDirectML graph optimizerの回避設定があるが、保存された失敗JSONを修正後の成功とは読み替えない。Intel OpenVINOは実名でIntel GPUを確認し、そのデバイスを指定する再測定が必要。

UHD 630のWebGPUはLinuxで109～122 ms/枚、Windowsで181～214 ms/枚と記録されている。同OSのRTX 2070 Max-Q値と極めて近く、両環境ともNVIDIAとIntelが列挙されている。JSONの `webgpu_hardware` はIntel、`webgpu_device_index` は1だが、プロファイルはEP名までの記録。この近さだけで誤選択とは断定できないものの、GPUごとの実負荷や実行アダプターの診断ログによる確認までは参考値として扱う。

AMD Linuxの今回と同条件のCPU/MIGraphX/WebGPUマトリクスは収録されていない。下記の旧AMD記録で代用せず、未測定として扱う。

## 過去の結果

### AMD Barcelo Linux: 専用VRAM 512 MiBと4 GiB

詳細条件・全5モデルのVRAM/GTT/RSSは [512 MiB測定](benchmarks/amd_barcelo_512mb.md) と [4 GiB測定](benchmarks/amd_barcelo_4gb.md) を参照。各JSONもその文書から参照できる。2026-09-22、単色合成画像、batch 1、warmup 3、10回×1セット、WebGPU plugin 0.3.0の別測定。

| モデル | 推論のみ 512 MiB → 4 GiB（ms/枚） | 短縮率 | 前処理込み 512 MiB → 4 GiB（ms/枚） |
| --- | --- | --- | --- |
| lightweight | 91.1 → 74.2 | 18.6% | 107.0 → 84.9 |
| balanced | 729.8 → 617.5 | 15.4% | 747.6 → 639.2 |
| high | 4,173.6 → 3,639.0 | 12.8% | 4,197.0 → 3,657.0 |
| ultra | 5,559.2 → 5,194.3 | 6.6% | 5,561.0 → 5,226.1 |
| wd14_v3 | 872.1 → 804.3 | 7.8% | 885.0 → 819.1 |

4 GiB設定では全モデルで短縮したが、専用VRAMの予約によりOSが利用できるRAMが約3.5 GiB減る。512 MiB時のultraには約6.6 GiBのシステムswapがあり、VRAM設定だけの因果効果とは断定できない。4 GiB時もultraは専用VRAMピーク約96%で、batch 4の余裕は未確認。

### 旧Windows要約

[nvidia_windows_summary.json](benchmarks/nvidia_windows_summary.json) は今回のマトリクスと数値が異なる別記録。CPU・生の反復時間・測定条件一式がないため、今回の92成功条件に混ぜない。次表で全12件を保存し比較する。

| モデル | batch | EP | ms/枚 | ピークRSS MiB | VRAM増分 MiB |
| --- | --- | --- | --- | --- | --- |
| balanced | 1 | CUDA | 78.833 | 1023.1 | 1129.0 |
| balanced | 1 | TensorRT | 61.159 | 3077.5 | 985.0 |
| balanced | 1 | WebGPU | 202.415 | 689.5 | 714.0 |
| balanced | 4 | CUDA | 71.936 | 1014.4 | 1142.0 |
| balanced | 4 | TensorRT | 55.615 | 3077.4 | 1228.0 |
| balanced | 4 | WebGPU | 182.170 | 729.3 | 1133.0 |
| wd14_v3 | 1 | CUDA | 85.446 | 1049.3 | 1133.0 |
| wd14_v3 | 1 | TensorRT | 53.898 | 2861.4 | 938.0 |
| wd14_v3 | 1 | WebGPU | 215.305 | 743.9 | 867.0 |
| wd14_v3 | 4 | CUDA | 74.800 | 1059.4 | 1127.0 |
| wd14_v3 | 4 | TensorRT | 46.933 | 2910.8 | 1353.0 |
| wd14_v3 | 4 | WebGPU | 195.604 | 780.3 | 1874.0 |


旧記録でもTensorRTがCUDAより22～37%短縮し、WebGPUはCUDAの2.52～2.62倍の時間だった。今回のWindows測定と傾向は一致するが、条件不足のため前後の改善率は算出しない。

[intel_windows_summary.json](benchmarks/intel_windows_summary.json) は成功0件。wd14_v3 b1のOpenVINOはGPUプログラム構築時のshared data上限超過（`0xc020 > 0xc000`）、DirectMLとWebGPUは `missing`。今回のIntel Windows結果とは別の失敗履歴であり、missingは実行失敗とは区別する。

### README掲載の旧実測

[旧測定記録](benchmarks/legacy_results.md) に、DirectML全5モデルのVRAM・速度表、承認待ちモデルの推定値、2026-09-21 Intel Iris Xeおよび2026-09-22 AMD Barceloの動作確認を移動した。推定値と実測値、単発推論と今回の反復測定を区別して参照する。

## 再測定・参照先

- [Windows測定手順](WEBGPU_BENCHMARK_WINDOWS.md) / [Linux測定手順](WEBGPU_BENCHMARK_LINUX.md)
- [測定ランナー](benchmark_matrix.py) / [速度・メモリ測定本体](benchmark_nvidia_ep.py) / [要約処理](summarize_benchmark_matrix.py)
- [全100条件の数値・失敗理由と個別JSONへのリンク](benchmarks/matrix_details_20260923.md)

2026-09-23マトリクスの分析では再推論を行っていない。保存済みの生データを維持し、同一条件の照合・計算・文書化のみ実施した。

### Issue #27: PS4自動分割の実機検証（2026-10-08 JST）

PR #23の説明にあったPS4の128MiB上限がコードに存在せず、容量推定だけで大区間・常駐を選ぶ状態だった。報告された1024MiBでのGPUリセットは再実行せず、LinuxのLiverpool／PlayStation 4／RADV Oberonについて、128MiBを超えるモデルの自動選択を最大128MiBに制限した。低予算では64MiBを維持し、明示指定は優先する。

実機Liverpool、kernel `7.1.7-Strawberry-General-FullLTO+`、ultra FP32で分割サイズを省略し、空きRAM 2723MiB、選択GPU予算3932MiB、重み2642MiBから128MiB／14区間（最大242.7MiB）を選択した。[測定JSON](benchmarks/ncnn_ps4_issue27_20261008/auto-b1.json)と[起動ログ](benchmarks/ncnn_ps4_issue27_20261008/auto-b1.log)。起動検証114.32秒、warmup 1回・測定1回で98.92秒/枚、終了コード0。全12,476確率が有限かつCPU参照の `atol=1e-4, rtol=1e-3` 内、最大差2.742e-6、採用タグ差0。RSS最大218.45MiB、VRAM最大715.11MiB、GTT最大32.61MiB。新規のring timeout／GPU reset／GPU Recovery Failed／OOMログなし。[検証結果](benchmarks/ncnn_ps4_issue27_20261008/validation.json)。

通常テスト132件（skip 5）、ネイティブ分割テスト2件、Bash構文検証とdiff検査が成功。今回の測定は既存キャッシュを使用し、XMP／Server／Clientの再検証は含まない。単回測定から速度改善は主張しない。毎画像の再読み込みは残り、この対策はドライバー自体の修復や明示的な大区間の安全性を保証しない。

#### 区間を共有する4枚処理

画像ごとに全14区間を再ロードする順序を、区間ごとに最大4枚を順次推論する順序へ変更した。アプリ／Serverに公開するバッチ上限も1から4へ変更し、既定batch-size=4が適用される。GPUには1区間・1画像ずつ載せ、中間出力をCPUに最大4枚保持する。4枚で重みロードは56回から14回になる。画像の前処理そのものは各画像に必要で、1枚だけの依頼では再ロードコストが残る。

同じ実機・ultra FP32・自動128MiB・既存キャッシュで[4枚処理](benchmarks/ncnn_ps4_issue27_20261008/section-batch4.json)をwarmup 1回／測定1回実行し、86.48秒/枚、起動102.70秒、終了0。直前の単画像測定98.92秒/枚より約12.6%短いが、バッチ数とキャッシュ状態が異なる単回比較であり、旧方式のbatch4に対する改善率ではない。代表の先頭画像の全確率がCPU参照の許容差内、タグ差0、新規GPU reset／OOMなし。入力は同じ画像4枚で、実機JSONの精度比較は先頭画像のみ。通常テスト133件成功（skip 5）で異なる入力の順序維持、区間共有、5枚の分割と空入力、同時リクエストの排他を確認した。
