# ベンチマーク結果と分析

集計日: 2026-09-23。[README](README.md) / [全100条件の詳細](benchmarks/matrix_details_20260923.md) / [READMEから移動した旧測定](benchmarks/legacy_results.md)

[速度比較](#速度比較) · [CPU比とbatch効果](#cpu比とbatch効果) · [メモリ](#メモリと起動時間) · [失敗条件](#失敗と追加確認が必要な条件) · [過去の結果](#過去の結果)

## 結果の要点

- NVIDIA RTX 2070 Max-Qでは、両OS・両モデル・両バッチでTensorRTが最速。CUDAより推論時間が約21～39%短い。メモリと初回構築時間も含めて選ぶ必要がある。
- Intel Iris Xe／HD 530のLinuxではOpenVINOが最速。HD 530のWebGPUはCPUより遅く、GPU経路なら必ず速くなるわけではない。
- AMD WindowsではDirectMLが全4条件で最速。WebGPUのCPU比は約1.03～1.42倍にとどまる。
- UHD 630のWebGPUは記録上最速だが、同OSのNVIDIA WebGPU値と約0.3～1.8%差しかない。選択アダプターの記録はIntelであるものの、物理GPUの実行先は追加確認が必要。この値だけでUHD 630の性能優位を断定しない。
- 全100条件のうち92成功、8失敗。失敗はWindowsのOpenVINO 4条件とDirectML 4条件。失敗・未測定を0 msとして扱わない。

## 対象と読み方

今回の7組の `manifest.json`・`summary.json`・個別結果100件、過去のWindows要約2件、AMD Linuxのメモリ測定2件、READMEの旧実測を対象とした。数値の一次資料は個別JSON。ルートの `onnxruntime_profile__*.json` は空ファイルまたは初期化イベントのみで、追加の完了した速度測定ではない。

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

今回の分析では再推論を行っていない。保存済みの生データを維持し、同一条件の照合・計算・文書化のみ実施した。
