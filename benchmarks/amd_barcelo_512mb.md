# AMD Barcelo 512 MiB WebGPU baseline（2026-09-22）

後で BIOS の専用 VRAM を 4 GiB に変更した状態と比較するための基準値。生データは [amd_barcelo_512mb.json](amd_barcelo_512mb.json)、再測定コードは [benchmark_webgpu_memory.py](../benchmark_webgpu_memory.py)。

4 GiB時の測定と比較結果は [amd_barcelo_4gb.md](amd_barcelo_4gb.md) に記録した。

## 環境と測定条件

- Ubuntu 26.04.1、kernel 7.0.0-31-generic、Ryzen 5 7530U 内蔵 Radeon Graphics（PCI 1002:15e7）、Mesa RADV 26.0.8。
- 専用 VRAM 512 MiB、GTT（共有 GPU メモリ）の上限約 7,661 MiB、物理 RAM 約 15,322 MiB。ONNX Runtime 1.30.0、WebGPU EP 0.3.0、Python 3.13.15。
- 合成 RGB 640×480 単色画像 `(127, 63, 191)` を各モデル固有の前処理に通す。batch size 1、warmup 3 回、各経路 10 回。推論のみは前処理済み入力の `session.run`、処理全体は `predict_images`（前処理・推論・出力変換）を計時。表は中央値。モデル取得、起動、XMP、レポート処理は速度に含めない。
- 各モデルを独立プロセスで実行。50 ms 間隔で `/sys/class/drm/card1/device/mem_info_vram_used`、`mem_info_gtt_used` と子プロセスの `VmRSS` を記録。VRAM/GTT はシステム全体のカウンターであり、表の「増分」は各測定の直前とピークの差。専用 VRAM と GTT を単純に足してモデル占有量と解釈しない。

| モデル | 専用 VRAM 開始→ピーク MiB（増分） | GTT 開始→ピーク MiB（増分） | プロセス RAM ピーク RSS MiB | 推論のみ ms/枚 | 前処理込み ms/枚 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lightweight | 472.9→493.6（+20.7） | 311.4→610.2（+298.7） | 380.2 | 91.1 | 107.0 |
| balanced | 462.9→504.1（+41.2） | 333.9→974.9（+641.0） | 936.1 | 729.8 | 747.6 |
| high | 450.5→504.9（+54.4） | 348.3→1,934.8（+1,586.5） | 2,194.2 | 4,173.6 | 4,197.0 |
| ultra | 480.7→507.8（+27.0） | 260.0→3,585.8（+3,325.8） | 2,092.1 | 5,559.2 | 5,561.0 |
| wd14_v3 | 468.6→504.2（+35.6） | 319.9→1,169.6（+849.7） | 881.5 | 872.1 | 885.0 |

ピーク時の使用率は以下。専用 VRAM と GTT はシステム全体の使用率、RAM は測定プロセスの RSS を物理 RAM 総量で割った値であり、システム全体の RAM 使用率ではない。

| モデル | 専用 VRAM 使用率 | GTT 使用率 | プロセス RSS / 物理 RAM |
| --- | ---: | ---: | ---: |
| lightweight | 96.4% | 8.0% | 2.5% |
| balanced | 98.5% | 12.7% | 6.1% |
| high | 98.6% | 25.3% | 14.3% |
| ultra | 99.2% | 46.8% | 13.7% |
| wd14_v3 | 98.5% | 15.3% | 5.8% |

5 モデルとも active provider は `WebGpuExecutionProvider` と `CPUExecutionProvider`。各モデルの演算子別 GPU/CPU 割り当ては未測定。過去の WD14 V3 単発プロファイルでは WebGPU 1,693 ノード、CPU 885 ノードだった。
測定前からデスクトップ等で専用 VRAM の約 450～481 MiB が使用されていたため、ピーク使用率が高いことだけで各モデルの必要 VRAM を判断できない。

## 測定できなかったプロファイルと制約

- `compact_manual` と `medium_manual`: Hugging Face にはログイン済みだが、`selected_tags.csv` の取得が 403。リポジトリ管理者の承認待ちと返されたため、モデル本体を取得せず測定しなかった。
- `future_1b`: このリポジトリでは ONNX モデル未公開として実行不可。
- `ultra`: モデル読み込み中からシステム swap 使用量が約 6.6 GiB に増えた。表の RSS は swap されたページを含まず、実際のメモリ負荷を表しきらない。5.6 秒/枚はこのメモリ圧迫下の値として扱う。
- 単一の合成画像、batch size 1、各 10 回の実測値。実画像の品質や大量処理の持続性能、VRAM 4 GiB 時の性能はこの表から断定できない。
- 生データの `load_seconds` は初回のモデル取得時間を含む場合があり、モデル間の起動性能比較には使わない。

## 4 GiB 設定後の再測定

同じ PC で VRAM 設定を変更して再起動後、専用 VRAM の値を確認し、同じスクリプト・画像・batch size・warmup・反復回数で実行する。結果は別ファイルへ保存し、この 512 MiB の生データは保持する。

```bash
cat /sys/class/drm/card1/device/mem_info_vram_total
venv_webgpu/bin/python benchmark_webgpu_memory.py \
  lightweight balanced high ultra wd14_v3 \
  --iterations 10 --output benchmarks/amd_barcelo_4gb.json
```

4 GiB 時はプロセス swap、システム swap、空き RAM の推移も記録し、512 MiB 時の `ultra` にあったメモリ圧迫が解消したか確認する。`compact_manual` と `medium_manual` は承認後に測定し、異なる時点・条件の結果として区別する。
