# 全測定の速度・RAM・VRAM一覧

[総合記録](../BENCHMARKS.md) / [使い方](../README.md)

保存済みJSONから集計（2026-10-04）。単位は時間ms/枚、メモリMiB、使用率%。未測定は0ではない。

- RSSはプロセス常駐RAM。RAM使用率はRSS÷記録された総RAMで、システム全体の使用率ではない。
- VRAMピーク・増分・スナップショットは別指標。GPU全体の値には画面表示や他プロセスも含む。スナップショット最大は推論中の連続監視ピークではない。
- VRAM使用率はピーク÷記録されたVRAM総容量。Intel/AMDの共有メモリ、GTT、Switchの統合RAMは独立VRAMと合算しない。総容量未記録では割合を算出しない。
- Linux AEROのGPUメモリ時点使用率は同じ測定組のhardware.logにある8192MiBを分母にする。旧DirectMLはlegacy_results.mdの8GiBを使用。
- GPU稼働率は測定中の平均で、VRAM使用率とは異なる。旧DirectMLは前処理込み3回の平均で、中央値と混ぜない。Windowsの最新測定ではRSS監視がなく、未測定として表示する。
- 2026-09-23はwarmup3・20回×3セットのsession.run時間。最新ultraは画像が異なり、warmup/回数も各JSONに従う。ncnnのbatch4は順次4枚。PS4/Switchは測定1回。
- 初期化時間は記録方式によってセッション構築のみ／取得・変換・起動検証込みが異なる。別日・別OS・別モデルの値から条件を揃えた性能差は断定しない。
- AMD診断の精度不一致や失敗も残す。「記録あり」は速度が保存された意味で、精度合格の意味ではない。ΔはCPU参照との最大絶対差。全確率の合格判定は各comparison-summary.jsonを参照。

## 測定組

- [aero_regression_20261004](#aero_regression_20261004)
- [amd_barcelo_4gb](#amd_barcelo_4gb)
- [amd_barcelo_512mb](#amd_barcelo_512mb)
- [amd_windows_20260923](#amd_windows_20260923)
- [intel2_linux_20260923](#intel2_linux_20260923)
- [intel3_linux_20260923](#intel3_linux_20260923)
- [intel_linux_20260923](#intel_linux_20260923)
- [intel_windows_20260923](#intel_windows_20260923)
- [ncnn_amd_20261003](#ncnn_amd_20261003)
- [ncnn_amd_bazzite_20261003](#ncnn_amd_bazzite_20261003)
- [ncnn_intel_a13m_20261003](#ncnn_intel_a13m_20261003)
- [ncnn_local_20261002](#ncnn_local_20261002)
- [ncnn_ps4_20261004](#ncnn_ps4_20261004)
- [ncnn_switch_20261003](#ncnn_switch_20261003)
- [ncnn_windows_20261003](#ncnn_windows_20261003)
- [nvidia_linux_20260923](#nvidia_linux_20260923)
- [nvidia_linux_20261004](#nvidia_linux_20261004)
- [nvidia_windows_20260923](#nvidia_windows_20260923)
- [nvidia_windows_summary](#nvidia_windows_summary)
- [openvino_intel_a13m_20261004](#openvino_intel_a13m_20261004)
- [directml_legacy](#directml_legacy)

## aero_regression_20261004

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [cuda-b1](aero_regression_20261004/cuda-b1.json) | ultra | 1 | cuda | — / 0 | 403.69 | 9.76 | 記録あり; Δ 3.04e-06; タグ差0 |
| [cuda-b4](aero_regression_20261004/cuda-b4.json) | ultra | 4 | cuda | — / 0 | 397.79 | 4.39 | 記録あり; Δ 3.34e-06; タグ差0 |
| [intel-openvino-b1](aero_regression_20261004/intel-openvino-b1.json) | ultra | 1 | intel | — / 0 | 7,967.46 | 117.56 | 記録あり; Δ 5.45e-06; タグ差0 |
| [intel-openvino-b4](aero_regression_20261004/intel-openvino-b4.json) | ultra | 4 | intel | — / 0 | 8,637.68 | 115.12 | 記録あり; Δ 5.33e-06; タグ差0 |
| [rtx-ncnn-b1](aero_regression_20261004/rtx-ncnn-b1.json) | ultra | 1 | ncnn | fp32 / 0 | 1,012.66 | 7.29 | 記録あり; Δ 3.58e-06; タグ差0 |
| [rtx-ncnn-b4](aero_regression_20261004/rtx-ncnn-b4.json) | ultra | 4 | ncnn | fp32 / 0 | 1,014.69 | 4.76 | 記録あり; Δ 3.58e-06; タグ差0 |
| [tensorrt-b1](aero_regression_20261004/tensorrt-b1.json) | ultra | 1 | tensorrt | — / 0 | 313.30 | 10.97 | 記録あり; Δ 4.71e-06; タグ差0 |
| [tensorrt-b4](aero_regression_20261004/tensorrt-b4.json) | ultra | 4 | tensorrt | — / 0 | 307.09 | 6.23 | 記録あり; Δ 4.71e-06; タグ差0 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [cuda-b1 / ultra / b1](aero_regression_20261004/cuda-b1.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 4,297.00 | 未測定 | 未測定 |
| [cuda-b4 / ultra / b4](aero_regression_20261004/cuda-b4.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 5,323.00 | 未測定 | 未測定 |
| [intel-openvino-b1 / ultra / b1](aero_regression_20261004/intel-openvino-b1.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [intel-openvino-b4 / ultra / b4](aero_regression_20261004/intel-openvino-b4.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [rtx-ncnn-b1 / ultra / b1](aero_regression_20261004/rtx-ncnn-b1.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [rtx-ncnn-b4 / ultra / b4](aero_regression_20261004/rtx-ncnn-b4.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [tensorrt-b1 / ultra / b1](aero_regression_20261004/tensorrt-b1.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 4,101.00 | 未測定 | 未測定 |
| [tensorrt-b4 / ultra / b4](aero_regression_20261004/tensorrt-b4.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 4,117.00 | 未測定 | 未測定 |

## amd_barcelo_4gb

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [amd_barcelo_4gb](amd_barcelo_4gb.json) | lightweight | 1 | WebGpuExecutionProvider+CPUExecutionProvider | — / 0 | 74.17 | 12.48 | ok; 未照合 |
| [amd_barcelo_4gb](amd_barcelo_4gb.json) | balanced | 1 | WebGpuExecutionProvider+CPUExecutionProvider | — / 0 | 617.53 | 25.34 | ok; 未照合 |
| [amd_barcelo_4gb](amd_barcelo_4gb.json) | high | 1 | WebGpuExecutionProvider+CPUExecutionProvider | — / 0 | 3,639.04 | 51.39 | ok; 未照合 |
| [amd_barcelo_4gb](amd_barcelo_4gb.json) | ultra | 1 | WebGpuExecutionProvider+CPUExecutionProvider | — / 0 | 5,194.29 | 4.26 | ok; 未照合 |
| [amd_barcelo_4gb](amd_barcelo_4gb.json) | wd14_v3 | 1 | WebGpuExecutionProvider+CPUExecutionProvider | — / 0 | 804.30 | 2.42 | ok; 未照合 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [amd_barcelo_4gb / lightweight / b1](amd_barcelo_4gb.json) | 393.60 | 3.34 | 764.52 | 18.66 | 300.80 | 未測定 | 94.77 | 未測定 |
| [amd_barcelo_4gb / balanced / b1](amd_barcelo_4gb.json) | 936.32 | 7.94 | 1,116.21 | 27.25 | 652.50 | 未測定 | 98.81 | 未測定 |
| [amd_barcelo_4gb / high / b1](amd_barcelo_4gb.json) | 2,196.20 | 18.62 | 2,145.53 | 52.38 | 1,681.80 | 未測定 | 110.81 | 未測定 |
| [amd_barcelo_4gb / ultra / b1](amd_barcelo_4gb.json) | 257.81 | 2.19 | 3,926.68 | 95.87 | 3,369.50 | 未測定 | 161.06 | 未測定 |
| [amd_barcelo_4gb / wd14_v3 / b1](amd_barcelo_4gb.json) | 646.02 | 5.48 | 1,425.30 | 34.80 | 949.00 | 未測定 | 100.80 | 未測定 |

## amd_barcelo_512mb

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [amd_barcelo_512mb](amd_barcelo_512mb.json) | wd14_v3 | 1 | WebGpuExecutionProvider+CPUExecutionProvider | — / 0 | 872.12 | 20.38 | ok; 未照合 |
| [amd_barcelo_512mb](amd_barcelo_512mb.json) | lightweight | 1 | WebGpuExecutionProvider+CPUExecutionProvider | — / 0 | 91.12 | 24.51 | ok; 未照合 |
| [amd_barcelo_512mb](amd_barcelo_512mb.json) | balanced | 1 | WebGpuExecutionProvider+CPUExecutionProvider | — / 0 | 729.78 | 25.89 | ok; 未照合 |
| [amd_barcelo_512mb](amd_barcelo_512mb.json) | high | 1 | WebGpuExecutionProvider+CPUExecutionProvider | — / 0 | 4,173.63 | 46.89 | ok; 未照合 |
| [amd_barcelo_512mb](amd_barcelo_512mb.json) | ultra | 1 | WebGpuExecutionProvider+CPUExecutionProvider | — / 0 | 5,559.19 | 109.66 | ok; 未照合 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [amd_barcelo_512mb / wd14_v3 / b1](amd_barcelo_512mb.json) | 881.48 | 5.75 | 504.15 | 98.47 | 35.60 | 未測定 | 1,169.59 | 未測定 |
| [amd_barcelo_512mb / lightweight / b1](amd_barcelo_512mb.json) | 380.24 | 2.48 | 493.61 | 96.41 | 20.70 | 未測定 | 610.15 | 未測定 |
| [amd_barcelo_512mb / balanced / b1](amd_barcelo_512mb.json) | 936.11 | 6.11 | 504.12 | 98.46 | 41.20 | 未測定 | 974.95 | 未測定 |
| [amd_barcelo_512mb / high / b1](amd_barcelo_512mb.json) | 2,194.16 | 14.32 | 504.88 | 98.61 | 54.40 | 未測定 | 1,934.83 | 未測定 |
| [amd_barcelo_512mb / ultra / b1](amd_barcelo_512mb.json) | 2,092.11 | 13.65 | 507.78 | 99.18 | 27.00 | 未測定 | 3,585.79 | 未測定 |

## amd_windows_20260923

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [balanced-b1-cpu](amd_windows_20260923/balanced-b1-cpu.json) | balanced | 1 | cpu | — / 0 | 808.58 | 1.64 | ok; 未照合 |
| [balanced-b1-directml](amd_windows_20260923/balanced-b1-directml.json) | balanced | 1 | directml | — / 0 | 562.46 | 1.85 | ok; 未照合 |
| [balanced-b1-webgpu](amd_windows_20260923/balanced-b1-webgpu.json) | balanced | 1 | webgpu | — / 0 | 736.16 | 2.21 | ok; 未照合 |
| [balanced-b4-cpu](amd_windows_20260923/balanced-b4-cpu.json) | balanced | 4 | cpu | — / 0 | 1,063.75 | 1.06 | ok; 未照合 |
| [balanced-b4-directml](amd_windows_20260923/balanced-b4-directml.json) | balanced | 4 | directml | — / 0 | 576.66 | 1.47 | ok; 未照合 |
| [balanced-b4-webgpu](amd_windows_20260923/balanced-b4-webgpu.json) | balanced | 4 | webgpu | — / 0 | 749.72 | 2.84 | ok; 未照合 |
| [wd14_v3-b1-cpu](amd_windows_20260923/wd14_v3-b1-cpu.json) | wd14_v3 | 1 | cpu | — / 0 | 1,015.19 | 2.05 | ok; 未照合 |
| [wd14_v3-b1-directml](amd_windows_20260923/wd14_v3-b1-directml.json) | wd14_v3 | 1 | directml | — / 0 | 635.64 | 3.28 | ok; 未照合 |
| [wd14_v3-b1-webgpu](amd_windows_20260923/wd14_v3-b1-webgpu.json) | wd14_v3 | 1 | webgpu | — / 0 | 984.13 | 4.23 | ok; 未照合 |
| [wd14_v3-b4-cpu](amd_windows_20260923/wd14_v3-b4-cpu.json) | wd14_v3 | 4 | cpu | — / 0 | 1,145.20 | 2.04 | ok; 未照合 |
| [wd14_v3-b4-directml](amd_windows_20260923/wd14_v3-b4-directml.json) | wd14_v3 | 4 | directml | — / 0 | 626.38 | 4.20 | ok; 未照合 |
| [wd14_v3-b4-webgpu](amd_windows_20260923/wd14_v3-b4-webgpu.json) | wd14_v3 | 4 | webgpu | — / 0 | 937.04 | 4.34 | ok; 未照合 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [balanced-b1-cpu / balanced / b1](amd_windows_20260923/balanced-b1-cpu.json) | 1,060.88 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-directml / balanced / b1](amd_windows_20260923/balanced-b1-directml.json) | 636.77 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-webgpu / balanced / b1](amd_windows_20260923/balanced-b1-webgpu.json) | 718.79 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-cpu / balanced / b4](amd_windows_20260923/balanced-b4-cpu.json) | 1,367.15 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-directml / balanced / b4](amd_windows_20260923/balanced-b4-directml.json) | 601.12 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-webgpu / balanced / b4](amd_windows_20260923/balanced-b4-webgpu.json) | 601.95 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-cpu / wd14_v3 / b1](amd_windows_20260923/wd14_v3-b1-cpu.json) | 1,265.62 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-directml / wd14_v3 / b1](amd_windows_20260923/wd14_v3-b1-directml.json) | 745.66 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-webgpu / wd14_v3 / b1](amd_windows_20260923/wd14_v3-b1-webgpu.json) | 782.88 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-cpu / wd14_v3 / b4](amd_windows_20260923/wd14_v3-b4-cpu.json) | 1,878.36 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-directml / wd14_v3 / b4](amd_windows_20260923/wd14_v3-b4-directml.json) | 633.63 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-webgpu / wd14_v3 / b4](amd_windows_20260923/wd14_v3-b4-webgpu.json) | 708.09 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |

## intel2_linux_20260923

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [balanced-b1-cpu](intel2_linux_20260923/balanced-b1-cpu.json) | balanced | 1 | cpu | — / 0 | 524.27 | 0.80 | ok; 未照合 |
| [balanced-b1-intel](intel2_linux_20260923/balanced-b1-intel.json) | balanced | 1 | intel | — / 0 | 222.12 | 6.89 | ok; 未照合 |
| [balanced-b1-webgpu](intel2_linux_20260923/balanced-b1-webgpu.json) | balanced | 1 | webgpu | — / 0 | 430.66 | 1.56 | ok; 未照合 |
| [balanced-b4-cpu](intel2_linux_20260923/balanced-b4-cpu.json) | balanced | 4 | cpu | — / 0 | 611.91 | 0.84 | ok; 未照合 |
| [balanced-b4-intel](intel2_linux_20260923/balanced-b4-intel.json) | balanced | 4 | intel | — / 0 | 212.00 | 4.59 | ok; 未照合 |
| [balanced-b4-webgpu](intel2_linux_20260923/balanced-b4-webgpu.json) | balanced | 4 | webgpu | — / 0 | 411.68 | 1.57 | ok; 未照合 |
| [wd14_v3-b1-cpu](intel2_linux_20260923/wd14_v3-b1-cpu.json) | wd14_v3 | 1 | cpu | — / 0 | 508.45 | 0.90 | ok; 未照合 |
| [wd14_v3-b1-intel](intel2_linux_20260923/wd14_v3-b1-intel.json) | wd14_v3 | 1 | intel | — / 0 | 131.70 | 4.69 | ok; 未照合 |
| [wd14_v3-b1-webgpu](intel2_linux_20260923/wd14_v3-b1-webgpu.json) | wd14_v3 | 1 | webgpu | — / 0 | 482.13 | 1.86 | ok; 未照合 |
| [wd14_v3-b4-cpu](intel2_linux_20260923/wd14_v3-b4-cpu.json) | wd14_v3 | 4 | cpu | — / 0 | 648.49 | 0.91 | ok; 未照合 |
| [wd14_v3-b4-intel](intel2_linux_20260923/wd14_v3-b4-intel.json) | wd14_v3 | 4 | intel | — / 0 | 131.43 | 4.69 | ok; 未照合 |
| [wd14_v3-b4-webgpu](intel2_linux_20260923/wd14_v3-b4-webgpu.json) | wd14_v3 | 4 | webgpu | — / 0 | 470.14 | 1.64 | ok; 未照合 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [balanced-b1-cpu / balanced / b1](intel2_linux_20260923/balanced-b1-cpu.json) | 1,094.38 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-intel / balanced / b1](intel2_linux_20260923/balanced-b1-intel.json) | 1,918.58 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-webgpu / balanced / b1](intel2_linux_20260923/balanced-b1-webgpu.json) | 625.59 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-cpu / balanced / b4](intel2_linux_20260923/balanced-b4-cpu.json) | 1,380.54 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-intel / balanced / b4](intel2_linux_20260923/balanced-b4-intel.json) | 1,517.64 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-webgpu / balanced / b4](intel2_linux_20260923/balanced-b4-webgpu.json) | 601.07 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-cpu / wd14_v3 / b1](intel2_linux_20260923/wd14_v3-b1-cpu.json) | 1,267.35 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-intel / wd14_v3 / b1](intel2_linux_20260923/wd14_v3-b1-intel.json) | 1,656.09 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-webgpu / wd14_v3 / b1](intel2_linux_20260923/wd14_v3-b1-webgpu.json) | 595.79 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-cpu / wd14_v3 / b4](intel2_linux_20260923/wd14_v3-b4-cpu.json) | 1,893.98 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-intel / wd14_v3 / b4](intel2_linux_20260923/wd14_v3-b4-intel.json) | 1,656.07 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-webgpu / wd14_v3 / b4](intel2_linux_20260923/wd14_v3-b4-webgpu.json) | 597.27 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |

## intel3_linux_20260923

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [balanced-b1-cpu](intel3_linux_20260923/balanced-b1-cpu.json) | balanced | 1 | cpu | — / 0 | 1,015.37 | 1.72 | ok; 未照合 |
| [balanced-b1-intel](intel3_linux_20260923/balanced-b1-intel.json) | balanced | 1 | intel | — / 0 | 689.83 | 5.02 | ok; 未照合 |
| [balanced-b1-webgpu](intel3_linux_20260923/balanced-b1-webgpu.json) | balanced | 1 | webgpu | — / 0 | 2,017.63 | 1.87 | ok; 未照合 |
| [balanced-b4-cpu](intel3_linux_20260923/balanced-b4-cpu.json) | balanced | 4 | cpu | — / 0 | 1,189.94 | 1.90 | ok; 未照合 |
| [balanced-b4-intel](intel3_linux_20260923/balanced-b4-intel.json) | balanced | 4 | intel | — / 0 | 658.96 | 5.23 | ok; 未照合 |
| [balanced-b4-webgpu](intel3_linux_20260923/balanced-b4-webgpu.json) | balanced | 4 | webgpu | — / 0 | 1,926.70 | 1.84 | ok; 未照合 |
| [wd14_v3-b1-cpu](intel3_linux_20260923/wd14_v3-b1-cpu.json) | wd14_v3 | 1 | cpu | — / 0 | 1,198.24 | 2.10 | ok; 未照合 |
| [wd14_v3-b1-intel](intel3_linux_20260923/wd14_v3-b1-intel.json) | wd14_v3 | 1 | intel | — / 0 | 439.04 | 5.16 | ok; 未照合 |
| [wd14_v3-b1-webgpu](intel3_linux_20260923/wd14_v3-b1-webgpu.json) | wd14_v3 | 1 | webgpu | — / 0 | 2,202.89 | 2.00 | ok; 未照合 |
| [wd14_v3-b4-cpu](intel3_linux_20260923/wd14_v3-b4-cpu.json) | wd14_v3 | 4 | cpu | — / 0 | 1,277.45 | 2.09 | ok; 未照合 |
| [wd14_v3-b4-intel](intel3_linux_20260923/wd14_v3-b4-intel.json) | wd14_v3 | 4 | intel | — / 0 | 432.41 | 5.10 | ok; 未照合 |
| [wd14_v3-b4-webgpu](intel3_linux_20260923/wd14_v3-b4-webgpu.json) | wd14_v3 | 4 | webgpu | — / 0 | 2,088.64 | 2.01 | ok; 未照合 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [balanced-b1-cpu / balanced / b1](intel3_linux_20260923/balanced-b1-cpu.json) | 1,053.02 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-intel / balanced / b1](intel3_linux_20260923/balanced-b1-intel.json) | 1,538.06 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-webgpu / balanced / b1](intel3_linux_20260923/balanced-b1-webgpu.json) | 576.84 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-cpu / balanced / b4](intel3_linux_20260923/balanced-b4-cpu.json) | 1,350.27 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-intel / balanced / b4](intel3_linux_20260923/balanced-b4-intel.json) | 1,538.02 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-webgpu / balanced / b4](intel3_linux_20260923/balanced-b4-webgpu.json) | 577.22 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-cpu / wd14_v3 / b1](intel3_linux_20260923/wd14_v3-b1-cpu.json) | 1,221.97 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-intel / wd14_v3 / b1](intel3_linux_20260923/wd14_v3-b1-intel.json) | 1,688.90 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-webgpu / wd14_v3 / b1](intel3_linux_20260923/wd14_v3-b1-webgpu.json) | 588.10 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-cpu / wd14_v3 / b4](intel3_linux_20260923/wd14_v3-b4-cpu.json) | 1,847.31 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-intel / wd14_v3 / b4](intel3_linux_20260923/wd14_v3-b4-intel.json) | 1,688.70 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-webgpu / wd14_v3 / b4](intel3_linux_20260923/wd14_v3-b4-webgpu.json) | 588.75 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |

## intel_linux_20260923

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [balanced-b1-cpu](intel_linux_20260923/balanced-b1-cpu.json) | balanced | 1 | cpu | — / 0 | 853.91 | 1.16 | ok; 未照合 |
| [balanced-b1-intel](intel_linux_20260923/balanced-b1-intel.json) | balanced | 1 | intel | — / 0 | 800.19 | 6.58 | ok; 未照合 |
| [balanced-b1-webgpu](intel_linux_20260923/balanced-b1-webgpu.json) | balanced | 1 | webgpu | — / 0 | 118.26 | 1.78 | ok; 未照合 |
| [balanced-b4-cpu](intel_linux_20260923/balanced-b4-cpu.json) | balanced | 4 | cpu | — / 0 | 938.49 | 1.35 | ok; 未照合 |
| [balanced-b4-intel](intel_linux_20260923/balanced-b4-intel.json) | balanced | 4 | intel | — / 0 | 780.84 | 3.59 | ok; 未照合 |
| [balanced-b4-webgpu](intel_linux_20260923/balanced-b4-webgpu.json) | balanced | 4 | webgpu | — / 0 | 109.04 | 1.79 | ok; 未照合 |
| [wd14_v3-b1-cpu](intel_linux_20260923/wd14_v3-b1-cpu.json) | wd14_v3 | 1 | cpu | — / 0 | 727.81 | 1.55 | ok; 未照合 |
| [wd14_v3-b1-intel](intel_linux_20260923/wd14_v3-b1-intel.json) | wd14_v3 | 1 | intel | — / 0 | 512.54 | 9.64 | ok; 未照合 |
| [wd14_v3-b1-webgpu](intel_linux_20260923/wd14_v3-b1-webgpu.json) | wd14_v3 | 1 | webgpu | — / 0 | 122.25 | 1.96 | ok; 未照合 |
| [wd14_v3-b4-cpu](intel_linux_20260923/wd14_v3-b4-cpu.json) | wd14_v3 | 4 | cpu | — / 0 | 796.23 | 1.60 | ok; 未照合 |
| [wd14_v3-b4-intel](intel_linux_20260923/wd14_v3-b4-intel.json) | wd14_v3 | 4 | intel | — / 0 | 518.31 | 3.73 | ok; 未照合 |
| [wd14_v3-b4-webgpu](intel_linux_20260923/wd14_v3-b4-webgpu.json) | wd14_v3 | 4 | webgpu | — / 0 | 114.21 | 1.97 | ok; 未照合 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [balanced-b1-cpu / balanced / b1](intel_linux_20260923/balanced-b1-cpu.json) | 1,071.17 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-intel / balanced / b1](intel_linux_20260923/balanced-b1-intel.json) | 1,924.47 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-webgpu / balanced / b1](intel_linux_20260923/balanced-b1-webgpu.json) | 699.18 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-cpu / balanced / b4](intel_linux_20260923/balanced-b4-cpu.json) | 1,360.68 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-intel / balanced / b4](intel_linux_20260923/balanced-b4-intel.json) | 1,550.22 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-webgpu / balanced / b4](intel_linux_20260923/balanced-b4-webgpu.json) | 698.71 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-cpu / wd14_v3 / b1](intel_linux_20260923/wd14_v3-b1-cpu.json) | 1,236.95 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-intel / wd14_v3 / b1](intel_linux_20260923/wd14_v3-b1-intel.json) | 2,080.57 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-webgpu / wd14_v3 / b1](intel_linux_20260923/wd14_v3-b1-webgpu.json) | 653.65 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-cpu / wd14_v3 / b4](intel_linux_20260923/wd14_v3-b4-cpu.json) | 1,863.27 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-intel / wd14_v3 / b4](intel_linux_20260923/wd14_v3-b4-intel.json) | 1,700.34 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-webgpu / wd14_v3 / b4](intel_linux_20260923/wd14_v3-b4-webgpu.json) | 660.81 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |

## intel_windows_20260923

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [balanced-b1-cpu](intel_windows_20260923/balanced-b1-cpu.json) | balanced | 1 | cpu | — / 0 | 980.37 | 1.47 | ok; 未照合 |
| [balanced-b1-directml](intel_windows_20260923/balanced-b1-directml.json) | balanced | 1 | directml | — / 0 | 未測定 | 未測定 | failed; 未照合 |
| [balanced-b1-intel](intel_windows_20260923/balanced-b1-intel.json) | balanced | 1 | intel | — / 0 | 未測定 | 未測定 | failed; 未照合 |
| [balanced-b1-webgpu](intel_windows_20260923/balanced-b1-webgpu.json) | balanced | 1 | webgpu | — / 0 | 197.02 | 2.42 | ok; 未照合 |
| [balanced-b4-cpu](intel_windows_20260923/balanced-b4-cpu.json) | balanced | 4 | cpu | — / 0 | 930.74 | 1.42 | ok; 未照合 |
| [balanced-b4-directml](intel_windows_20260923/balanced-b4-directml.json) | balanced | 4 | directml | — / 0 | 未測定 | 未測定 | failed; 未照合 |
| [balanced-b4-intel](intel_windows_20260923/balanced-b4-intel.json) | balanced | 4 | intel | — / 0 | 未測定 | 未測定 | failed; 未照合 |
| [balanced-b4-webgpu](intel_windows_20260923/balanced-b4-webgpu.json) | balanced | 4 | webgpu | — / 0 | 181.36 | 2.38 | ok; 未照合 |
| [wd14_v3-b1-cpu](intel_windows_20260923/wd14_v3-b1-cpu.json) | wd14_v3 | 1 | cpu | — / 0 | 1,121.18 | 2.28 | ok; 未照合 |
| [wd14_v3-b1-directml](intel_windows_20260923/wd14_v3-b1-directml.json) | wd14_v3 | 1 | directml | — / 0 | 2,447.17 | 2.16 | ok; 未照合 |
| [wd14_v3-b1-intel](intel_windows_20260923/wd14_v3-b1-intel.json) | wd14_v3 | 1 | intel | — / 0 | 未測定 | 未測定 | failed; 未照合 |
| [wd14_v3-b1-webgpu](intel_windows_20260923/wd14_v3-b1-webgpu.json) | wd14_v3 | 1 | webgpu | — / 0 | 213.82 | 2.50 | ok; 未照合 |
| [wd14_v3-b4-cpu](intel_windows_20260923/wd14_v3-b4-cpu.json) | wd14_v3 | 4 | cpu | — / 0 | 1,044.63 | 2.15 | ok; 未照合 |
| [wd14_v3-b4-directml](intel_windows_20260923/wd14_v3-b4-directml.json) | wd14_v3 | 4 | directml | — / 0 | 2,330.25 | 2.13 | ok; 未照合 |
| [wd14_v3-b4-intel](intel_windows_20260923/wd14_v3-b4-intel.json) | wd14_v3 | 4 | intel | — / 0 | 未測定 | 未測定 | failed; 未照合 |
| [wd14_v3-b4-webgpu](intel_windows_20260923/wd14_v3-b4-webgpu.json) | wd14_v3 | 4 | webgpu | — / 0 | 193.57 | 2.56 | ok; 未照合 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [balanced-b1-cpu / balanced / b1](intel_windows_20260923/balanced-b1-cpu.json) | 1,045.84 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-directml / balanced / b1](intel_windows_20260923/balanced-b1-directml.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-intel / balanced / b1](intel_windows_20260923/balanced-b1-intel.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-webgpu / balanced / b1](intel_windows_20260923/balanced-b1-webgpu.json) | 708.13 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-cpu / balanced / b4](intel_windows_20260923/balanced-b4-cpu.json) | 1,369.37 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-directml / balanced / b4](intel_windows_20260923/balanced-b4-directml.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-intel / balanced / b4](intel_windows_20260923/balanced-b4-intel.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-webgpu / balanced / b4](intel_windows_20260923/balanced-b4-webgpu.json) | 603.27 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-cpu / wd14_v3 / b1](intel_windows_20260923/wd14_v3-b1-cpu.json) | 1,252.09 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-directml / wd14_v3 / b1](intel_windows_20260923/wd14_v3-b1-directml.json) | 1,498.73 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-intel / wd14_v3 / b1](intel_windows_20260923/wd14_v3-b1-intel.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-webgpu / wd14_v3 / b1](intel_windows_20260923/wd14_v3-b1-webgpu.json) | 716.78 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-cpu / wd14_v3 / b4](intel_windows_20260923/wd14_v3-b4-cpu.json) | 1,879.89 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-directml / wd14_v3 / b4](intel_windows_20260923/wd14_v3-b4-directml.json) | 2,577.54 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-intel / wd14_v3 / b4](intel_windows_20260923/wd14_v3-b4-intel.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-webgpu / wd14_v3 / b4](intel_windows_20260923/wd14_v3-b4-webgpu.json) | 781.94 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |

## ncnn_amd_20261003

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [candidate-fp32](ncnn_amd_20261003/candidate-fp32.json) | ultra | 1 | ncnn | fp32 / 0 | 4,935.93 | 10.80 | 記録あり; Δ 0.0591; タグ差0 |
| [conv1x1-direct](ncnn_amd_20261003/conv1x1-direct.json) | ultra | 1 | ncnn | fp32 / 0 | 4,918.91 | 10.54 | 記録あり; Δ 0.0566; タグ差1（タグ不一致） |
| [conv1x1-fp32-b1](ncnn_amd_20261003/conv1x1-fp32-b1.json) | ultra | 1 | ncnn | fp32 / 0 | 4,882.48 | 10.51 | 記録あり; Δ 0.0861; タグ差1（タグ不一致） |
| [cpu-layernorm](ncnn_amd_20261003/cpu-layernorm.json) | ultra | 1 | ncnn | fp32 / 0 | 6,809.03 | 11.49 | 記録あり; Δ 0.192; タグ差3（タグ不一致） |
| [cpu-reduction](ncnn_amd_20261003/cpu-reduction.json) | ultra | 1 | ncnn | fp32 / 0 | 6,764.50 | 11.28 | 記録あり; Δ 0.141; タグ差5（タグ不一致） |
| [cpu-reshape](ncnn_amd_20261003/cpu-reshape.json) | ultra | 1 | ncnn | fp32 / 0 | 8,015.21 | 13.23 | 記録あり; Δ 0.0799; タグ差0 |
| [fp32-b1](ncnn_amd_20261003/fp32-b1.json) | ultra | 1 | ncnn | fp32 / 0 | 6,614.24 | 11.43 | 記録あり; Δ 0.123; タグ差4（タグ不一致） |
| [native-no-shared-memory](ncnn_amd_20261003/native-no-shared-memory.json) | ultra | 1 | ncnn | fp32 / 0 | 6,545.72 | 10.41 | 記録あり; Δ 0.144; タグ差2（タグ不一致） |
| [ncnn-cpu-check](ncnn_amd_20261003/ncnn-cpu-check.json) | ultra | 1 | ncnn CPU (conversion validation only) | fp32 / 0 | 5,503.91（cold 1回） | 1.41 | 記録あり; Δ 0.000157; タグ差0 |
| [ncnn-cpu-no-packing](ncnn_amd_20261003/ncnn-cpu-no-packing.json) | ultra | 1 | ncnn CPU (conversion validation only) | fp32 / 0 | 5,987.85（cold 1回） | 1.34 | 記録あり; Δ 5.17e-06; タグ差0 |
| [no-packing](ncnn_amd_20261003/no-packing.json) | ultra | 1 | ncnn | fp32 / 0 | 6,617.08 | 10.35 | 記録あり; Δ 0.0645; タグ差0 |
| [no-winograd](ncnn_amd_20261003/no-winograd.json) | ultra | 1 | ncnn | fp32 / 0 | 6,557.02 | 10.34 | 記録あり; Δ 0.16; タグ差4（タグ不一致） |
| [patched-no-subgroup](ncnn_amd_20261003/patched-no-subgroup.json) | ultra | 1 | ncnn | fp32 / 0 | 4,991.25 | 10.55 | 記録あり; Δ 0.193; タグ差3（タグ不一致） |
| [radv-fullsync](ncnn_amd_20261003/radv-fullsync.json) | ultra | 1 | ncnn | fp32 / 0 | 6,546.80 | 11.63 | 記録あり; Δ 0.301; タグ差6（タグ不一致） |
| [radv-syncshaders](ncnn_amd_20261003/radv-syncshaders.json) | ultra | 1 | ncnn | fp32 / 0 | 6,563.64 | 10.32 | 記録あり; Δ 2.74e-06; タグ差0 |
| [sync-default-fp32-b1](ncnn_amd_20261003/sync-default-fp32-b1.json) | ultra | 1 | ncnn | fp32 / 0 | 6,505.56 | 10.30 | 記録あり; Δ 2.74e-06; タグ差0 |
| [sync-fp32-b4](ncnn_amd_20261003/sync-fp32-b4.json) | ultra | 4 | ncnn | fp32 / 0 | 6,659.12 | 10.39 | 記録あり; Δ 2.74e-06; タグ差0 |
| [webgpu-b1](ncnn_amd_20261003/webgpu-b1.json) | ultra | 1 | webgpu | — / 0 | 4,964.40 | 11.74 | 記録あり; Δ 2.92e-06; タグ差0 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [candidate-fp32 / ultra / b1](ncnn_amd_20261003/candidate-fp32.json) | 1,223.98 | 未測定 | 487.55 | 未測定 | 未測定 | 未測定 | 2,673.36 | 93.50 |
| [conv1x1-direct / ultra / b1](ncnn_amd_20261003/conv1x1-direct.json) | 1,552.27 | 未測定 | 497.76 | 未測定 | 未測定 | 未測定 | 2,532.58 | 92.50 |
| [conv1x1-fp32-b1 / ultra / b1](ncnn_amd_20261003/conv1x1-fp32-b1.json) | 1,552.00 | 未測定 | 497.76 | 未測定 | 未測定 | 未測定 | 2,532.58 | 97.00 |
| [cpu-layernorm / ultra / b1](ncnn_amd_20261003/cpu-layernorm.json) | 241.17 | 未測定 | 500.03 | 未測定 | 未測定 | 未測定 | 2,880.02 | 94.50 |
| [cpu-reduction / ultra / b1](ncnn_amd_20261003/cpu-reduction.json) | 307.44 | 未測定 | 500.02 | 未測定 | 未測定 | 未測定 | 2,841.60 | 91.50 |
| [cpu-reshape / ultra / b1](ncnn_amd_20261003/cpu-reshape.json) | 615.28 | 未測定 | 500.03 | 未測定 | 未測定 | 未測定 | 3,012.05 | 81.50 |
| [fp32-b1 / ultra / b1](ncnn_amd_20261003/fp32-b1.json) | 225.63 | 未測定 | 503.32 | 未測定 | 未測定 | 未測定 | 2,982.07 | 98.00 |
| [native-no-shared-memory / ultra / b1](ncnn_amd_20261003/native-no-shared-memory.json) | 229.56 | 未測定 | 499.11 | 未測定 | 未測定 | 未測定 | 2,791.24 | 96.00 |
| [ncnn-cpu-check / ultra / b1](ncnn_amd_20261003/ncnn-cpu-check.json) | 3,300.60 | 未測定 | 362.06 | 未測定 | 未測定 | 未測定 | 140.00 | 3.00 |
| [ncnn-cpu-no-packing / ultra / b1](ncnn_amd_20261003/ncnn-cpu-no-packing.json) | 3,379.73 | 未測定 | 354.07 | 未測定 | 未測定 | 未測定 | 124.86 | 0.00 |
| [no-packing / ultra / b1](ncnn_amd_20261003/no-packing.json) | 220.18 | 未測定 | 498.27 | 未測定 | 未測定 | 未測定 | 2,999.08 | 96.50 |
| [no-winograd / ultra / b1](ncnn_amd_20261003/no-winograd.json) | 219.54 | 未測定 | 500.81 | 未測定 | 未測定 | 未測定 | 2,984.15 | 96.50 |
| [patched-no-subgroup / ultra / b1](ncnn_amd_20261003/patched-no-subgroup.json) | 1,616.49 | 未測定 | 491.25 | 未測定 | 未測定 | 未測定 | 2,817.12 | 91.50 |
| [radv-fullsync / ultra / b1](ncnn_amd_20261003/radv-fullsync.json) | 219.92 | 未測定 | 500.02 | 未測定 | 未測定 | 未測定 | 2,841.51 | 96.50 |
| [radv-syncshaders / ultra / b1](ncnn_amd_20261003/radv-syncshaders.json) | 235.25 | 未測定 | 500.02 | 未測定 | 未測定 | 未測定 | 2,841.51 | 96.50 |
| [sync-default-fp32-b1 / ultra / b1](ncnn_amd_20261003/sync-default-fp32-b1.json) | 219.16 | 未測定 | 500.02 | 未測定 | 未測定 | 未測定 | 2,841.51 | 98.17 |
| [sync-fp32-b4 / ultra / b4](ncnn_amd_20261003/sync-fp32-b4.json) | 219.07 | 未測定 | 500.02 | 未測定 | 未測定 | 未測定 | 2,841.51 | 98.75 |
| [webgpu-b1 / ultra / b1](ncnn_amd_20261003/webgpu-b1.json) | 200.34 | 未測定 | 501.94 | 未測定 | 未測定 | 未測定 | 3,507.48 | 98.45 |

## ncnn_amd_bazzite_20261003

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [fp32-b1](ncnn_amd_bazzite_20261003/fp32-b1.json) | ultra | 1 | ncnn | fp32 / 0 | 7,509.21 | 9.74 | 記録あり; Δ 2.62e-06; タグ差0 |
| [fp32-b4](ncnn_amd_bazzite_20261003/fp32-b4.json) | ultra | 4 | ncnn | fp32 / 0 | 7,558.89 | 10.26 | 記録あり; Δ 2.62e-06; タグ差0 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [fp32-b1 / ultra / b1](ncnn_amd_bazzite_20261003/fp32-b1.json) | 198.77 | 未測定 | 493.02 | 未測定 | 未測定 | 未測定 | 2,992.02 | 97.86 |
| [fp32-b4 / ultra / b4](ncnn_amd_bazzite_20261003/fp32-b4.json) | 198.97 | 未測定 | 493.02 | 未測定 | 未測定 | 未測定 | 2,992.02 | 98.63 |

## ncnn_intel_a13m_20261003

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [fp32-b1](ncnn_intel_a13m_20261003/fp32-b1.json) | ultra | 1 | ncnn | fp32 / 0 | 32,900.08 | 40.62 | 記録あり; Δ 3.28e-06; タグ差0 |
| [fp32-b4](ncnn_intel_a13m_20261003/fp32-b4.json) | ultra | 4 | ncnn | fp32 / 0 | 33,299.30 | 36.46 | 記録あり; Δ 3.28e-06; タグ差0 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [fp32-b1 / ultra / b1](ncnn_intel_a13m_20261003/fp32-b1.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [fp32-b4 / ultra / b4](ncnn_intel_a13m_20261003/fp32-b4.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |

## ncnn_local_20261002

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [conv1x1-cpu-check](ncnn_local_20261002/conv1x1-cpu-check.json) | ultra | 1 | ncnn CPU (conversion validation only) | fp32 / 0 | 135,230.40（cold 1回） | 4.00 | 記録あり; Δ 5.47e-06; タグ差0 |
| [cpu-b1](ncnn_local_20261002/cpu-b1.json) | ultra | 1 | cpu | — / 0 | 8,063.17 | 128.50 | 記録あり; 未照合 |
| [ncnn-cpu-conversion-check](ncnn_local_20261002/ncnn-cpu-conversion-check.json) | ultra | 1 | ncnn CPU (conversion validation only) | fp32 / 0 | 10,087.08（cold 1回） | 1.53 | 記録あり; Δ 0.000228; タグ差0 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [conv1x1-cpu-check / ultra / b1](ncnn_local_20261002/conv1x1-cpu-check.json) | 4,846.84 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [cpu-b1 / ultra / b1](ncnn_local_20261002/cpu-b1.json) | 3,578.36 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [ncnn-cpu-conversion-check / ultra / b1](ncnn_local_20261002/ncnn-cpu-conversion-check.json) | 3,299.70 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |

## ncnn_ps4_20261004

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [ultra-stream128-b1](ncnn_ps4_20261004/ultra-stream128-b1.json) | ultra | 1 | ncnn | fp32 / 128 | 87,825.23 | 296.66 | 記録あり; Δ 2.74e-06; タグ差0 |
| [ultra-stream128-b4](ncnn_ps4_20261004/ultra-stream128-b4.json) | ultra | 4 | ncnn | fp32 / 128 | 80,302.54 | 90.41 | 記録あり; Δ 2.74e-06; タグ差0 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [ultra-stream128-b1 / ultra / b1](ncnn_ps4_20261004/ultra-stream128-b1.json) | 274.14 | 未測定 | 690.33 | 33.71 | 未測定 | 未測定 | 32.61 | 未測定 |
| [ultra-stream128-b4 / ultra / b4](ncnn_ps4_20261004/ultra-stream128-b4.json) | 229.41 | 未測定 | 690.33 | 33.71 | 未測定 | 未測定 | 36.61 | 未測定 |

## ncnn_switch_20261003

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [ultra-stream128-b1](ncnn_switch_20261003/ultra-stream128-b1.json) | ultra | 1 | ncnn | fp32 / 128 | 75,004.67 | 217.71 | 記録あり; Δ 3.34e-06; タグ差0 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [ultra-stream128-b1 / ultra / b1](ncnn_switch_20261003/ultra-stream128-b1.json) | 544.77 | 13.65 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |

## ncnn_windows_20261003

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [cuda-b1](ncnn_windows_20261003/cuda-b1.json) | ultra | 1 | cuda | — / 0 | 490.67 | 6.29 | 記録あり; Δ 3.04e-06; タグ差0 |
| [cuda-b4](ncnn_windows_20261003/cuda-b4.json) | ultra | 4 | cuda | — / 0 | 472.30 | 6.23 | 記録あり; Δ 3.34e-06; タグ差0 |
| [rtx-fp32-b1](ncnn_windows_20261003/rtx-fp32-b1.json) | ultra | 1 | ncnn | fp32 / 0 | 1,131.62 | 7.69 | 記録あり; Δ 3.58e-06; タグ差0 |
| [rtx-fp32-b4](ncnn_windows_20261003/rtx-fp32-b4.json) | ultra | 4 | ncnn | fp32 / 0 | 1,128.64 | 12.27 | 記録あり; Δ 3.58e-06; タグ差0 |
| [rtx-memory-check](ncnn_windows_20261003/rtx-memory-check.json) | ultra | 1 | ncnn | fp32 / 0 | 1,138.39 | 5.13 | 記録あり; Δ 3.58e-06; タグ差0 |
| [tensorrt-b1](ncnn_windows_20261003/tensorrt-b1.json) | ultra | 1 | tensorrt | — / 0 | 377.54 | 12.95 | 記録あり; Δ 4.71e-06; タグ差0 |
| [tensorrt-b4](ncnn_windows_20261003/tensorrt-b4.json) | ultra | 4 | tensorrt | — / 0 | 359.47 | 8.65 | 記録あり; Δ 4.71e-06; タグ差0 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [cuda-b1 / ultra / b1](ncnn_windows_20261003/cuda-b1.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 4,958.00 | 未測定 | 未測定 |
| [cuda-b4 / ultra / b4](ncnn_windows_20261003/cuda-b4.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 7,873.00 | 未測定 | 未測定 |
| [rtx-fp32-b1 / ultra / b1](ncnn_windows_20261003/rtx-fp32-b1.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [rtx-fp32-b4 / ultra / b4](ncnn_windows_20261003/rtx-fp32-b4.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 3,624.00 | 未測定 | 未測定 |
| [rtx-memory-check / ultra / b1](ncnn_windows_20261003/rtx-memory-check.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 3,732.00 | 未測定 | 未測定 |
| [tensorrt-b1 / ultra / b1](ncnn_windows_20261003/tensorrt-b1.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 4,605.00 | 未測定 | 未測定 |
| [tensorrt-b4 / ultra / b4](ncnn_windows_20261003/tensorrt-b4.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 4,621.00 | 未測定 | 未測定 |

## nvidia_linux_20260923

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [balanced-b1-cpu](nvidia_linux_20260923/balanced-b1-cpu.json) | balanced | 1 | cpu | — / 0 | 593.10 | 1.29 | ok; 未照合 |
| [balanced-b1-cuda](nvidia_linux_20260923/balanced-b1-cuda.json) | balanced | 1 | cuda | — / 0 | 58.16 | 1.38 | ok; 未照合 |
| [balanced-b1-tensorrt](nvidia_linux_20260923/balanced-b1-tensorrt.json) | balanced | 1 | tensorrt | — / 0 | 46.06 | 3.23 | ok; 未照合 |
| [balanced-b1-webgpu](nvidia_linux_20260923/balanced-b1-webgpu.json) | balanced | 1 | webgpu | — / 0 | 120.08 | 1.79 | ok; 未照合 |
| [balanced-b4-cpu](nvidia_linux_20260923/balanced-b4-cpu.json) | balanced | 4 | cpu | — / 0 | 702.79 | 1.30 | ok; 未照合 |
| [balanced-b4-cuda](nvidia_linux_20260923/balanced-b4-cuda.json) | balanced | 4 | cuda | — / 0 | 57.97 | 1.42 | ok; 未照合 |
| [balanced-b4-tensorrt](nvidia_linux_20260923/balanced-b4-tensorrt.json) | balanced | 4 | tensorrt | — / 0 | 43.67 | 3.22 | ok; 未照合 |
| [balanced-b4-webgpu](nvidia_linux_20260923/balanced-b4-webgpu.json) | balanced | 4 | webgpu | — / 0 | 110.63 | 1.78 | ok; 未照合 |
| [wd14_v3-b1-cpu](nvidia_linux_20260923/wd14_v3-b1-cpu.json) | wd14_v3 | 1 | cpu | — / 0 | 728.89 | 1.46 | ok; 未照合 |
| [wd14_v3-b1-cuda](nvidia_linux_20260923/wd14_v3-b1-cuda.json) | wd14_v3 | 1 | cuda | — / 0 | 66.89 | 1.63 | ok; 未照合 |
| [wd14_v3-b1-tensorrt](nvidia_linux_20260923/wd14_v3-b1-tensorrt.json) | wd14_v3 | 1 | tensorrt | — / 0 | 42.37 | 2.79 | ok; 未照合 |
| [wd14_v3-b1-webgpu](nvidia_linux_20260923/wd14_v3-b1-webgpu.json) | wd14_v3 | 1 | webgpu | — / 0 | 124.20 | 2.05 | ok; 未照合 |
| [wd14_v3-b4-cpu](nvidia_linux_20260923/wd14_v3-b4-cpu.json) | wd14_v3 | 4 | cpu | — / 0 | 767.28 | 1.47 | ok; 未照合 |
| [wd14_v3-b4-cuda](nvidia_linux_20260923/wd14_v3-b4-cuda.json) | wd14_v3 | 4 | cuda | — / 0 | 62.31 | 1.66 | ok; 未照合 |
| [wd14_v3-b4-tensorrt](nvidia_linux_20260923/wd14_v3-b4-tensorrt.json) | wd14_v3 | 4 | tensorrt | — / 0 | 39.49 | 3.10 | ok; 未照合 |
| [wd14_v3-b4-webgpu](nvidia_linux_20260923/wd14_v3-b4-webgpu.json) | wd14_v3 | 4 | webgpu | — / 0 | 115.29 | 1.90 | ok; 未照合 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [balanced-b1-cpu / balanced / b1](nvidia_linux_20260923/balanced-b1-cpu.json) | 1,040.49 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-cuda / balanced / b1](nvidia_linux_20260923/balanced-b1-cuda.json) | 1,222.22 | 未測定 | 1,180.00 | 未測定 | 1,146.00 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-tensorrt / balanced / b1](nvidia_linux_20260923/balanced-b1-tensorrt.json) | 3,150.19 | 未測定 | 988.00 | 未測定 | 954.00 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-webgpu / balanced / b1](nvidia_linux_20260923/balanced-b1-webgpu.json) | 658.32 | 未測定 | 721.00 | 未測定 | 687.00 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-cpu / balanced / b4](nvidia_linux_20260923/balanced-b4-cpu.json) | 1,360.75 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-cuda / balanced / b4](nvidia_linux_20260923/balanced-b4-cuda.json) | 1,211.48 | 未測定 | 1,180.00 | 未測定 | 1,146.00 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-tensorrt / balanced / b4](nvidia_linux_20260923/balanced-b4-tensorrt.json) | 3,188.66 | 未測定 | 1,238.00 | 未測定 | 1,204.00 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-webgpu / balanced / b4](nvidia_linux_20260923/balanced-b4-webgpu.json) | 669.47 | 未測定 | 1,137.00 | 未測定 | 1,103.00 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-cpu / wd14_v3 / b1](nvidia_linux_20260923/wd14_v3-b1-cpu.json) | 1,236.57 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-cuda / wd14_v3 / b1](nvidia_linux_20260923/wd14_v3-b1-cuda.json) | 1,258.54 | 未測定 | 1,180.00 | 未測定 | 1,147.00 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-tensorrt / wd14_v3 / b1](nvidia_linux_20260923/wd14_v3-b1-tensorrt.json) | 3,345.49 | 未測定 | 942.00 | 未測定 | 909.00 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-webgpu / wd14_v3 / b1](nvidia_linux_20260923/wd14_v3-b1-webgpu.json) | 690.62 | 未測定 | 874.00 | 未測定 | 841.00 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-cpu / wd14_v3 / b4](nvidia_linux_20260923/wd14_v3-b4-cpu.json) | 1,863.11 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-cuda / wd14_v3 / b4](nvidia_linux_20260923/wd14_v3-b4-cuda.json) | 1,260.76 | 未測定 | 1,180.00 | 未測定 | 1,147.00 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-tensorrt / wd14_v3 / b4](nvidia_linux_20260923/wd14_v3-b4-tensorrt.json) | 3,331.25 | 未測定 | 1,356.00 | 未測定 | 1,323.00 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-webgpu / wd14_v3 / b4](nvidia_linux_20260923/wd14_v3-b4-webgpu.json) | 662.17 | 未測定 | 1,865.00 | 未測定 | 1,832.00 | 未測定 | 未測定 | 未測定 |

## nvidia_linux_20261004

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [cuda-b1](nvidia_linux_20261004/cuda-b1.json) | ultra | 1 | cuda | — / 0 | 402.75 | 8.10 | 記録あり; Δ 3.04e-06; タグ差0 |
| [cuda-b4](nvidia_linux_20261004/cuda-b4.json) | ultra | 4 | cuda | — / 0 | 397.56 | 3.68 | 記録あり; Δ 3.34e-06; タグ差0 |
| [ncnn-b1](nvidia_linux_20261004/ncnn-b1.json) | ultra | 1 | ncnn | fp32 / 0 | 1,015.81 | 9.33 | 記録あり; Δ 3.58e-06; タグ差0 |
| [ncnn-b4](nvidia_linux_20261004/ncnn-b4.json) | ultra | 4 | ncnn | fp32 / 0 | 1,019.59 | 5.28 | 記録あり; Δ 3.58e-06; タグ差0 |
| [tensorrt-b1](nvidia_linux_20261004/tensorrt-b1.json) | ultra | 1 | tensorrt | — / 0 | 295.60 | 6.26 | 記録あり; Δ 3.78e-06; タグ差0 |
| [tensorrt-b4](nvidia_linux_20261004/tensorrt-b4.json) | ultra | 4 | tensorrt | — / 0 | 296.39 | 4.40 | 記録あり; Δ 4.65e-06; タグ差0 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [cuda-b1 / ultra / b1](nvidia_linux_20261004/cuda-b1.json) | 976.75 | 未測定 | 未測定 | 未測定 | 未測定 | 4,220.00 (51.51%) | 未測定 | 未測定 |
| [cuda-b4 / ultra / b4](nvidia_linux_20261004/cuda-b4.json) | 1,103.75 | 未測定 | 未測定 | 未測定 | 未測定 | 5,244.00 (64.01%) | 未測定 | 未測定 |
| [ncnn-b1 / ultra / b1](nvidia_linux_20261004/ncnn-b1.json) | 372.70 | 未測定 | 未測定 | 未測定 | 未測定 | 3,025.00 (36.93%) | 未測定 | 未測定 |
| [ncnn-b4 / ultra / b4](nvidia_linux_20261004/ncnn-b4.json) | 360.63 | 未測定 | 未測定 | 未測定 | 未測定 | 3,025.00 (36.93%) | 未測定 | 未測定 |
| [tensorrt-b1 / ultra / b1](nvidia_linux_20261004/tensorrt-b1.json) | 3,566.25 | 未測定 | 未測定 | 未測定 | 未測定 | 3,408.00 (41.60%) | 未測定 | 未測定 |
| [tensorrt-b4 / ultra / b4](nvidia_linux_20261004/tensorrt-b4.json) | 7,237.22 | 未測定 | 未測定 | 未測定 | 未測定 | 4,238.00 (51.73%) | 未測定 | 未測定 |

## nvidia_windows_20260923

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [balanced-b1-cpu](nvidia_windows_20260923/balanced-b1-cpu.json) | balanced | 1 | cpu | — / 0 | 944.55 | 1.71 | ok; 未照合 |
| [balanced-b1-cuda](nvidia_windows_20260923/balanced-b1-cuda.json) | balanced | 1 | cuda | — / 0 | 76.75 | 1.51 | ok; 未照合 |
| [balanced-b1-directml](nvidia_windows_20260923/balanced-b1-directml.json) | balanced | 1 | directml | — / 0 | 未測定 | 未測定 | failed; 未照合 |
| [balanced-b1-tensorrt](nvidia_windows_20260923/balanced-b1-tensorrt.json) | balanced | 1 | tensorrt | — / 0 | 60.90 | 3.47 | ok; 未照合 |
| [balanced-b1-webgpu](nvidia_windows_20260923/balanced-b1-webgpu.json) | balanced | 1 | webgpu | — / 0 | 197.99 | 2.44 | ok; 未照合 |
| [balanced-b4-cpu](nvidia_windows_20260923/balanced-b4-cpu.json) | balanced | 4 | cpu | — / 0 | 929.54 | 1.43 | ok; 未照合 |
| [balanced-b4-cuda](nvidia_windows_20260923/balanced-b4-cuda.json) | balanced | 4 | cuda | — / 0 | 70.93 | 1.50 | ok; 未照合 |
| [balanced-b4-directml](nvidia_windows_20260923/balanced-b4-directml.json) | balanced | 4 | directml | — / 0 | 未測定 | 未測定 | failed; 未照合 |
| [balanced-b4-tensorrt](nvidia_windows_20260923/balanced-b4-tensorrt.json) | balanced | 4 | tensorrt | — / 0 | 55.30 | 3.29 | ok; 未照合 |
| [balanced-b4-webgpu](nvidia_windows_20260923/balanced-b4-webgpu.json) | balanced | 4 | webgpu | — / 0 | 181.62 | 2.49 | ok; 未照合 |
| [wd14_v3-b1-cpu](nvidia_windows_20260923/wd14_v3-b1-cpu.json) | wd14_v3 | 1 | cpu | — / 0 | 1,052.67 | 2.05 | ok; 未照合 |
| [wd14_v3-b1-cuda](nvidia_windows_20260923/wd14_v3-b1-cuda.json) | wd14_v3 | 1 | cuda | — / 0 | 87.53 | 2.12 | ok; 未照合 |
| [wd14_v3-b1-directml](nvidia_windows_20260923/wd14_v3-b1-directml.json) | wd14_v3 | 1 | directml | — / 0 | 208.45 | 2.38 | ok; 未照合 |
| [wd14_v3-b1-tensorrt](nvidia_windows_20260923/wd14_v3-b1-tensorrt.json) | wd14_v3 | 1 | tensorrt | — / 0 | 53.72 | 3.73 | ok; 未照合 |
| [wd14_v3-b1-webgpu](nvidia_windows_20260923/wd14_v3-b1-webgpu.json) | wd14_v3 | 1 | webgpu | — / 0 | 214.92 | 2.56 | ok; 未照合 |
| [wd14_v3-b4-cpu](nvidia_windows_20260923/wd14_v3-b4-cpu.json) | wd14_v3 | 4 | cpu | — / 0 | 1,032.19 | 2.15 | ok; 未照合 |
| [wd14_v3-b4-cuda](nvidia_windows_20260923/wd14_v3-b4-cuda.json) | wd14_v3 | 4 | cuda | — / 0 | 74.39 | 2.13 | ok; 未照合 |
| [wd14_v3-b4-directml](nvidia_windows_20260923/wd14_v3-b4-directml.json) | wd14_v3 | 4 | directml | — / 0 | 128.72 | 2.55 | ok; 未照合 |
| [wd14_v3-b4-tensorrt](nvidia_windows_20260923/wd14_v3-b4-tensorrt.json) | wd14_v3 | 4 | tensorrt | — / 0 | 46.88 | 3.00 | ok; 未照合 |
| [wd14_v3-b4-webgpu](nvidia_windows_20260923/wd14_v3-b4-webgpu.json) | wd14_v3 | 4 | webgpu | — / 0 | 194.55 | 2.68 | ok; 未照合 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [balanced-b1-cpu / balanced / b1](nvidia_windows_20260923/balanced-b1-cpu.json) | 1,045.88 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-cuda / balanced / b1](nvidia_windows_20260923/balanced-b1-cuda.json) | 1,024.37 | 未測定 | 2,065.00 | 未測定 | 1,131.00 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-directml / balanced / b1](nvidia_windows_20260923/balanced-b1-directml.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-tensorrt / balanced / b1](nvidia_windows_20260923/balanced-b1-tensorrt.json) | 3,075.07 | 未測定 | 1,919.00 | 未測定 | 985.00 | 未測定 | 未測定 | 未測定 |
| [balanced-b1-webgpu / balanced / b1](nvidia_windows_20260923/balanced-b1-webgpu.json) | 671.45 | 未測定 | 1,643.00 | 未測定 | 709.00 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-cpu / balanced / b4](nvidia_windows_20260923/balanced-b4-cpu.json) | 1,368.91 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-cuda / balanced / b4](nvidia_windows_20260923/balanced-b4-cuda.json) | 1,017.29 | 未測定 | 2,129.00 | 未測定 | 1,195.00 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-directml / balanced / b4](nvidia_windows_20260923/balanced-b4-directml.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-tensorrt / balanced / b4](nvidia_windows_20260923/balanced-b4-tensorrt.json) | 3,096.50 | 未測定 | 2,171.00 | 未測定 | 1,237.00 | 未測定 | 未測定 | 未測定 |
| [balanced-b4-webgpu / balanced / b4](nvidia_windows_20260923/balanced-b4-webgpu.json) | 583.24 | 未測定 | 2,067.00 | 未測定 | 1,133.00 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-cpu / wd14_v3 / b1](nvidia_windows_20260923/wd14_v3-b1-cpu.json) | 1,250.84 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-cuda / wd14_v3 / b1](nvidia_windows_20260923/wd14_v3-b1-cuda.json) | 1,048.73 | 未測定 | 2,065.00 | 未測定 | 1,131.00 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-directml / wd14_v3 / b1](nvidia_windows_20260923/wd14_v3-b1-directml.json) | 870.01 | 未測定 | 1,800.00 | 未測定 | 866.00 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-tensorrt / wd14_v3 / b1](nvidia_windows_20260923/wd14_v3-b1-tensorrt.json) | 2,870.95 | 未測定 | 1,873.00 | 未測定 | 939.00 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b1-webgpu / wd14_v3 / b1](nvidia_windows_20260923/wd14_v3-b1-webgpu.json) | 766.92 | 未測定 | 1,802.00 | 未測定 | 868.00 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-cpu / wd14_v3 / b4](nvidia_windows_20260923/wd14_v3-b4-cpu.json) | 1,879.95 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-cuda / wd14_v3 / b4](nvidia_windows_20260923/wd14_v3-b4-cuda.json) | 1,058.55 | 未測定 | 2,065.00 | 未測定 | 1,131.00 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-directml / wd14_v3 / b4](nvidia_windows_20260923/wd14_v3-b4-directml.json) | 862.66 | 未測定 | 2,902.00 | 未測定 | 1,968.00 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-tensorrt / wd14_v3 / b4](nvidia_windows_20260923/wd14_v3-b4-tensorrt.json) | 2,929.23 | 未測定 | 2,287.00 | 未測定 | 1,353.00 | 未測定 | 未測定 | 未測定 |
| [wd14_v3-b4-webgpu / wd14_v3 / b4](nvidia_windows_20260923/wd14_v3-b4-webgpu.json) | 783.65 | 未測定 | 2,797.00 | 未測定 | 1,863.00 | 未測定 | 未測定 | 未測定 |

## nvidia_windows_summary

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [nvidia_windows_summary](nvidia_windows_summary.json) | balanced | 1 | cuda | — / 0 | 78.83 | 未測定 | 記録あり; 未照合 |
| [nvidia_windows_summary](nvidia_windows_summary.json) | balanced | 1 | tensorrt | — / 0 | 61.16 | 未測定 | 記録あり; 未照合 |
| [nvidia_windows_summary](nvidia_windows_summary.json) | balanced | 1 | webgpu | — / 0 | 202.41 | 未測定 | 記録あり; 未照合 |
| [nvidia_windows_summary](nvidia_windows_summary.json) | balanced | 4 | cuda | — / 0 | 71.94 | 未測定 | 記録あり; 未照合 |
| [nvidia_windows_summary](nvidia_windows_summary.json) | balanced | 4 | tensorrt | — / 0 | 55.62 | 未測定 | 記録あり; 未照合 |
| [nvidia_windows_summary](nvidia_windows_summary.json) | balanced | 4 | webgpu | — / 0 | 182.17 | 未測定 | 記録あり; 未照合 |
| [nvidia_windows_summary](nvidia_windows_summary.json) | wd14_v3 | 1 | cuda | — / 0 | 85.45 | 未測定 | 記録あり; 未照合 |
| [nvidia_windows_summary](nvidia_windows_summary.json) | wd14_v3 | 1 | tensorrt | — / 0 | 53.90 | 未測定 | 記録あり; 未照合 |
| [nvidia_windows_summary](nvidia_windows_summary.json) | wd14_v3 | 1 | webgpu | — / 0 | 215.31 | 未測定 | 記録あり; 未照合 |
| [nvidia_windows_summary](nvidia_windows_summary.json) | wd14_v3 | 4 | cuda | — / 0 | 74.80 | 未測定 | 記録あり; 未照合 |
| [nvidia_windows_summary](nvidia_windows_summary.json) | wd14_v3 | 4 | tensorrt | — / 0 | 46.93 | 未測定 | 記録あり; 未照合 |
| [nvidia_windows_summary](nvidia_windows_summary.json) | wd14_v3 | 4 | webgpu | — / 0 | 195.60 | 未測定 | 記録あり; 未照合 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [nvidia_windows_summary / balanced / b1](nvidia_windows_summary.json) | 1,023.07 | 未測定 | 未測定 | 未測定 | 1,129.00 | 未測定 | 未測定 | 未測定 |
| [nvidia_windows_summary / balanced / b1](nvidia_windows_summary.json) | 3,077.52 | 未測定 | 未測定 | 未測定 | 985.00 | 未測定 | 未測定 | 未測定 |
| [nvidia_windows_summary / balanced / b1](nvidia_windows_summary.json) | 689.48 | 未測定 | 未測定 | 未測定 | 714.00 | 未測定 | 未測定 | 未測定 |
| [nvidia_windows_summary / balanced / b4](nvidia_windows_summary.json) | 1,014.39 | 未測定 | 未測定 | 未測定 | 1,142.00 | 未測定 | 未測定 | 未測定 |
| [nvidia_windows_summary / balanced / b4](nvidia_windows_summary.json) | 3,077.44 | 未測定 | 未測定 | 未測定 | 1,228.00 | 未測定 | 未測定 | 未測定 |
| [nvidia_windows_summary / balanced / b4](nvidia_windows_summary.json) | 729.25 | 未測定 | 未測定 | 未測定 | 1,133.00 | 未測定 | 未測定 | 未測定 |
| [nvidia_windows_summary / wd14_v3 / b1](nvidia_windows_summary.json) | 1,049.34 | 未測定 | 未測定 | 未測定 | 1,133.00 | 未測定 | 未測定 | 未測定 |
| [nvidia_windows_summary / wd14_v3 / b1](nvidia_windows_summary.json) | 2,861.39 | 未測定 | 未測定 | 未測定 | 938.00 | 未測定 | 未測定 | 未測定 |
| [nvidia_windows_summary / wd14_v3 / b1](nvidia_windows_summary.json) | 743.88 | 未測定 | 未測定 | 未測定 | 867.00 | 未測定 | 未測定 | 未測定 |
| [nvidia_windows_summary / wd14_v3 / b4](nvidia_windows_summary.json) | 1,059.43 | 未測定 | 未測定 | 未測定 | 1,127.00 | 未測定 | 未測定 | 未測定 |
| [nvidia_windows_summary / wd14_v3 / b4](nvidia_windows_summary.json) | 2,910.82 | 未測定 | 未測定 | 未測定 | 1,353.00 | 未測定 | 未測定 | 未測定 |
| [nvidia_windows_summary / wd14_v3 / b4](nvidia_windows_summary.json) | 780.28 | 未測定 | 未測定 | 未測定 | 1,874.00 | 未測定 | 未測定 | 未測定 |

## openvino_intel_a13m_20261004

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [fp32-b1](openvino_intel_a13m_20261004/fp32-b1.json) | ultra | 1 | intel | — / 0 | 1,599.69 | 90.22 | 記録あり; Δ 5.66e-06; タグ差0 |
| [fp32-b4](openvino_intel_a13m_20261004/fp32-b4.json) | ultra | 4 | intel | — / 0 | 1,542.60 | 16.97 | 記録あり; Δ 5.72e-06; タグ差0 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [fp32-b1 / ultra / b1](openvino_intel_a13m_20261004/fp32-b1.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| [fp32-b4 / ultra / b4](openvino_intel_a13m_20261004/fp32-b4.json) | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |

## directml_legacy

### 速度と精度

| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |
|---|---|---:|---|---|---:|---:|---|
| [legacy_results](legacy_results.md) | lightweight | 4 | DirectML | — / 0 | 120.30 | 未測定 | 旧平均3回; 未照合 |
| [legacy_results](legacy_results.md) | balanced | 4 | DirectML | — / 0 | 458.80 | 未測定 | 旧平均3回; 未照合 |
| [legacy_results](legacy_results.md) | high | 4 | DirectML | — / 0 | 1,273.90 | 未測定 | 旧平均3回; 未照合 |
| [legacy_results](legacy_results.md) | ultra | 4 | DirectML | — / 0 | 1,735.30 | 未測定 | 旧平均3回; 未照合 |
| [legacy_results](legacy_results.md) | wd14_v3 | 4 | DirectML | — / 0 | 442.30 | 未測定 | 旧平均3回; 未照合 |

### メモリとGPU稼働率

| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [legacy_results / lightweight / b4](legacy_results.md) | 未測定 | 未測定 | 1,614.00 | 19.70 | 771.00 | 未測定 | 未測定 | 未測定 |
| [legacy_results / balanced / b4](legacy_results.md) | 未測定 | 未測定 | 2,051.00 | 25.04 | 1,199.00 | 未測定 | 未測定 | 未測定 |
| [legacy_results / high / b4](legacy_results.md) | 未測定 | 未測定 | 3,618.00 | 44.17 | 2,771.00 | 未測定 | 未測定 | 未測定 |
| [legacy_results / ultra / b4](legacy_results.md) | 未測定 | 未測定 | 6,583.00 | 80.36 | 5,743.00 | 未測定 | 未測定 | 未測定 |
| [legacy_results / wd14_v3 / b4](legacy_results.md) | 未測定 | 未測定 | 2,973.00 | 36.29 | 1,886.00 | 未測定 | 未測定 | 未測定 |

## 同じ測定組でのモデル差

balancedを基準とした対象モデル÷balanced。時間比が1より大きければ遅く、メモリ比が1より大きければ多く使用する。実行方式・batchが同じ保存値だけを比較する。VRAMは同じ指標同士（ピーク優先、なければ増分）。モデルごとに画像サイズ・処理が異なるため品質の比較ではない。最新ultraと9月の他モデルは測定日・入力・精度設定が異なり、直接の倍率表を作らない。

| 測定組 | 実行方式 | batch | 対象モデル | 時間比 | RSS比 | VRAM比・指標 | 出典 |
|---|---|---:|---|---:|---:|---|---|
| amd_barcelo_4gb | WebGpuExecutionProvider+CPUExecutionProvider | 1 | lightweight | 0.12× | 0.42× | 0.68× / ピーク | [対象](amd_barcelo_4gb.json) / [balanced](amd_barcelo_4gb.json) |
| amd_barcelo_4gb | WebGpuExecutionProvider+CPUExecutionProvider | 1 | high | 5.89× | 2.35× | 1.92× / ピーク | [対象](amd_barcelo_4gb.json) / [balanced](amd_barcelo_4gb.json) |
| amd_barcelo_4gb | WebGpuExecutionProvider+CPUExecutionProvider | 1 | ultra | 8.41× | 0.28× | 3.52× / ピーク | [対象](amd_barcelo_4gb.json) / [balanced](amd_barcelo_4gb.json) |
| amd_barcelo_4gb | WebGpuExecutionProvider+CPUExecutionProvider | 1 | wd14_v3 | 1.30× | 0.69× | 1.28× / ピーク | [対象](amd_barcelo_4gb.json) / [balanced](amd_barcelo_4gb.json) |
| amd_barcelo_512mb | WebGpuExecutionProvider+CPUExecutionProvider | 1 | wd14_v3 | 1.20× | 0.94× | 1.00× / ピーク | [対象](amd_barcelo_512mb.json) / [balanced](amd_barcelo_512mb.json) |
| amd_barcelo_512mb | WebGpuExecutionProvider+CPUExecutionProvider | 1 | lightweight | 0.12× | 0.41× | 0.98× / ピーク | [対象](amd_barcelo_512mb.json) / [balanced](amd_barcelo_512mb.json) |
| amd_barcelo_512mb | WebGpuExecutionProvider+CPUExecutionProvider | 1 | high | 5.72× | 2.34× | 1.00× / ピーク | [対象](amd_barcelo_512mb.json) / [balanced](amd_barcelo_512mb.json) |
| amd_barcelo_512mb | WebGpuExecutionProvider+CPUExecutionProvider | 1 | ultra | 7.62× | 2.23× | 1.01× / ピーク | [対象](amd_barcelo_512mb.json) / [balanced](amd_barcelo_512mb.json) |
| amd_windows_20260923 | cpu | 1 | wd14_v3 | 1.26× | 1.19× | — / 増分 | [対象](amd_windows_20260923/wd14_v3-b1-cpu.json) / [balanced](amd_windows_20260923/balanced-b1-cpu.json) |
| amd_windows_20260923 | directml | 1 | wd14_v3 | 1.13× | 1.17× | — / 増分 | [対象](amd_windows_20260923/wd14_v3-b1-directml.json) / [balanced](amd_windows_20260923/balanced-b1-directml.json) |
| amd_windows_20260923 | webgpu | 1 | wd14_v3 | 1.34× | 1.09× | — / 増分 | [対象](amd_windows_20260923/wd14_v3-b1-webgpu.json) / [balanced](amd_windows_20260923/balanced-b1-webgpu.json) |
| amd_windows_20260923 | cpu | 4 | wd14_v3 | 1.08× | 1.37× | — / 増分 | [対象](amd_windows_20260923/wd14_v3-b4-cpu.json) / [balanced](amd_windows_20260923/balanced-b4-cpu.json) |
| amd_windows_20260923 | directml | 4 | wd14_v3 | 1.09× | 1.05× | — / 増分 | [対象](amd_windows_20260923/wd14_v3-b4-directml.json) / [balanced](amd_windows_20260923/balanced-b4-directml.json) |
| amd_windows_20260923 | webgpu | 4 | wd14_v3 | 1.25× | 1.18× | — / 増分 | [対象](amd_windows_20260923/wd14_v3-b4-webgpu.json) / [balanced](amd_windows_20260923/balanced-b4-webgpu.json) |
| intel2_linux_20260923 | cpu | 1 | wd14_v3 | 0.97× | 1.16× | — / 増分 | [対象](intel2_linux_20260923/wd14_v3-b1-cpu.json) / [balanced](intel2_linux_20260923/balanced-b1-cpu.json) |
| intel2_linux_20260923 | intel | 1 | wd14_v3 | 0.59× | 0.86× | — / 増分 | [対象](intel2_linux_20260923/wd14_v3-b1-intel.json) / [balanced](intel2_linux_20260923/balanced-b1-intel.json) |
| intel2_linux_20260923 | webgpu | 1 | wd14_v3 | 1.12× | 0.95× | — / 増分 | [対象](intel2_linux_20260923/wd14_v3-b1-webgpu.json) / [balanced](intel2_linux_20260923/balanced-b1-webgpu.json) |
| intel2_linux_20260923 | cpu | 4 | wd14_v3 | 1.06× | 1.37× | — / 増分 | [対象](intel2_linux_20260923/wd14_v3-b4-cpu.json) / [balanced](intel2_linux_20260923/balanced-b4-cpu.json) |
| intel2_linux_20260923 | intel | 4 | wd14_v3 | 0.62× | 1.09× | — / 増分 | [対象](intel2_linux_20260923/wd14_v3-b4-intel.json) / [balanced](intel2_linux_20260923/balanced-b4-intel.json) |
| intel2_linux_20260923 | webgpu | 4 | wd14_v3 | 1.14× | 0.99× | — / 増分 | [対象](intel2_linux_20260923/wd14_v3-b4-webgpu.json) / [balanced](intel2_linux_20260923/balanced-b4-webgpu.json) |
| intel3_linux_20260923 | cpu | 1 | wd14_v3 | 1.18× | 1.16× | — / 増分 | [対象](intel3_linux_20260923/wd14_v3-b1-cpu.json) / [balanced](intel3_linux_20260923/balanced-b1-cpu.json) |
| intel3_linux_20260923 | intel | 1 | wd14_v3 | 0.64× | 1.10× | — / 増分 | [対象](intel3_linux_20260923/wd14_v3-b1-intel.json) / [balanced](intel3_linux_20260923/balanced-b1-intel.json) |
| intel3_linux_20260923 | webgpu | 1 | wd14_v3 | 1.09× | 1.02× | — / 増分 | [対象](intel3_linux_20260923/wd14_v3-b1-webgpu.json) / [balanced](intel3_linux_20260923/balanced-b1-webgpu.json) |
| intel3_linux_20260923 | cpu | 4 | wd14_v3 | 1.07× | 1.37× | — / 増分 | [対象](intel3_linux_20260923/wd14_v3-b4-cpu.json) / [balanced](intel3_linux_20260923/balanced-b4-cpu.json) |
| intel3_linux_20260923 | intel | 4 | wd14_v3 | 0.66× | 1.10× | — / 増分 | [対象](intel3_linux_20260923/wd14_v3-b4-intel.json) / [balanced](intel3_linux_20260923/balanced-b4-intel.json) |
| intel3_linux_20260923 | webgpu | 4 | wd14_v3 | 1.08× | 1.02× | — / 増分 | [対象](intel3_linux_20260923/wd14_v3-b4-webgpu.json) / [balanced](intel3_linux_20260923/balanced-b4-webgpu.json) |
| intel_linux_20260923 | cpu | 1 | wd14_v3 | 0.85× | 1.15× | — / 増分 | [対象](intel_linux_20260923/wd14_v3-b1-cpu.json) / [balanced](intel_linux_20260923/balanced-b1-cpu.json) |
| intel_linux_20260923 | intel | 1 | wd14_v3 | 0.64× | 1.08× | — / 増分 | [対象](intel_linux_20260923/wd14_v3-b1-intel.json) / [balanced](intel_linux_20260923/balanced-b1-intel.json) |
| intel_linux_20260923 | webgpu | 1 | wd14_v3 | 1.03× | 0.93× | — / 増分 | [対象](intel_linux_20260923/wd14_v3-b1-webgpu.json) / [balanced](intel_linux_20260923/balanced-b1-webgpu.json) |
| intel_linux_20260923 | cpu | 4 | wd14_v3 | 0.85× | 1.37× | — / 増分 | [対象](intel_linux_20260923/wd14_v3-b4-cpu.json) / [balanced](intel_linux_20260923/balanced-b4-cpu.json) |
| intel_linux_20260923 | intel | 4 | wd14_v3 | 0.66× | 1.10× | — / 増分 | [対象](intel_linux_20260923/wd14_v3-b4-intel.json) / [balanced](intel_linux_20260923/balanced-b4-intel.json) |
| intel_linux_20260923 | webgpu | 4 | wd14_v3 | 1.05× | 0.95× | — / 増分 | [対象](intel_linux_20260923/wd14_v3-b4-webgpu.json) / [balanced](intel_linux_20260923/balanced-b4-webgpu.json) |
| intel_windows_20260923 | cpu | 1 | wd14_v3 | 1.14× | 1.20× | — / 増分 | [対象](intel_windows_20260923/wd14_v3-b1-cpu.json) / [balanced](intel_windows_20260923/balanced-b1-cpu.json) |
| intel_windows_20260923 | webgpu | 1 | wd14_v3 | 1.09× | 1.01× | — / 増分 | [対象](intel_windows_20260923/wd14_v3-b1-webgpu.json) / [balanced](intel_windows_20260923/balanced-b1-webgpu.json) |
| intel_windows_20260923 | cpu | 4 | wd14_v3 | 1.12× | 1.37× | — / 増分 | [対象](intel_windows_20260923/wd14_v3-b4-cpu.json) / [balanced](intel_windows_20260923/balanced-b4-cpu.json) |
| intel_windows_20260923 | webgpu | 4 | wd14_v3 | 1.07× | 1.30× | — / 増分 | [対象](intel_windows_20260923/wd14_v3-b4-webgpu.json) / [balanced](intel_windows_20260923/balanced-b4-webgpu.json) |
| nvidia_linux_20260923 | cpu | 1 | wd14_v3 | 1.23× | 1.19× | — / 増分 | [対象](nvidia_linux_20260923/wd14_v3-b1-cpu.json) / [balanced](nvidia_linux_20260923/balanced-b1-cpu.json) |
| nvidia_linux_20260923 | cuda | 1 | wd14_v3 | 1.15× | 1.03× | 1.00× / ピーク | [対象](nvidia_linux_20260923/wd14_v3-b1-cuda.json) / [balanced](nvidia_linux_20260923/balanced-b1-cuda.json) |
| nvidia_linux_20260923 | tensorrt | 1 | wd14_v3 | 0.92× | 1.06× | 0.95× / ピーク | [対象](nvidia_linux_20260923/wd14_v3-b1-tensorrt.json) / [balanced](nvidia_linux_20260923/balanced-b1-tensorrt.json) |
| nvidia_linux_20260923 | webgpu | 1 | wd14_v3 | 1.03× | 1.05× | 1.21× / ピーク | [対象](nvidia_linux_20260923/wd14_v3-b1-webgpu.json) / [balanced](nvidia_linux_20260923/balanced-b1-webgpu.json) |
| nvidia_linux_20260923 | cpu | 4 | wd14_v3 | 1.09× | 1.37× | — / 増分 | [対象](nvidia_linux_20260923/wd14_v3-b4-cpu.json) / [balanced](nvidia_linux_20260923/balanced-b4-cpu.json) |
| nvidia_linux_20260923 | cuda | 4 | wd14_v3 | 1.07× | 1.04× | 1.00× / ピーク | [対象](nvidia_linux_20260923/wd14_v3-b4-cuda.json) / [balanced](nvidia_linux_20260923/balanced-b4-cuda.json) |
| nvidia_linux_20260923 | tensorrt | 4 | wd14_v3 | 0.90× | 1.04× | 1.10× / ピーク | [対象](nvidia_linux_20260923/wd14_v3-b4-tensorrt.json) / [balanced](nvidia_linux_20260923/balanced-b4-tensorrt.json) |
| nvidia_linux_20260923 | webgpu | 4 | wd14_v3 | 1.04× | 0.99× | 1.64× / ピーク | [対象](nvidia_linux_20260923/wd14_v3-b4-webgpu.json) / [balanced](nvidia_linux_20260923/balanced-b4-webgpu.json) |
| nvidia_windows_20260923 | cpu | 1 | wd14_v3 | 1.11× | 1.20× | — / 増分 | [対象](nvidia_windows_20260923/wd14_v3-b1-cpu.json) / [balanced](nvidia_windows_20260923/balanced-b1-cpu.json) |
| nvidia_windows_20260923 | cuda | 1 | wd14_v3 | 1.14× | 1.02× | 1.00× / ピーク | [対象](nvidia_windows_20260923/wd14_v3-b1-cuda.json) / [balanced](nvidia_windows_20260923/balanced-b1-cuda.json) |
| nvidia_windows_20260923 | tensorrt | 1 | wd14_v3 | 0.88× | 0.93× | 0.98× / ピーク | [対象](nvidia_windows_20260923/wd14_v3-b1-tensorrt.json) / [balanced](nvidia_windows_20260923/balanced-b1-tensorrt.json) |
| nvidia_windows_20260923 | webgpu | 1 | wd14_v3 | 1.09× | 1.14× | 1.10× / ピーク | [対象](nvidia_windows_20260923/wd14_v3-b1-webgpu.json) / [balanced](nvidia_windows_20260923/balanced-b1-webgpu.json) |
| nvidia_windows_20260923 | cpu | 4 | wd14_v3 | 1.11× | 1.37× | — / 増分 | [対象](nvidia_windows_20260923/wd14_v3-b4-cpu.json) / [balanced](nvidia_windows_20260923/balanced-b4-cpu.json) |
| nvidia_windows_20260923 | cuda | 4 | wd14_v3 | 1.05× | 1.04× | 0.97× / ピーク | [対象](nvidia_windows_20260923/wd14_v3-b4-cuda.json) / [balanced](nvidia_windows_20260923/balanced-b4-cuda.json) |
| nvidia_windows_20260923 | tensorrt | 4 | wd14_v3 | 0.85× | 0.95× | 1.05× / ピーク | [対象](nvidia_windows_20260923/wd14_v3-b4-tensorrt.json) / [balanced](nvidia_windows_20260923/balanced-b4-tensorrt.json) |
| nvidia_windows_20260923 | webgpu | 4 | wd14_v3 | 1.07× | 1.34× | 1.35× / ピーク | [対象](nvidia_windows_20260923/wd14_v3-b4-webgpu.json) / [balanced](nvidia_windows_20260923/balanced-b4-webgpu.json) |
| nvidia_windows_summary | cuda | 1 | wd14_v3 | 1.08× | 1.03× | 1.00× / 増分 | [対象](nvidia_windows_summary.json) / [balanced](nvidia_windows_summary.json) |
| nvidia_windows_summary | tensorrt | 1 | wd14_v3 | 0.88× | 0.93× | 0.95× / 増分 | [対象](nvidia_windows_summary.json) / [balanced](nvidia_windows_summary.json) |
| nvidia_windows_summary | webgpu | 1 | wd14_v3 | 1.06× | 1.08× | 1.21× / 増分 | [対象](nvidia_windows_summary.json) / [balanced](nvidia_windows_summary.json) |
| nvidia_windows_summary | cuda | 4 | wd14_v3 | 1.04× | 1.04× | 0.99× / 増分 | [対象](nvidia_windows_summary.json) / [balanced](nvidia_windows_summary.json) |
| nvidia_windows_summary | tensorrt | 4 | wd14_v3 | 0.84× | 0.95× | 1.10× / 増分 | [対象](nvidia_windows_summary.json) / [balanced](nvidia_windows_summary.json) |
| nvidia_windows_summary | webgpu | 4 | wd14_v3 | 1.07× | 1.07× | 1.65× / 増分 | [対象](nvidia_windows_summary.json) / [balanced](nvidia_windows_summary.json) |
| directml_legacy | DirectML | 4 | lightweight | 0.26× | — | 0.79× / ピーク | [対象](legacy_results.md) / [balanced](legacy_results.md) |
| directml_legacy | DirectML | 4 | high | 2.78× | — | 1.76× / ピーク | [対象](legacy_results.md) / [balanced](legacy_results.md) |
| directml_legacy | DirectML | 4 | ultra | 3.78× | — | 3.21× / ピーク | [対象](legacy_results.md) / [balanced](legacy_results.md) |
| directml_legacy | DirectML | 4 | wd14_v3 | 0.96× | — | 1.45× / ピーク | [対象](legacy_results.md) / [balanced](legacy_results.md) |

## 補助観測と旧記録

- Switchの分割実行について、[外部メモリ監視](ncnn_switch_20261003/ultra-stream128-memory-watch.json)も保存。下表は全サンプルから集計し、プロセス内監視と測定範囲が異なる。

| 観測 | 終了コード | RSS最大 MiB | 空きRAM最小 MiB | 空きswap最小 MiB | GPU稼働最大 % |
|---|---:|---:|---:|---:|---:|
| [ultra-memory-watch](ncnn_switch_20261003/ultra-memory-watch.json) | -15 | 未測定 | 未測定 | 未測定 | 未測定 |
| [ultra-stream128-memory-watch](ncnn_switch_20261003/ultra-stream128-memory-watch.json) | 0 | 660.27 | 1,375.48 | 2,067.36 | 99.70 |

[9月の全100条件・ばらつき・p95](matrix_details_20260923.md)、[旧測定の速度・VRAM記録](legacy_results.md)、[AMD 4GiBの説明](amd_barcelo_4gb.md)も参照。古いMarkdownだけの記録は元の条件とともに保持する。
