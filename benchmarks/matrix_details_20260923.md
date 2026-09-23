# 2026-09-23 マトリクス全条件

[総合比較へ](../BENCHMARKS.md)

全100条件（92成功・8失敗）。時間はms/枚。中央値はセット中央値の中央値、範囲はセット中央値の最小～最大、p95は各セットのp95の最小～最大であり全60回をまとめたp95ではない。RSS/VRAMはMiB、セッション構築は秒。`—` は失敗による値なし、`未測定` はnull。EPの略称はCPU / CUDA / TensorRT / OpenVINO / DirectML / WebGPU。

## RTX 2070 Max-Q / Linux

[manifest](nvidia_linux_20260923/manifest.json) / [summary](nvidia_linux_20260923/summary.json)

| モデル | batch | EP | 状態 | 中央値 | 中央値範囲 | p95範囲 | CPU比 | RSS | VRAM増分 | 構築秒 | JSON |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 1 | CPU | ok | 593.10 | 586.67～601.21 | 597.35～630.13 | 1.00× | 1040.49 | 未測定 | 1.29 | [balanced-b1-cpu](nvidia_linux_20260923/balanced-b1-cpu.json) |
| balanced | 1 | CUDA | ok | 58.16 | 57.90～58.23 | 58.95～59.36 | 10.20× | 1222.22 | 1146.00 | 1.38 | [balanced-b1-cuda](nvidia_linux_20260923/balanced-b1-cuda.json) |
| balanced | 1 | TensorRT | ok | 46.06 | 45.88～46.14 | 47.23～47.66 | 12.88× | 3150.19 | 954.00 | 3.23 | [balanced-b1-tensorrt](nvidia_linux_20260923/balanced-b1-tensorrt.json) |
| balanced | 1 | WebGPU | ok | 120.08 | 119.84～120.62 | 121.39～122.12 | 4.94× | 658.32 | 687.00 | 1.79 | [balanced-b1-webgpu](nvidia_linux_20260923/balanced-b1-webgpu.json) |
| balanced | 4 | CPU | ok | 702.79 | 699.57～704.09 | 707.54～714.87 | 1.00× | 1360.75 | 未測定 | 1.30 | [balanced-b4-cpu](nvidia_linux_20260923/balanced-b4-cpu.json) |
| balanced | 4 | CUDA | ok | 57.97 | 57.94～58.14 | 58.14～60.75 | 12.12× | 1211.48 | 1146.00 | 1.42 | [balanced-b4-cuda](nvidia_linux_20260923/balanced-b4-cuda.json) |
| balanced | 4 | TensorRT | ok | 43.67 | 43.51～43.70 | 43.65～44.02 | 16.09× | 3188.66 | 1204.00 | 3.22 | [balanced-b4-tensorrt](nvidia_linux_20260923/balanced-b4-tensorrt.json) |
| balanced | 4 | WebGPU | ok | 110.63 | 110.28～111.04 | 110.58～111.23 | 6.35× | 669.47 | 1103.00 | 1.78 | [balanced-b4-webgpu](nvidia_linux_20260923/balanced-b4-webgpu.json) |
| wd14_v3 | 1 | CPU | ok | 728.89 | 717.97～733.82 | 723.96～757.72 | 1.00× | 1236.57 | 未測定 | 1.46 | [wd14_v3-b1-cpu](nvidia_linux_20260923/wd14_v3-b1-cpu.json) |
| wd14_v3 | 1 | CUDA | ok | 66.89 | 66.79～73.89 | 68.01～74.92 | 10.90× | 1258.54 | 1147.00 | 1.63 | [wd14_v3-b1-cuda](nvidia_linux_20260923/wd14_v3-b1-cuda.json) |
| wd14_v3 | 1 | TensorRT | ok | 42.37 | 42.36～42.39 | 43.55～44.37 | 17.20× | 3345.49 | 909.00 | 2.79 | [wd14_v3-b1-tensorrt](nvidia_linux_20260923/wd14_v3-b1-tensorrt.json) |
| wd14_v3 | 1 | WebGPU | ok | 124.20 | 123.57～124.45 | 124.38～125.24 | 5.87× | 690.62 | 841.00 | 2.05 | [wd14_v3-b1-webgpu](nvidia_linux_20260923/wd14_v3-b1-webgpu.json) |
| wd14_v3 | 4 | CPU | ok | 767.28 | 767.14～776.32 | 773.14～781.40 | 1.00× | 1863.11 | 未測定 | 1.47 | [wd14_v3-b4-cpu](nvidia_linux_20260923/wd14_v3-b4-cpu.json) |
| wd14_v3 | 4 | CUDA | ok | 62.31 | 62.07～62.49 | 62.36～62.69 | 12.31× | 1260.76 | 1147.00 | 1.66 | [wd14_v3-b4-cuda](nvidia_linux_20260923/wd14_v3-b4-cuda.json) |
| wd14_v3 | 4 | TensorRT | ok | 39.49 | 39.42～39.60 | 39.60～39.79 | 19.43× | 3331.25 | 1323.00 | 3.10 | [wd14_v3-b4-tensorrt](nvidia_linux_20260923/wd14_v3-b4-tensorrt.json) |
| wd14_v3 | 4 | WebGPU | ok | 115.29 | 115.13～115.60 | 115.39～115.79 | 6.66× | 662.17 | 1832.00 | 1.90 | [wd14_v3-b4-webgpu](nvidia_linux_20260923/wd14_v3-b4-webgpu.json) |


実行ノードのEP（成功行）:

| 要求EP | 実行ノードEP |
| --- | --- |
| CPU | CPU |
| CUDA | CPU, CUDA |
| TensorRT | Tensorrt |
| WebGPU | CPU, WebGpu |

## RTX 2070 Max-Q / Windows

[manifest](nvidia_windows_20260923/manifest.json) / [summary](nvidia_windows_20260923/summary.json)

| モデル | batch | EP | 状態 | 中央値 | 中央値範囲 | p95範囲 | CPU比 | RSS | VRAM増分 | 構築秒 | JSON |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 1 | CPU | ok | 944.55 | 935.19～978.23 | 960.04～992.76 | 1.00× | 1045.88 | 未測定 | 1.71 | [balanced-b1-cpu](nvidia_windows_20260923/balanced-b1-cpu.json) |
| balanced | 1 | CUDA | ok | 76.75 | 76.51～76.83 | 78.84～79.68 | 12.31× | 1024.37 | 1131.00 | 1.51 | [balanced-b1-cuda](nvidia_windows_20260923/balanced-b1-cuda.json) |
| balanced | 1 | DirectML | 失敗 | — | — | — | — | — | — | — | [balanced-b1-directml](nvidia_windows_20260923/balanced-b1-directml.json) |
| balanced | 1 | TensorRT | ok | 60.90 | 60.88～61.12 | 62.72～65.49 | 15.51× | 3075.07 | 985.00 | 3.47 | [balanced-b1-tensorrt](nvidia_windows_20260923/balanced-b1-tensorrt.json) |
| balanced | 1 | WebGPU | ok | 197.99 | 197.39～198.14 | 199.84～201.05 | 4.77× | 671.45 | 709.00 | 2.44 | [balanced-b1-webgpu](nvidia_windows_20260923/balanced-b1-webgpu.json) |
| balanced | 4 | CPU | ok | 929.54 | 921.21～940.12 | 930.43～952.45 | 1.00× | 1368.91 | 未測定 | 1.43 | [balanced-b4-cpu](nvidia_windows_20260923/balanced-b4-cpu.json) |
| balanced | 4 | CUDA | ok | 70.93 | 70.87～71.00 | 71.34～72.15 | 13.11× | 1017.29 | 1195.00 | 1.50 | [balanced-b4-cuda](nvidia_windows_20260923/balanced-b4-cuda.json) |
| balanced | 4 | DirectML | 失敗 | — | — | — | — | — | — | — | [balanced-b4-directml](nvidia_windows_20260923/balanced-b4-directml.json) |
| balanced | 4 | TensorRT | ok | 55.30 | 55.22～55.31 | 55.49～55.65 | 16.81× | 3096.50 | 1237.00 | 3.29 | [balanced-b4-tensorrt](nvidia_windows_20260923/balanced-b4-tensorrt.json) |
| balanced | 4 | WebGPU | ok | 181.62 | 181.56～181.62 | 181.81～182.21 | 5.12× | 583.24 | 1133.00 | 2.49 | [balanced-b4-webgpu](nvidia_windows_20260923/balanced-b4-webgpu.json) |
| wd14_v3 | 1 | CPU | ok | 1052.67 | 1037.89～1059.33 | 1056.28～1100.98 | 1.00× | 1250.84 | 未測定 | 2.05 | [wd14_v3-b1-cpu](nvidia_windows_20260923/wd14_v3-b1-cpu.json) |
| wd14_v3 | 1 | CUDA | ok | 87.53 | 84.78～93.47 | 88.22～112.25 | 12.03× | 1048.73 | 1131.00 | 2.12 | [wd14_v3-b1-cuda](nvidia_windows_20260923/wd14_v3-b1-cuda.json) |
| wd14_v3 | 1 | DirectML | ok | 208.45 | 207.88～209.61 | 212.66～228.81 | 5.05× | 870.01 | 866.00 | 2.38 | [wd14_v3-b1-directml](nvidia_windows_20260923/wd14_v3-b1-directml.json) |
| wd14_v3 | 1 | TensorRT | ok | 53.72 | 53.71～53.83 | 55.90～56.51 | 19.60× | 2870.95 | 939.00 | 3.73 | [wd14_v3-b1-tensorrt](nvidia_windows_20260923/wd14_v3-b1-tensorrt.json) |
| wd14_v3 | 1 | WebGPU | ok | 214.92 | 214.75～215.15 | 216.43～217.42 | 4.90× | 766.92 | 868.00 | 2.56 | [wd14_v3-b1-webgpu](nvidia_windows_20260923/wd14_v3-b1-webgpu.json) |
| wd14_v3 | 4 | CPU | ok | 1032.19 | 1029.73～1034.71 | 1034.21～1069.13 | 1.00× | 1879.95 | 未測定 | 2.15 | [wd14_v3-b4-cpu](nvidia_windows_20260923/wd14_v3-b4-cpu.json) |
| wd14_v3 | 4 | CUDA | ok | 74.39 | 74.37～74.69 | 74.94～75.37 | 13.88× | 1058.55 | 1131.00 | 2.13 | [wd14_v3-b4-cuda](nvidia_windows_20260923/wd14_v3-b4-cuda.json) |
| wd14_v3 | 4 | DirectML | ok | 128.72 | 128.05～128.78 | 131.52～134.05 | 8.02× | 862.66 | 1968.00 | 2.55 | [wd14_v3-b4-directml](nvidia_windows_20260923/wd14_v3-b4-directml.json) |
| wd14_v3 | 4 | TensorRT | ok | 46.88 | 46.88～46.90 | 47.45～47.69 | 22.02× | 2929.23 | 1353.00 | 3.00 | [wd14_v3-b4-tensorrt](nvidia_windows_20260923/wd14_v3-b4-tensorrt.json) |
| wd14_v3 | 4 | WebGPU | ok | 194.55 | 194.39～194.58 | 195.07～195.23 | 5.31× | 783.65 | 1863.00 | 2.68 | [wd14_v3-b4-webgpu](nvidia_windows_20260923/wd14_v3-b4-webgpu.json) |

- [balanced-b1-directml](nvidia_windows_20260923/balanced-b1-directml.json): `RuntimeError: requested DmlExecutionProvider, active providers are ['CPUExecutionProvider']`

- [balanced-b4-directml](nvidia_windows_20260923/balanced-b4-directml.json): `RuntimeError: requested DmlExecutionProvider, active providers are ['CPUExecutionProvider']`


実行ノードのEP（成功行）:

| 要求EP | 実行ノードEP |
| --- | --- |
| CPU | CPU |
| CUDA | CPU, CUDA |
| TensorRT | Tensorrt |
| WebGPU | CPU, WebGpu |
| DirectML | CPU, Dml |

## UHD 630 / Linux

[manifest](intel_linux_20260923/manifest.json) / [summary](intel_linux_20260923/summary.json)

| モデル | batch | EP | 状態 | 中央値 | 中央値範囲 | p95範囲 | CPU比 | RSS | VRAM増分 | 構築秒 | JSON |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 1 | CPU | ok | 853.91 | 842.89～863.42 | 849.75～906.60 | 1.00× | 1071.17 | 未測定 | 1.16 | [balanced-b1-cpu](intel_linux_20260923/balanced-b1-cpu.json) |
| balanced | 1 | OpenVINO | ok | 800.19 | 799.89～801.03 | 804.19～817.64 | 1.07× | 1924.47 | 未測定 | 6.58 | [balanced-b1-intel](intel_linux_20260923/balanced-b1-intel.json) |
| balanced | 1 | WebGPU | ok | 118.26 | 118.08～118.61 | 119.28～121.67 | 7.22× | 699.18 | 未測定 | 1.78 | [balanced-b1-webgpu](intel_linux_20260923/balanced-b1-webgpu.json) |
| balanced | 4 | CPU | ok | 938.49 | 935.87～945.52 | 937.50～947.16 | 1.00× | 1360.68 | 未測定 | 1.35 | [balanced-b4-cpu](intel_linux_20260923/balanced-b4-cpu.json) |
| balanced | 4 | OpenVINO | ok | 780.84 | 778.30～782.11 | 786.26～791.11 | 1.20× | 1550.22 | 未測定 | 3.59 | [balanced-b4-intel](intel_linux_20260923/balanced-b4-intel.json) |
| balanced | 4 | WebGPU | ok | 109.04 | 108.13～109.74 | 108.90～110.15 | 8.61× | 698.71 | 未測定 | 1.79 | [balanced-b4-webgpu](intel_linux_20260923/balanced-b4-webgpu.json) |
| wd14_v3 | 1 | CPU | ok | 727.81 | 712.25～731.61 | 720.37～737.85 | 1.00× | 1236.95 | 未測定 | 1.55 | [wd14_v3-b1-cpu](intel_linux_20260923/wd14_v3-b1-cpu.json) |
| wd14_v3 | 1 | OpenVINO | ok | 512.54 | 511.18～516.28 | 514.28～560.24 | 1.42× | 2080.57 | 未測定 | 9.64 | [wd14_v3-b1-intel](intel_linux_20260923/wd14_v3-b1-intel.json) |
| wd14_v3 | 1 | WebGPU | ok | 122.25 | 121.99～122.28 | 122.87～123.08 | 5.95× | 653.65 | 未測定 | 1.96 | [wd14_v3-b1-webgpu](intel_linux_20260923/wd14_v3-b1-webgpu.json) |
| wd14_v3 | 4 | CPU | ok | 796.23 | 769.38～797.07 | 815.58～985.49 | 1.00× | 1863.27 | 未測定 | 1.60 | [wd14_v3-b4-cpu](intel_linux_20260923/wd14_v3-b4-cpu.json) |
| wd14_v3 | 4 | OpenVINO | ok | 518.31 | 500.99～604.42 | 505.54～718.68 | 1.54× | 1700.34 | 未測定 | 3.73 | [wd14_v3-b4-intel](intel_linux_20260923/wd14_v3-b4-intel.json) |
| wd14_v3 | 4 | WebGPU | ok | 114.21 | 113.68～114.95 | 115.00～126.50 | 6.97× | 660.81 | 未測定 | 1.97 | [wd14_v3-b4-webgpu](intel_linux_20260923/wd14_v3-b4-webgpu.json) |


実行ノードのEP（成功行）:

| 要求EP | 実行ノードEP |
| --- | --- |
| CPU | CPU |
| OpenVINO | OpenVINO |
| WebGPU | CPU, WebGpu |

## UHD 630 / Windows

[manifest](intel_windows_20260923/manifest.json) / [summary](intel_windows_20260923/summary.json)

| モデル | batch | EP | 状態 | 中央値 | 中央値範囲 | p95範囲 | CPU比 | RSS | VRAM増分 | 構築秒 | JSON |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 1 | CPU | ok | 980.37 | 974.46～985.98 | 1004.82～1305.57 | 1.00× | 1045.84 | 未測定 | 1.47 | [balanced-b1-cpu](intel_windows_20260923/balanced-b1-cpu.json) |
| balanced | 1 | DirectML | 失敗 | — | — | — | — | — | — | — | [balanced-b1-directml](intel_windows_20260923/balanced-b1-directml.json) |
| balanced | 1 | OpenVINO | 失敗 | — | — | — | — | — | — | — | [balanced-b1-intel](intel_windows_20260923/balanced-b1-intel.json) |
| balanced | 1 | WebGPU | ok | 197.02 | 196.89～197.10 | 197.51～198.35 | 4.98× | 708.13 | 未測定 | 2.42 | [balanced-b1-webgpu](intel_windows_20260923/balanced-b1-webgpu.json) |
| balanced | 4 | CPU | ok | 930.74 | 928.55～934.73 | 937.46～941.06 | 1.00× | 1369.37 | 未測定 | 1.42 | [balanced-b4-cpu](intel_windows_20260923/balanced-b4-cpu.json) |
| balanced | 4 | DirectML | 失敗 | — | — | — | — | — | — | — | [balanced-b4-directml](intel_windows_20260923/balanced-b4-directml.json) |
| balanced | 4 | OpenVINO | 失敗 | — | — | — | — | — | — | — | [balanced-b4-intel](intel_windows_20260923/balanced-b4-intel.json) |
| balanced | 4 | WebGPU | ok | 181.36 | 181.26～181.43 | 181.53～181.70 | 5.13× | 603.27 | 未測定 | 2.38 | [balanced-b4-webgpu](intel_windows_20260923/balanced-b4-webgpu.json) |
| wd14_v3 | 1 | CPU | ok | 1121.18 | 1115.04～1138.62 | 1136.66～1163.09 | 1.00× | 1252.09 | 未測定 | 2.28 | [wd14_v3-b1-cpu](intel_windows_20260923/wd14_v3-b1-cpu.json) |
| wd14_v3 | 1 | DirectML | ok | 2447.17 | 2435.00～2451.97 | 2452.15～2467.05 | 0.46× | 1498.73 | 未測定 | 2.16 | [wd14_v3-b1-directml](intel_windows_20260923/wd14_v3-b1-directml.json) |
| wd14_v3 | 1 | OpenVINO | 失敗 | — | — | — | — | — | — | — | [wd14_v3-b1-intel](intel_windows_20260923/wd14_v3-b1-intel.json) |
| wd14_v3 | 1 | WebGPU | ok | 213.82 | 213.70～213.96 | 214.97～215.31 | 5.24× | 716.78 | 未測定 | 2.50 | [wd14_v3-b1-webgpu](intel_windows_20260923/wd14_v3-b1-webgpu.json) |
| wd14_v3 | 4 | CPU | ok | 1044.63 | 1043.78～1050.57 | 1051.61～1064.11 | 1.00× | 1879.89 | 未測定 | 2.15 | [wd14_v3-b4-cpu](intel_windows_20260923/wd14_v3-b4-cpu.json) |
| wd14_v3 | 4 | DirectML | ok | 2330.25 | 2315.73～2341.42 | 2333.71～2378.81 | 0.45× | 2577.54 | 未測定 | 2.13 | [wd14_v3-b4-directml](intel_windows_20260923/wd14_v3-b4-directml.json) |
| wd14_v3 | 4 | OpenVINO | 失敗 | — | — | — | — | — | — | — | [wd14_v3-b4-intel](intel_windows_20260923/wd14_v3-b4-intel.json) |
| wd14_v3 | 4 | WebGPU | ok | 193.57 | 193.52～193.62 | 193.77～194.00 | 5.40× | 781.94 | 未測定 | 2.56 | [wd14_v3-b4-webgpu](intel_windows_20260923/wd14_v3-b4-webgpu.json) |

- [balanced-b1-directml](intel_windows_20260923/balanced-b1-directml.json): `RuntimeError: requested DmlExecutionProvider, active providers are ['CPUExecutionProvider']`

- [balanced-b1-intel](intel_windows_20260923/balanced-b1-intel.json): `RuntimeError: OpenVINO GPU.0 is NVIDIA GeForce RTX 2070 with Max-Q Design (dGPU); an Intel GPU is required for the Intel benchmark`

- [balanced-b4-directml](intel_windows_20260923/balanced-b4-directml.json): `RuntimeError: requested DmlExecutionProvider, active providers are ['CPUExecutionProvider']`

- [balanced-b4-intel](intel_windows_20260923/balanced-b4-intel.json): `RuntimeError: OpenVINO GPU.0 is NVIDIA GeForce RTX 2070 with Max-Q Design (dGPU); an Intel GPU is required for the Intel benchmark`

- [wd14_v3-b1-intel](intel_windows_20260923/wd14_v3-b1-intel.json): `RuntimeError: OpenVINO GPU.0 is NVIDIA GeForce RTX 2070 with Max-Q Design (dGPU); an Intel GPU is required for the Intel benchmark`

- [wd14_v3-b4-intel](intel_windows_20260923/wd14_v3-b4-intel.json): `RuntimeError: OpenVINO GPU.0 is NVIDIA GeForce RTX 2070 with Max-Q Design (dGPU); an Intel GPU is required for the Intel benchmark`


実行ノードのEP（成功行）:

| 要求EP | 実行ノードEP |
| --- | --- |
| CPU | CPU |
| WebGPU | CPU, WebGpu |
| DirectML | CPU, Dml |

## Iris Xe / Linux

[manifest](intel2_linux_20260923/manifest.json) / [summary](intel2_linux_20260923/summary.json)

| モデル | batch | EP | 状態 | 中央値 | 中央値範囲 | p95範囲 | CPU比 | RSS | VRAM増分 | 構築秒 | JSON |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 1 | CPU | ok | 524.27 | 516.02～596.11 | 519.34～723.62 | 1.00× | 1094.38 | 未測定 | 0.80 | [balanced-b1-cpu](intel2_linux_20260923/balanced-b1-cpu.json) |
| balanced | 1 | OpenVINO | ok | 222.12 | 218.55～240.45 | 219.82～250.73 | 2.36× | 1918.58 | 未測定 | 6.89 | [balanced-b1-intel](intel2_linux_20260923/balanced-b1-intel.json) |
| balanced | 1 | WebGPU | ok | 430.66 | 429.71～432.51 | 431.71～434.89 | 1.22× | 625.59 | 未測定 | 1.56 | [balanced-b1-webgpu](intel2_linux_20260923/balanced-b1-webgpu.json) |
| balanced | 4 | CPU | ok | 611.91 | 611.30～612.49 | 613.46～615.45 | 1.00× | 1380.54 | 未測定 | 0.84 | [balanced-b4-cpu](intel2_linux_20260923/balanced-b4-cpu.json) |
| balanced | 4 | OpenVINO | ok | 212.00 | 211.96～213.20 | 212.39～237.68 | 2.89× | 1517.64 | 未測定 | 4.59 | [balanced-b4-intel](intel2_linux_20260923/balanced-b4-intel.json) |
| balanced | 4 | WebGPU | ok | 411.68 | 411.49～412.49 | 412.18～415.27 | 1.49× | 601.07 | 未測定 | 1.57 | [balanced-b4-webgpu](intel2_linux_20260923/balanced-b4-webgpu.json) |
| wd14_v3 | 1 | CPU | ok | 508.45 | 502.63～648.55 | 507.62～656.95 | 1.00× | 1267.35 | 未測定 | 0.90 | [wd14_v3-b1-cpu](intel2_linux_20260923/wd14_v3-b1-cpu.json) |
| wd14_v3 | 1 | OpenVINO | ok | 131.70 | 130.74～139.97 | 131.66～143.21 | 3.86× | 1656.09 | 未測定 | 4.69 | [wd14_v3-b1-intel](intel2_linux_20260923/wd14_v3-b1-intel.json) |
| wd14_v3 | 1 | WebGPU | ok | 482.13 | 479.54～482.89 | 480.86～485.10 | 1.05× | 595.79 | 未測定 | 1.86 | [wd14_v3-b1-webgpu](intel2_linux_20260923/wd14_v3-b1-webgpu.json) |
| wd14_v3 | 4 | CPU | ok | 648.49 | 648.17～650.17 | 649.34～736.21 | 1.00× | 1893.98 | 未測定 | 0.91 | [wd14_v3-b4-cpu](intel2_linux_20260923/wd14_v3-b4-cpu.json) |
| wd14_v3 | 4 | OpenVINO | ok | 131.43 | 130.80～133.63 | 132.23～151.12 | 4.93× | 1656.07 | 未測定 | 4.69 | [wd14_v3-b4-intel](intel2_linux_20260923/wd14_v3-b4-intel.json) |
| wd14_v3 | 4 | WebGPU | ok | 470.14 | 470.14～470.31 | 471.30～474.28 | 1.38× | 597.27 | 未測定 | 1.64 | [wd14_v3-b4-webgpu](intel2_linux_20260923/wd14_v3-b4-webgpu.json) |


実行ノードのEP（成功行）:

| 要求EP | 実行ノードEP |
| --- | --- |
| CPU | CPU |
| OpenVINO | OpenVINO |
| WebGPU | CPU, WebGpu |

## HD 530 / Linux

[manifest](intel3_linux_20260923/manifest.json) / [summary](intel3_linux_20260923/summary.json)

| モデル | batch | EP | 状態 | 中央値 | 中央値範囲 | p95範囲 | CPU比 | RSS | VRAM増分 | 構築秒 | JSON |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 1 | CPU | ok | 1015.37 | 979.40～1017.39 | 1195.70～1237.94 | 1.00× | 1053.02 | 未測定 | 1.72 | [balanced-b1-cpu](intel3_linux_20260923/balanced-b1-cpu.json) |
| balanced | 1 | OpenVINO | ok | 689.83 | 689.71～690.16 | 691.54～692.73 | 1.47× | 1538.06 | 未測定 | 5.02 | [balanced-b1-intel](intel3_linux_20260923/balanced-b1-intel.json) |
| balanced | 1 | WebGPU | ok | 2017.63 | 2017.55～2018.46 | 2028.23～2087.39 | 0.50× | 576.84 | 未測定 | 1.87 | [balanced-b1-webgpu](intel3_linux_20260923/balanced-b1-webgpu.json) |
| balanced | 4 | CPU | ok | 1189.94 | 1163.33～1889.90 | 1314.82～2863.24 | 1.00× | 1350.27 | 未測定 | 1.90 | [balanced-b4-cpu](intel3_linux_20260923/balanced-b4-cpu.json) |
| balanced | 4 | OpenVINO | ok | 658.96 | 657.71～659.12 | 660.04～660.74 | 1.81× | 1538.02 | 未測定 | 5.23 | [balanced-b4-intel](intel3_linux_20260923/balanced-b4-intel.json) |
| balanced | 4 | WebGPU | ok | 1926.70 | 1924.99～1926.84 | 1932.14～1936.82 | 0.62× | 577.22 | 未測定 | 1.84 | [balanced-b4-webgpu](intel3_linux_20260923/balanced-b4-webgpu.json) |
| wd14_v3 | 1 | CPU | ok | 1198.24 | 1198.08～1204.34 | 1394.33～1514.63 | 1.00× | 1221.97 | 未測定 | 2.10 | [wd14_v3-b1-cpu](intel3_linux_20260923/wd14_v3-b1-cpu.json) |
| wd14_v3 | 1 | OpenVINO | ok | 439.04 | 438.55～439.30 | 442.01～443.87 | 2.73× | 1688.90 | 未測定 | 5.16 | [wd14_v3-b1-intel](intel3_linux_20260923/wd14_v3-b1-intel.json) |
| wd14_v3 | 1 | WebGPU | ok | 2202.89 | 2201.53～2204.28 | 2214.16～2239.34 | 0.54× | 588.10 | 未測定 | 2.00 | [wd14_v3-b1-webgpu](intel3_linux_20260923/wd14_v3-b1-webgpu.json) |
| wd14_v3 | 4 | CPU | ok | 1277.45 | 1270.49～1284.83 | 1390.00～1428.23 | 1.00× | 1847.31 | 未測定 | 2.09 | [wd14_v3-b4-cpu](intel3_linux_20260923/wd14_v3-b4-cpu.json) |
| wd14_v3 | 4 | OpenVINO | ok | 432.41 | 432.17～432.43 | 434.07～435.97 | 2.95× | 1688.70 | 未測定 | 5.10 | [wd14_v3-b4-intel](intel3_linux_20260923/wd14_v3-b4-intel.json) |
| wd14_v3 | 4 | WebGPU | ok | 2088.64 | 2088.09～2090.28 | 2097.11～2097.72 | 0.61× | 588.75 | 未測定 | 2.01 | [wd14_v3-b4-webgpu](intel3_linux_20260923/wd14_v3-b4-webgpu.json) |


実行ノードのEP（成功行）:

| 要求EP | 実行ノードEP |
| --- | --- |
| CPU | CPU |
| OpenVINO | OpenVINO |
| WebGPU | CPU, WebGpu |

## Radeon Graphics / Windows

[manifest](amd_windows_20260923/manifest.json) / [summary](amd_windows_20260923/summary.json)

| モデル | batch | EP | 状態 | 中央値 | 中央値範囲 | p95範囲 | CPU比 | RSS | VRAM増分 | 構築秒 | JSON |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 1 | CPU | ok | 808.58 | 802.06～816.70 | 817.67～970.26 | 1.00× | 1060.88 | 未測定 | 1.64 | [balanced-b1-cpu](amd_windows_20260923/balanced-b1-cpu.json) |
| balanced | 1 | DirectML | ok | 562.46 | 557.38～582.45 | 562.64～624.97 | 1.44× | 636.77 | 未測定 | 1.85 | [balanced-b1-directml](amd_windows_20260923/balanced-b1-directml.json) |
| balanced | 1 | WebGPU | ok | 736.16 | 727.79～741.24 | 733.49～765.83 | 1.10× | 718.79 | 未測定 | 2.21 | [balanced-b1-webgpu](amd_windows_20260923/balanced-b1-webgpu.json) |
| balanced | 4 | CPU | ok | 1063.75 | 1061.85～1066.46 | 1124.74～1497.70 | 1.00× | 1367.15 | 未測定 | 1.06 | [balanced-b4-cpu](amd_windows_20260923/balanced-b4-cpu.json) |
| balanced | 4 | DirectML | ok | 576.66 | 568.37～577.65 | 578.72～583.37 | 1.84× | 601.12 | 未測定 | 1.47 | [balanced-b4-directml](amd_windows_20260923/balanced-b4-directml.json) |
| balanced | 4 | WebGPU | ok | 749.72 | 718.22～821.03 | 753.41～964.75 | 1.42× | 601.95 | 未測定 | 2.84 | [balanced-b4-webgpu](amd_windows_20260923/balanced-b4-webgpu.json) |
| wd14_v3 | 1 | CPU | ok | 1015.19 | 1007.55～1031.58 | 1035.79～1067.26 | 1.00× | 1265.62 | 未測定 | 2.05 | [wd14_v3-b1-cpu](amd_windows_20260923/wd14_v3-b1-cpu.json) |
| wd14_v3 | 1 | DirectML | ok | 635.64 | 635.37～642.07 | 645.48～664.96 | 1.60× | 745.66 | 未測定 | 3.28 | [wd14_v3-b1-directml](amd_windows_20260923/wd14_v3-b1-directml.json) |
| wd14_v3 | 1 | WebGPU | ok | 984.13 | 975.87～985.01 | 1012.39～1075.58 | 1.03× | 782.88 | 未測定 | 4.23 | [wd14_v3-b1-webgpu](amd_windows_20260923/wd14_v3-b1-webgpu.json) |
| wd14_v3 | 4 | CPU | ok | 1145.20 | 1140.84～1313.78 | 1179.47～1805.05 | 1.00× | 1878.36 | 未測定 | 2.04 | [wd14_v3-b4-cpu](amd_windows_20260923/wd14_v3-b4-cpu.json) |
| wd14_v3 | 4 | DirectML | ok | 626.38 | 613.41～639.71 | 627.64～676.93 | 1.83× | 633.63 | 未測定 | 4.20 | [wd14_v3-b4-directml](amd_windows_20260923/wd14_v3-b4-directml.json) |
| wd14_v3 | 4 | WebGPU | ok | 937.04 | 930.90～954.40 | 935.60～1013.31 | 1.22× | 708.09 | 未測定 | 4.34 | [wd14_v3-b4-webgpu](amd_windows_20260923/wd14_v3-b4-webgpu.json) |


実行ノードのEP（成功行）:

| 要求EP | 実行ノードEP |
| --- | --- |
| CPU | CPU |
| DirectML | CPU, Dml |
| WebGPU | CPU, WebGpu |
