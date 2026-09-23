# ベンチマーク修正の本体反映（2026-09-23）

対象はベンチマーク導入 `34aa6d7` から `45b44c4` までの履歴とコミット差分。本体・通常ランチャー・Serverに必要な変更を移し、測定固有の機能は測定側へ残した。

## コミットごとの対応

| 元コミット | 修正・機能 | 本体への対応 |
| --- | --- | --- |
| `57cdd4c`（導入直前） | AMD Barcelo WebGPU | 既存経路を維持。全デバイス登録をやめ、1デバイス選択と製造元照合を共通化 |
| `caf81e6` | tmpfs環境のHF認証情報の保存先 | `run_tagger.sh`に適用済み。維持 |
| `4846ae4` | CUDA/WebGPU測定、shape・EP確認 | 本体の既存shape検証を維持。起動時の合成画像によるGPUノード検証を追加 |
| `81c3f68` | TensorRT/OpenVINO/DirectML、プロファイルによるEP実行確認 | 通常CLI・ServerのEP明示指定、起動時ノード検証へ反映 |
| `ffb2909` | CUDA依存パッケージ・DLL preload | CUDA 12用`onnxruntime-gpu[cuda,cudnn]<1.27`をランチャーへ反映。`prepare_cuda`を共通化 |
| `3ae58e0` | Windows cuDNN tensor IR DLLとfallback抑止 | DLL検索パス・ハンドル保持と`disable_fallback()`を本体へ反映 |
| `0839aca` | TensorRTランタイム配置手順 | README・通常CLIのライブラリ場所指定へ反映。実行時の探索は共通化 |
| `a20b77a` | OpenVINO DLLと対応runtime導入 | Windows DLLロードと`onnxruntime-openvino==1.24.1` / `openvino==2025.4.1`を通常環境へ反映 |
| `47e4575` | プロバイダー別環境、デバイス番号、CPU基準、再試行 | 通常ランチャーの環境選択とデバイス番号に反映。CPU基準・測定再開はベンチ専用として維持 |
| `2c2fde8` | Intel実名照合、WebGPU製造元・インデックス、DirectML/DXGI区別 | OpenVINOの実名検証、WebGPU単一選択、別WebGPU環境でのDirectML DXGI照合へ反映 |
| `2894766` | Windows/Linux TensorRT探索、ORT bridgeロード、Linux library path | `gpu_runtime.py`へ移動し本体とベンチが共有。Linux起動前の検索パスも通常ランチャーへ反映 |
| `8ed34bb` | OpenVINO照会の別プロセス化、PCI vendor ID、PowerShell login | 照会を共通化し、本体でもプロセスを分離。vendor ID変換を利用。`-Login`は適用済みで維持 |
| `42c0b03` | MIGraphX import/EPの事前確認 | Bashから別プロセスで確認し失敗時は停止。明示MIGraphXは他EPへ切り替えない |
| `5a1ced2` | DirectML optimizer無効化、PowerShell引数配列 | optimizer対策は本体に先行実装済みだったため共通関数へ統合。通常PSも引数を配列で構築 |

`34aa6d7`、`e645bd8`、データ追加コミット（`c266aaf`、`31d9fb9`、`53840ce`、`a4eb569`、`18812ae`、`b277224`）、文書化（`45b44c4`）、一時ファイル整理（`c34ad3f`、`2ebaf77`、`73cecf2`）も確認した。これらには移植すべき追加の通常推論処理はない。ベンチマークの反復回数・VRAM計測・要約JSON・失敗再試行は通常処理へ追加しない。

## 反映先と動作

- [gpu_runtime.py](gpu_runtime.py): CUDA/cuDNN preload、TensorRT探索とORT bridge検証、OpenVINO DLL・別プロセス照会、WebGPU選択、DirectML設定、ノードプロファイル解析。
- [embed_tags_universal.py](embed_tags_universal.py): 通常処理・Serverの両方で共通初期化を利用。起動時の合成画像で出力shapeと要求EPのノード実行を確認し、CPUのみの実行を拒否。プロファイルは一時ディレクトリに保存し検証後に削除。
- [benchmark_nvidia_ep.py](benchmark_nvidia_ep.py): 初期化コードの複製を削除し、同じ共通処理を利用。反復測定と出力JSON形式を維持。
- [run_tagger.sh](run_tagger.sh) / [run_tagger.ps1](run_tagger.ps1): EP・デバイス番号・TensorRTパスの指定、依存環境、終了コード。Bash/Zsh補完にも引数を追加。

CPUノードとの混在は許容する。TensorRTを明示した場合、CUDAだけが動いても成功にしない。自動選択の場合は利用できないEPの理由を表示し、利用可能なGPU EPを選ぶ。Python本体でGPUを要求した状態で候補がなければ停止する。Bashの従来の自動検出でGPUが見つからない場合のCPU選択は維持する。WebGPUの選択情報は実行先の指定の確認であり、物理GPUの負荷を別途測定した証明ではない。

## 通常処理での使い方

通常処理・Server・ベンチマークはGPU初期化を共有する。明示したEPが使えない場合は停止する。GPU起動時は合成画像でノード実行を検証し、その後の処理ではプロファイリングを終了する。これは物理GPU単位の負荷検証や出力精度の保証ではない。

```bash
./run_tagger.sh --provider tensorrt --gpu-index 0 --model-profile balanced /path/to/images
./run_tagger.sh --provider intel --openvino-device GPU.1 /path/to/images
./run_tagger.sh --provider webgpu --webgpu-device-index 1 --target-vendor intel /path/to/images
```

```powershell
.\run_tagger.ps1 -Provider tensorrt -GpuIndex 0 -TensorRtLibDir 'C:\TensorRT-10\bin' -Path C:\Images
.\run_tagger.ps1 -Provider intel -OpenVinoDevice GPU.1 -Path C:\Images
.\run_tagger.ps1 -Provider directml -DirectMlDeviceIndex 1 -TargetVendor intel -Path C:\Images
.\run_tagger.ps1 -Provider webgpu -WebGpuDeviceIndex 1 -TargetVendor intel -Path C:\Images
```

デバイス番号は例。OpenVINOの`GPU.N`、CUDAの`--gpu-index`、DirectMLのDXGI番号、WebGPU番号は別体系なので、実機の一覧に合わせて指定する。WebGPUは番号省略時に候補が1つの場合だけ自動選択する。`--target-vendor`で候補を絞ることもできる。DirectMLで製造元を検証するときは、PowerShellが別の`venv_webgpu`でアダプターを列挙する。

Windowsの`-Gpu`は従来どおりDirectML。`-Provider`でCUDA／TensorRT／OpenVINO／WebGPUも選択できる。CUDA 12対応のドライバーが必要で、Windows TensorRTは別途TensorRT 10のランタイムを配置する。`-TensorRtLibDir`／`--tensorrt-lib-dir`または`TENSORRT_LIB_DIR`で指定でき、複数候補がある場合は明示指定が必要。Linuxの既存ARM64経路は維持する。`--provider migraphx`は対応GPU・ROCm・wheelが揃った環境に限り、利用不可ならCPUに切り替えず停止する。

Pythonを直接起動する場合も同名の長いCLI引数を使えるが、依存環境とLinuxの共有ライブラリ検索パスは自分で準備する。通常は上記ランチャーを使う。ServerではBashの`--server`またはPowerShellの`-Server`と組み合わせる。Clientはサーバー側で推論するためローカルGPU初期化を行わない。

## 検証範囲

`python -m unittest discover -s tests -p "test_*.py"`で既存34件＋追加21件の全55件が成功。追加のGPUランタイム回帰テストで、DLLの事前ロードと保持、Intel/NVIDIA取り違え、DXGIとWebGPU番号の混同、WebGPU複数候補、TensorRT欠落、CPU fallback、ノード未実行を確認する。既存のモデル・前処理・Client/Server・整理処理テストも実行した。Bashランチャーの実行テスト（依存環境はスタブ）でEP・デバイス番号・空白入りパス・終了コードの受け渡しを確認し、Pythonのコンパイル、Bash/Zshの構文検証、Python本体とベンチマークの`--help`も成功した。

この作業環境にはGPUデバイスとPowerShellがないため、Windows DLLの実ロード、PowerShellの実行、実GPUでの再推論・性能比較は未検証。保存済みベンチマークは変更前の測定記録として維持し、今回の変更で再検証済みとは扱わない。

## 実機検証の記録と引き継ぎ

- [このPCで実施した検証](LOCAL_VALIDATION.md): 保存済みbalancedの実CPU推論・通常CLI・Server/Client・ベンチマークCPU経路。
- [Windows＋NVIDIA検証手順](WINDOWS_NVIDIA_VALIDATION.md): 通常処理20条件、Server/Client、異常系と提出物。
