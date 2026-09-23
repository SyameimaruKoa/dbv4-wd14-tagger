# 本体反映後のローカル検証結果（2026-09-23）

対象は[GPU初期化修正](GPU_RUNTIME_PORT.md)。[Windows＋NVIDIAの実機検証手順](WINDOWS_NVIDIA_VALIDATION.md)は別文書。

## 実行環境

- この作業のLinux実行環境: kernel 7.0.0-31-generic / x86_64 / glibc 2.43、Python 3.13.15。
- `.venv_bench_cpu/bin/python`、ONNX Runtime 1.30.0。
- 列挙EP: `AzureExecutionProvider`, `CPUExecutionProvider`。実推論で使用したEPはCPUのみ。
- この実行環境には`/dev/dri`、`nvidia-smi`、PowerShell（`pwsh`）がない。ホスト全体のハードウェア構成を断定するものではない。
- 保存済み`animetimm/caformer_b36.dbv4-full`（balanced）を利用。metadata version `8c7ba23eaed2d01d`。
- 実モデル試験は`HF_HUB_CACHE=/opt/wd14-tagger-xmp/.hf-cache/hub`と`HF_HUB_OFFLINE=1`で実行。追加ダウンロードなし。入力は検証用の640×480合成PNG。

## 結果

| 項目 | 実施内容 | 結果 |
| --- | --- | --- |
| 回帰テスト | `python -m unittest discover -s tests -p 'test_*.py'` | 55件成功、skipなし |
| Python構文 | 本体、共通GPU処理、ベンチマーク、testsを`compileall` | 成功 |
| Bash/Zsh構文 | 本体ランチャーと補完を`bash -n` / `zsh -n` | 成功 |
| CLIヘルプ | Python本体・ベンチマークの`--help` | ともに終了0 |
| Bashランチャー | テスト用環境でCUDA/Intel/WebGPUの引数・空白入りパス・終了コードを確認 | 成功。依存環境はスタブ |
| 通常本体の実CPU推論 | `load_runtime_model(..., provider='cpu')`からbatch 1・4で`predict_images` | `(1,12476)` / `(4,12476)`。全値が有限かつ0～1 |
| 通常CLIの実CPU推論 | balanced、batch 1、合成PNG1枚、`--no-tag --no-report --force` | 終了0、推論1枚、スキップ0枚 |
| ベンチマークCPU経路 | balanced、batch 1、warmup 1、iterations 2、sets 1 | 終了0、JSON `status=ok`、出力`[1,12476]`、CPUノード実行確認 |
| 実CPU Server/Client | localhostでCPU Serverを起動し、`/metadata`確認後にClientから合成PNG1枚を送信 | モデル一致、12,476ラベル、Client終了0・推論1枚。Serverは試験後停止 |
| 利用不可EPの異常系 | CPU専用環境で本体CLIに`--provider cuda` | 終了1、`CUDAExecutionProvider unavailable`。画像推論へ進まず停止 |
| 差分・文書リンク | `git diff --check`と追加・更新MDのローカルリンク | 成功 |

Server/Clientのテストはlocalhostソケットが必要なため、サンドボックスの通信制限の外で実行した。Windows DLL・デバイス選択などの回帰テストはモックであり、GPU実機試験ではない。

### 実行例

リポジトリ直下で実行したコマンド。画像と出力は`/tmp`に置いた。

```bash
export HF_HUB_CACHE="$PWD/.hf-cache/hub"
export HF_HUB_OFFLINE=1
.venv_bench_cpu/bin/python embed_tags_universal.py \
  --provider cpu --model-profile balanced --batch-size 1 \
  --no-tag --no-report --force /tmp/gpu-port-smoke.png

.venv_bench_cpu/bin/python benchmark_nvidia_ep.py \
  --provider cpu --profile balanced --batch-size 1 \
  --warmup 1 --iterations 2 --sets 1 \
  --output /tmp/gpu-port-benchmark-smoke.json
```

## 未実施・解釈上の範囲

- Windows PowerShellの実行とWindows DLLの実ロード、NVIDIA/CUDA/TensorRT/DirectML/WebGPUの実GPU推論は未実施。
- 実モデル試験は保存済みbalancedのみ。wd14_v3と他プロファイルの今回の再推論は未実施。
- 実CPU試験では`--no-tag --no-report`を指定したため、実ExifToolによるXMP書き込みやHTML生成の動作証明ではない。これらはWindows実機手順で別途確認する。
- 短いCPU試験は動作確認であり、性能比較の測定値として採用しない。既存のベンチマークJSONは変更していない。
- ORTからtelemetry ID保存不可の警告が出たが、メモリ内IDで続行し、上記試験はいずれも想定どおり完了した。
