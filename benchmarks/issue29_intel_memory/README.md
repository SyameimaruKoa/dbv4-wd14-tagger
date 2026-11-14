# Issue #29: ultraとIntel共有メモリ

追加の[a13m・Windowsネイティブ／WSLCコンテナ実験](a13m_20261008/README.md)でも4枚時のメモリ圧迫とWindowsページファイル増加を再現した。両環境で1枚への制限後の余裕と数値一致を確認。WSLC内のswapが0でもWindows側でページングが発生するため、Linux指標だけで判定しない。

2026-10-08、OptiPlex-3040-MFF、Intel HD Graphics 530 (iGPU)、RAM 15,377MiB、Linux 7.0.0-38-generic、Python 3.13、onnxruntime-openvino 1.24.1／OpenVINO 2025.4.1。既存の約2.58GiBのultra ONNX外部重みを使用。GPU.0の実ノード実行をORTプロファイルで確認した。

## 調査と対策

ultraはFP32重みだけで約2.58GiBある。ORTとOpenVINOの初期化・GPU作業用メモリもRAMを消費し、バッチ増加と並列要求は中間テンソルの保持量を増やす。共有メモリではRSSだけでGPUドライバーの使用量を表せない。

- Intel providerのグラフ最適化をOpenVINOへ任せ、ORTのCPUアリーナ・メモリパターンを無効化する。
- GPUはFP32を維持し、LATENCY／NUM_STREAMS=1を明示する。
- Intel providerのultraは動的バッチを1枚へ制限する。CLIがバッチを調整し、Serverは同じ上限をClientへ通知する。固定バッチ入力はモデルの仕様を優先する。
- ultraの推論ロックは画像入力・前処理済み入力の両方へ適用し、複数Clientが同時にsession.runへ入ることを防ぐ。通信worker数は維持する。

ORTのグラフ最適化を無効にする設定は[公式OpenVINO EP文書](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#onnxruntime-graph-level-optimization)に沿う。ストリームごとの中間バッファについては[OpenVINO文書](https://docs.openvino.ai/2023.3/openvino_docs_deployment_optimization_guide_tput_advanced.html)を参照。これらの設定だけでRAMが減ると断定せず、バッチと推論同時数を制限する。

## 同じ1枚の比較

各条件を別プロセスで実行。NCHW・512角・seed=29の合成テンソル1枚を3回推論し、50ms間隔でRSS、プロセスswap、システム空きRAM・swapを採取した。ロードを含むピークであり、全工程の主プロセス・ドライバー・GPU共有メモリを分離した厳密な帰属計測ではない。ロード完了の値は完了マーカーを最初に観測した時点なので、最初の推論開始後になりうる。

| 指標 | 変更前・batch 1 | 変更後・batch 1 |
|---|---:|---:|
| モデルロード時間 | 22.83秒 | 15.08秒 |
| 初回推論 | 12.72秒 | 18.44秒 |
| 2／3回目 | 5.43／5.12秒 | 6.13／5.11秒 |
| プロセスRSSピーク | 7,516.76MiB | 8,301.46MiB |
| プロセスswapピーク | 952.51MiB | 330.75MiB |
| システムswap増分最大 | 4,424.20MiB | 2,415.93MiB |

全12,476出力が有限で、変更前後の最大絶対差は0。これは合成入力の数値回帰検証であり、実画像のタグ品質・XMP一致試験ではない。[集計JSON](batch1/summary.json)、[変更前の時系列](batch1/baseline/samples.json)、[変更後の時系列](batch1/current/samples.json)を保存した。

他のサービスが稼働し、変更前の測定で他プロセスのページがswapへ移動した。開始時の空きRAMは変更前6,785MiB／変更後10,056MiBと異なる。RSS増加はswapとの交換も含むため、この順序比較から総メモリ削減率・速度改善を算出できない。1枚でも約8GiBのRSSとドライバー使用量が残り、16GiB環境のスワップ解消を保証しない。

## 従来の4枚バッチ

同じ入力を4枚に複製した変更前設定で再現。ロードを含む3回の推論測定が2分27秒経っても完了せず、メモリ圧迫のため停止した。停止直前のプロセスswapは5,474MiB、システム全体のswap使用量は20,522MiB、空きRAMは1,291MiB。RSS時点値は1,463MiBで、ピークではない。GPU共有メモリはこのRSSにすべて含まれない。全体swapには他のサービス・前の測定の影響がある。

[停止時のスナップショット](batch4_pressure.json)に/procの値を保存した。4枚の出力は得られていないため、4枚対1枚の出力・速度・ピークメモリ比較は未完了として扱う。4枚バッチを避ける対策を採用した根拠は、このメモリ圧迫の再現と、1枚推論の完了・数値一致である。

## 再測定

既存のIntel用仮想環境で、OpenVINOの共有ライブラリのパスを起動時と同じように設定して実行する。モデル取得・変換は行わない。

```bash
export LD_LIBRARY_PATH="$PWD/venv_intel/lib/python3.13/site-packages/openvino/libs:${LD_LIBRARY_PATH:-}"
venv_intel/bin/python benchmark_intel_memory.py models/53e96223459a3046/model.onnx --output benchmarks/intel_memory_repeat
```

`--baseline-batch-size 4`で従来の既定バッチを再現できるが、今回の環境では大量のswapが発生した。`--timeout`（既定900秒）で各条件の実行時間を制限できる。出力・プロファイル・ログも実行先へ保存し、失敗時には診断ログを残す。システム指標の条件を揃えるには、他サービスや既存swapの影響を減らした環境で測定する。

最新mainへ変更を適用後、155件の回帰テストが成功（環境条件により21件スキップ）。追加テストはIntel設定、動的バッチ上限、Serverへの通知、同時要求の直列化、他provider・profile・固定入力の上限維持を確認した。
