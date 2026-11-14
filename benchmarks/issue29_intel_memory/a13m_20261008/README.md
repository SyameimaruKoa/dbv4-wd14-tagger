# a13m: WindowsネイティブとWSLCコンテナの再現実験

2026-10-08 20:39〜20:47 JST、`ssh a13m`で実施。Windowsネイティブと、WSLC（WSL 3.0.1）のUbuntu 24.04コンテナで、ultra・OpenVINO FP32を測定した。WSLの製品バージョンが3.0.1で、カーネル／仮想化方式はWSL2。通常のUbuntuディストリビューションへ直接入れたPython環境はこの比較に使っていない。

**両環境で4枚バッチのメモリ圧迫とWindowsのページファイル増加を再現した。WSLC側ではLinux内のswapは0のままでも、Windows側の空きRAMが最少6MiBまで低下した。** 1枚への制限とOpenVINO設定変更で余裕が戻り、両環境の全12,476出力が許容誤差内で一致した。

## 条件

- MSI Prestige 13 A13M、Intel Iris Xe（PCI ID `0xa7a1`）、RAM 32,469MiB。Windows 11 Home、build 26300.9550、Intelドライバー32.0.101.7092。
- Windows: Python 3.13.15、NumPy 2.3.5、onnxruntime-openvino 1.24.1、OpenVINO 2025.4.1。
- WSLC: Ubuntu 24.04、Python 3.12.3、同じNumPy／ORT／OpenVINO版。カーネル6.18.40.1-microsoft-standard-WSL2。標準ContainerfileのIntel OpenCL ICDを使用し、`--gpus all`でIris Xeを認識。GPUはソフトウェア描画ではない。
- コンテナから見えるRAMは15,847MiB、swapは16,234MiB。cgroupのmemory／swap上限は`max`。これらはWindowsホストの物理メモリ・ページファイルと同じ指標ではない。
- 同じWindows上のONNXと外部重みをネイティブと読み取り専用コンテナマウントで共有。ファイル量とSHA256は[model-manifest.json](model-manifest.json)に記録。
- seed=29、RGB NCHW 512角の合成テンソル。同じ画像を変更前は4枚、変更後は1枚。各条件を別プロセスで起動し、各5回推論。初回はGPU準備を含むため、速度の参考値は残り4回の中央値／枚。
- 変更前: ORT既定のグラフ最適化・CPU arena・memory pattern、GPU FP32。変更後: ORT最適化・arena・patternを無効にし、GPU FP32・LATENCY・1ストリーム。バッチと設定の両方を変更した比較で、各要因の効果を分離した試験ではない。
- コードは`ee0e706`の実験用worktreeと測定スクリプトの追補。Windows用カウンター・子プロセス追跡・通常起動と同じDLL初期化を追加。コンテナ試行前に進捗・経過時間・失敗／timeoutの部分記録も追加した。GPU設定と入力生成は同じ。

## Windowsネイティブ

| 観測指標 | 変更前・4枚 | 変更後・1枚 |
|---|---:|---:|
| プロセスツリーworking setピーク | 16,036MiB（15.66GiB） | 8,400MiB（8.20GiB） |
| プロセスツリーprivate commitピーク | 16,636MiB | 8,748MiB |
| 対象PIDのGPU shared usageピーク | 15,094MiB | 7,971MiB |
| システム空き物理RAM最少・50ms目安の監視 | 16.78MiB | 8,417.65MiB |
| Windowsページファイル時点値: 開始→最大 | 160→1,038MiB | 983→983MiB |
| 初回推論・バッチ全体 | 19.59秒 | 4.01秒 |
| 2〜5回の中央値／枚 | 2,522ms | 2,582ms |

4枚は完走したが、RAMの余裕がほぼなくなり、ページファイル使用量が878MiB増えた。変更後はこの観測区間でページファイル使用量の追加増加なし。ただし前の試行で退避されたページは残っており、使用量が0になったという意味ではない。

生データは[summary.json](native/comparison/summary.json)、[Windowsホスト時系列](native/host-samples.jsonl)、各条件の[samples.json](native/comparison/baseline/samples.json)、[環境](native/host-environment.json)。WindowsのGPU shared usageとworking setは重複しうるため、足して「総RAM」にしていない。private commitも実際のディスクswap使用量ではない。

## WSLCコンテナ

| 観測指標 | 変更前・4枚 | 変更後・1枚 |
|---|---:|---:|
| Linux推論プロセスRSSピーク | 5,884MiB | 5,886MiB |
| Linuxプロセスswap／システムswap増分 | 0／0MiB | 0／0MiB |
| Windowsから見たWSLC VMのworking setピーク | 16,797MiB（16.40GiB） | 10,370MiB（10.13GiB） |
| Windowsシステム空き物理RAM最少 | 6MiB | 7,084MiB |
| Windowsページファイル時点値: 開始→最大 | 951→4,172MiB | 1,575→1,575MiB |
| 初回推論・バッチ全体 | 38.89秒 | 10.78秒 |
| 2〜5回の中央値／枚 | 2,589ms | 2,613ms |

4枚は完走したが、Windowsのページファイル使用量が3,221MiB増えた。**コンテナ内のRSSや`free`だけでは、Windows側のGPU共有メモリとページングを把握できない。** VM working setはVM全体のWindowsからの観測であり、Linux推論プロセス単体のRSSやWindowsネイティブのworking setと同一物として扱わない。WSLC試行は1コンテナずつ実行し、終了後にコンテナが残っていないことを確認した。

[コンテナの集計](container/comparison/summary.json)、[Windowsホスト時系列](container/host-samples.jsonl)、[GPU・ライブラリ環境](container/container-environment.json)、[clinfo](container/clinfo.txt)、[イメージ情報](container-image.json)を保存した。Windows側のGPUメモリをLinux PIDへ帰属できなかったため、コンテナのGPU shared usageは未取得。VM working setで代用したと偽らない。

## 数値照合と制約

各OS内の4枚→1枚の最大絶対差は`3.0517578125e-05`で、`rtol=1e-4, atol=1e-5`に合格。両OSの変更後1枚の出力は完全一致した。[OS間の照合](cross-os-output-comparison.json)も保存した。ORTプロファイルで全条件のOpenVINO実ノード実行を確認し、CPUへ落ちた結果はGPU成功にしていない。

測定は合成入力の診断で、実画像タグ・XMP・複数Clientの実機同時要求は対象外。1条件1プロセス・5回で、正式な複数set統計ではない。他のWindowsアプリは稼働したまま。ネイティブ試行の後半にはWSLCイメージ構築も進行したため、OS全体のメモリ差は本プログラムだけの厳密な帰属量ではない。変更後のページファイルには変更前の影響も残る。

Python側は50msの待機間隔に計測処理時間が加わる。WindowsのWMI監視は1秒待機＋クエリ時間で実際は約2〜3秒間隔。WMIの時点ピークと50ms目安の観測ピークは一致しなくても異常ではない。[ホスト集計](host-comparison.json)の工程境界は、ORTプロファイル名のセッション開始時刻で区切った（ネイティブ20:40:55／コンテナ20:46:21 JST）。

最初のWindows試行は測定側のDLL初期化不足によりOpenVINOを作成できずCPUへ落ちた。GPU実行確認で失敗になったため除外し、通常起動の`prepare_openvino`を測定へ適用してやり直した。Windowsのvenv launcherの子孫プロセスも追跡するよう修正し、launcherだけの小さなRSSをモデル使用量にしていない。

## 再現方法

[監視・起動スクリプト](scripts/run_issue29_container.ps1)の`-Mode native`／`-Mode wsl`を使用する。記録用スクリプトはこのPCの実験パス`C:\tmp\dbv4-issue29`と既存モデルパスを明示しており、別の保存先ではパスを合わせる。WSLC側の[実行スクリプト](scripts/run_issue29_container.sh)は実験専用ボリューム`dbv4-issue29-memory-workspace`を使用し、通常利用の作業ボリュームと区別する。

```powershell
wslc build --file C:\tmp\dbv4-issue29\containers\wsl\Containerfile --tag dbv4-tagger:issue29 C:\tmp\dbv4-issue29
powershell -NoProfile -ExecutionPolicy Bypass -File C:\tmp\dbv4-issue29\run_issue29_container.ps1 -Mode native
powershell -NoProfile -ExecutionPolicy Bypass -File C:\tmp\dbv4-issue29\run_issue29_container.ps1 -Mode wsl
```

推論ケースのtimeoutは各300秒。変更前が失敗／timeoutしても、その部分記録を残して変更後を測定する。WSLCは`--rm`で終了時にコンテナを削除し、再測定用イメージと実験用ボリュームは保持する。157件の回帰テスト成功（21件スキップ）に加え、Windows APIの実計測、Linux／Windowsの実GPU出力、実子プロセスの失敗・timeout記録を検証した。
