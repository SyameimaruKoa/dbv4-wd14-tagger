
# 実装履歴

## 2026-08-15

- Nintendo Switch (Tegra X1 / Switchroot L4T) および ARM64 環境への対応
  - `run_tagger.sh`: Tegra SoC (`/dev/nvhost-gpu`, `/etc/nv_tegra_release`, `/usr/lib/aarch64-linux-gnu/tegra` 等) の自動検出を追加し、Nintendo Switch や Jetson 等の統合 GPU 環境で NVIDIA GPU を正しく認識するように改善。
  - `run_tagger.sh`: Tegra L4T ドライバからの CUDA バージョン判定ロジックを追加。
  - `run_tagger.sh`: ARM64 (aarch64) 環境下での依存ライブラリインストール処理を最適化し、x86_64 専用パッケージによるエラーを防止。
  - `run_tagger.sh`: Tegra ドライバ格納パス (`/usr/lib/aarch64-linux-gnu/tegra`) を `LD_LIBRARY_PATH` に自動追加。
  - `embed_tags_universal.py`: ARM64 / Tegra 等で利用可能な GPU プロバイダが存在しない場合に、分かりやすい案内を出力して CPU (ARM NEON) モードへフォールバックする機能を追加。
  - `README.md`: Switchroot Ubuntu 24.04 の仕様・制限事項（CUDA 10.0 ランタイムあり、CUDA コンパイラ・cuDNN なし）と、メイン PC と連携した高速 GPU クライアント/サーバー利用方法をドキュメント化。

## 2026-08-01

- NVIDIA GPU (CUDA / TensorRT) 高速化機能の追加および動作の最適化
  - `run_tagger.sh`: システムの CUDA バージョンに合わせた `nvidia-*` ランタイムパッケージおよび ONNX Runtime 互換の `tensorrt<11` (10.x系) パッケージの自動検出・自動インストールロジックを追加。
  - `run_tagger.sh`: `LD_LIBRARY_PATH` の動的生成ロジックを改善し、`site-packages` 内の NVIDIA / TensorRT ライブラリディレクトリを正確に追加して共有ライブラリのロードエラーを解決。
  - `embed_tags_universal.py`: コンパイルを伴う ExecutionProvider (`TensorrtExecutionProvider` 等) 使用時の注意事項出力機能を追加。
  - `embed_tags_universal.py`: 実効バッチサイズに対応したダミーテンソルによる事前ウォームアップ推論（エンジン自動構築）を追加し、ウォームアップ所要時間をログ出力。
  - `embed_tags_universal.py`: 1枚目（初回バッチ）の処理時間に関する外れ値判定ロジックを組み込み、コンパイルによる遅延を選択的に除外した実効推論速度の報告、および1枚目処理時間・全体の詳細サマリー出力を追加。

## 2026-05-15 (Update 2)

- デフォルトモデルを最新のV3系 (`SmilingWolf/wd-swinv2-tagger-v3`) に変更しました。
  - `embed_tags_universal.py` の `DEFAULT_CONFIG` を更新。
- `README.md` に現在利用可能な SmilingWolf 氏のモデル一覧（V3、Large V3、V2）とそれぞれの特徴を追加しました。
- READMEの推奨モデルの記述を、最新モデルに合わせて更新しました。

## 2026-05-15

- `avif` 画像フォーマットへの対応を追加しました。
  - `embed_tags_universal.py`: `VALID_EXTS` に `.avif` を追加。PillowがAVIFを読み込めるように `pillow_avif` のインポート処理を追記。
  - `run_tagger.ps1`, `run_tagger.sh`: 環境構築時のpipインストール対象に `pillow-avif-plugin` を追加。
- パッケージインストール処理をスマート化（リファクタリング）
  - 共通ライブラリを `requirements.txt` に分離。
  - カスタムインストールロジック（`pip list` との突き合わせなど）を廃止し、pip標準の依存関係解決と `-r requirements.txt` を活用する形に変更。
- GPUの遊休時間を減らすため、バッチ推論と前処理並列化を追加しました。
  - `embed_tags_universal.py`: `--batch-size` と `--io-workers` を追加し、ローカル推論時にまとめて処理できるように変更。デフォルトでバッチサイズ 4、IO ワーカー自動計算で並列化を有効化。モデルがバッチ非対応の場合は自動で 1 にフォールバック。
  - `run_tagger.ps1`, `run_tagger.sh`, `README.md`: 新しいオプションを追記し、デフォルト値とフォールバックの説明を追加。
- モデル/タグの切り替え機能を追加しました。
  - `embed_tags_universal.py`: `--model-repo`, `--model-file`, `--tags-file` を追加し、config.json と CLI から切り替え可能に変更。
  - `run_tagger.ps1`, `run_tagger.sh`, `README.md`, `config.json`: 新しい項目とヘルプを追記。
- README に推奨モデルを追記しました。
- README にバッチ推論対応モデル（swinv2）の推奨を追記しました。
- デフォルトモデルを `SmilingWolf/wd-v1-4-swinv2-tagger-v2` に変更しました。
