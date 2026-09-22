# Windows セッションへの指示書：NVIDIA GPU で WebGPU と CUDA を比較

このファイルを Windows 側の新しい AI セッションの最初の指示として渡す。作業は Windows → Linux の順に別セッションで行う。Windows の結果と共通測定コードをコミット・push してから、Linux セッションへ `WEBGPU_BENCHMARK_LINUX.md` を渡す。

## 目的

同じ NVIDIA GPU、同じ ONNX モデル、同じ入力とバッチ条件で、Windows の WebGPU EP と CUDA EP の推論速度を比較する。結果は `WebGPU / CUDA` の所要時間比と、WebGPU の相対的な速度低下率で示す。既存の DirectML EP も参考値として測ってよいが、CUDA との比率を主結果にする。Windows と Linux の絶対速度を直接割って「WebGPU のオーバーヘッド」と呼ばない。

## 開始時に確認すること

1. `git status --short`、branch、HEAD、remote を記録する。既存変更は保持する。`AGENTS.md` があれば読む。
2. Windows のバージョン、CPU、NVIDIA GPU、ドライバー、CUDA、Python、ONNX Runtime、CUDA EP、DirectML EP、WebGPU EP、利用した Dawn backend（D3D12 または Vulkan）を記録する。backend を特定できなければ不明と明記する。
3. 現在の `run_tagger.ps1` は `-Gpu` で DirectML を入れるが CUDA/WebGPU 用環境を作らない。`embed_tags_universal.py` には `--webgpu` と plugin EP 登録がある。Windows 用に明示的な `-WebGpu` 経路を追加し、DirectML 環境を上書きしない専用 venv を使う。CUDA の比較環境も別 venv にする。PowerShell の既存引数と CPU/DirectML 経路を保つ。

## 共通測定コード

Windows セッションで、Linux でも同じ条件で使える独立したベンチマークコードをリポジトリへ追加する。既存の `benchmark_model_vram.py` は DirectML 前提、`nvidia-smi` 前提、反復3回、warmup と計測の分離なしなので、その数値を今回の比較に流用しない。

- 明示的に `cuda`、`webgpu`、参考値用 `directml` を選べるようにする。指定 EP が active にならない場合は失敗として停止する。CUDA 測定時は TensorRT を使わない。
- 同じ `DBV4Metadata` と `DBV4Preprocessor`、ONNX ファイル、float32 入力、出力を使う。推論だけを計時し、モデル取得、セッション作成、初回コンパイル、画像読み込み、前処理、XMP、レポート生成を計時から除く。それらの初期化時間は別列に記録する。
- 入力は seed 固定の合成 RGB 640×480 画像から作る。各 profile と batch size で一度前処理した同一バッファを両 EP へ渡す。実画像ディレクトリを探索しない。出力の最大絶対差も確認する。
- 少なくとも `wd14_v3` と `balanced`、batch size 1 と 4 を試す。モデルが認証・VRAM・演算子の理由で使えない場合は理由を記録し、成功した profile だけで比較する。比較条件を後から変えたら両 EP を再測定する。
- 各組み合わせで warmup を最低3回、計測を最低20回、測定セットを3回行う。同期的な `session.run` の壁時計時間を記録し、各セットの中央値と p95、全セットの中央値、ms/image、images/s を出す。EP の測定順を交互に変え、電源モード、温度、他アプリの負荷をできるだけ一定にする。
- `nvidia-smi` で測定前・セッション常駐・推論中ピークの VRAM を同じサンプリング間隔で記録する。WebGPU で `nvidia-smi` の値が取れない場合は欠測として扱う。
- ONNX Runtime profiling でノードの provider 割り当てを集計する。active provider の表示だけを GPU 実行の証拠にしない。CPU fallback の割合と、その制約を結果へ添える。

## 実施と成果物

1. Windows の CUDA と WebGPU で実モデルの推論、出力比較、測定を行う。余力があれば DirectML を参考測定する。WebGPU が失敗した場合は失敗ログと model/profile を残し、再現できる原因だけ修正して再試験する。
2. 環境、全測定条件、生データ、集計、計算式（`低下率 = (WebGPU_ms / CUDA_ms - 1) × 100%`）を `benchmarks/` 下の機械可読ファイルと Markdown に保存する。GPU ごとの測定であることを明示する。
3. Windows 用変更について既存 unit test、PowerShell 構文、変更した実行経路を確認する。README と changelog に結果の参照先と制約を追記する。
4. 変更をコミットして通常 push する。force push はしない。Linux 指示書にコミット SHA、測定コマンド、未解決事項を追記してから push する。

## Linux セッションへの引き継ぎ

Linux 側には、push 済み branch と commit SHA、ベンチマークコード、入力生成条件、成功した profile と batch size、Windows の生データの場所を渡す。Linux 側で Windows の測定値を更新しない。
