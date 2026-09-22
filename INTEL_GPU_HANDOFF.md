# Intel GPU動作検証セッションへの引き継ぎ指示

この文書を新しいCodexセッションの最初の指示として渡す。目的は、既存のNVIDIA/Windows動作を維持したまま、Linux上でIntel GPUのOpenVINO推論を実際に動かし、問題があれば最小限の修正・再検証を行うこと。調査だけで止めず、可能な範囲で実機推論まで進める。

## リポジトリと保護事項

- 作業場所: `/opt/wd14-tagger-xmp`
- Repository: `SyameimaruKoa/wd14-tagger-xmp`
- Branch: `feat/dbv4-migration`
- 既存PR: <https://github.com/SyameimaruKoa/wd14-tagger-xmp/pull/4>
- この指示書作成時のHEAD: `06ea678 feat: align client metadata with server model`。作業開始時に実際のHEADを再確認すること。
- まず `git status --short`、`git branch --show-current`、`git log -5 --oneline`、リポジトリ内の`AGENTS.md`を確認する。ユーザー変更と稼働中のServerを保護する。
- Force push、`git reset --hard`、無断の削除は禁止。既存PR以外を新規作成しない。pushは宛先と権限を確認してから通常pushのみ。
- Windows側の `C:\Users\kouki\Downloads\tast` は列挙・読取り・検索・hash計算を含め一切アクセスしない。
- Python packageは仮想環境へ導入する。Linux shell scriptを変更する場合はUTF-8、スペース4つのインデント、`show_help()`による`-h`/`--help`を維持する。既存コードの省略・削除・無関係な改変は禁止。
- ユーザーが指定していない実画像は探索・処理しない。最初は `/tmp` に作る合成画像だけを使用し、タグ品質の評価とは区別する。

## 開始前に確認済みの実測値

2026-09-21/22時点、このセッションのホストはUbuntu 26.04.1 LTS、kernel `7.0.0-31-generic`、Intel Core i7-1355U、Raptor Lake-P Iris Xe Graphics `8086:a7a1`。`/dev/dri/renderD128` が存在し、ユーザーは `render` と `video` グループに所属。`clinfo` はIntel OpenCL Graphicsを1 platformとして認識した。

システムPythonは3.14.4、`python3.13`は3.13.15。既存の`venv_intel`はPython 3.13で、`onnxruntime-openvino 1.24.1`を含み、ONNX Runtimeのprovider一覧は `OpenVINOExecutionProvider` と `CPUExecutionProvider`。ただし、これは`GPU.0`での実推論成功を意味しない。`import openvino`は現状失敗するので、OpenVINO Python APIを診断に使うなら必ず`venv_intel`へ追加する。`intel-opencl-icd`と`libze1`は導入済み、`intel-level-zero-gpu`は未導入。driverの追加・変更はエラーの根拠を確認してから行う。

Codexの通常サンドボックス内では`/dev/dri`が見えない場合がある。ホスト通常端末または承認された権限で実測し、サンドボックス制限による失敗をIntel GPUの不具合と混同しない。

## 優先手順

1. 状態・driver確認: `lspci -nn | grep -Ei 'vga|3d|display'`、`ls -l /dev/dri`、`id`、`clinfo`、`./venv_intel/bin/python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"`。kernel log、OpenCL/Level Zero package、OpenVINO GPU pluginのロードエラーを必要に応じて確認する。providerの列挙だけで成功判定しない。
2. `bash -n run_tagger.sh`と`./run_tagger.sh --help`を確認する。ヘルプの日本語と`--force-intel`、`--debug`を確認する。`config.json`の`openvino_gpu_device`既定は`GPU.0`。NVIDIA自動検出に依存せず、Intel検証では必ず`--force-intel`を指定する。
3. 合成PNGを`/tmp`の専用ディレクトリに用意し、`./run_tagger.sh --force-intel --debug --model-profile lightweight --force --no-report /tmp/<検証用ディレクトリ>`で試す。既存HFログインを再利用する。入力shape、ラベル数、OpenVINO要求デバイス、実際のactive provider、推論完了、XMP書き込みをログとExifToolで確認する。`CPUExecutionProvider`だけならIntel GPU成功と報告しない。
4. `--debug`なしでも同じ合成画像で再試験し、通常時にOpenVINO内部診断が過剰表示されないことを確認する。可能なら`balanced`と`wd14_v3`も試す。`wd14_v3`はNHWC、白背景square padding、BGR、0～255入力であることをunit testと実推論で確認する。
5. 保存scoreを使う2回目の整理、必要ならコピーした小規模なPixiv検証データ、localhostのServer/Clientを確認する。原本や稼働中の別Serverは変更・停止しない。Server/Clientは`/metadata`でClientがモデルprofileを自動整合する新経路を確認する。
6. 問題が出たら再現コマンドとログから原因を絞り、Intel固有の最小修正を行う。特に`embed_tags_universal.py`のOpenVINO provider選択と`run_tagger.sh`のIntel仮想環境経路を調べる。Windows、NVIDIA CUDA/TensorRT、CPU、既存Clientの動作を変えない。必要なpackageを導入するときは`venv_intel`だけに入れる。

## 測定・完了条件

- 実測で`OpenVINOExecutionProvider`がactiveでも、実効デバイスがGPUとは限らない。`GPU.0`指定時のログ、OpenVINO診断、GPU使用状況を照合し、CPU fallbackを成功扱いしない。
- Intel GPUの共有メモリ使用量をNVIDIAのVRAMと同一視しない。性能を記録するなら、GPU名・driver・provider・実効デバイス・batch-size・初回compile/warmup・通常推論のms/imgを一組としてREADMEのWindows/NVIDIA表とは別に記録する。
- `benchmark_model_vram.py`は`nvidia-smi`前提なので、Intelで使う場合は既存NVIDIA動作を壊さず測定backendを追加する。必須でなければ使用しない。
- `bash -n run_tagger.sh`、全unit test、`./venv_std/bin/python -m py_compile dbv4.py embed_tags_universal.py benchmark_model_vram.py`、`git diff --check`を通す。変更した実行経路はIntel実機で再試験する。
- READMEとchangelogへ実測環境・成功項目・未検証項目・制約を記録する。日本語ログが文字化けしないことを実出力で確認する。commit後、既存PRのbranchへforceなしで通常pushする。最終報告は実効デバイス、provider、結果、問題と修正、テスト数、commit SHAを明記する。
