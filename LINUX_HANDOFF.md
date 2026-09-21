# Linux動作検証セッションへの引き継ぎ指示

以下を新しいCodexセッションの最初の指示として使用すること。

---

## 目的

`wd14-tagger-xmp`のDBV4移行版について、Windowsで完了した実装を変更しすぎず、Linux上で実際にセットアップ・推論・整理機能を検証し、Linux固有の問題があれば修正して既存PRへ追加する。

## リポジトリとGit

- Repository: `SyameimaruKoa/wd14-tagger-xmp`
- Branch: `feat/dbv4-migration`
- Pull Request: <https://github.com/SyameimaruKoa/wd14-tagger-xmp/pull/4>
- Windows検証完了時の先頭commit: `ecb9dc0 feat: restore WD14 and document model selection`
- Force pushは禁止。
- このリポジトリについてはForce push以外の通常のcommit、push、依存関係導入、テスト、実装変更を許可済み。
- 作業開始時に`git status`、現在branch、`git log -5 --oneline`を確認する。ユーザーの既存変更があれば消さずに保護する。
- PRを新規作成しない。既存PR #4のbranchへ通常pushする。

## 絶対禁止事項

- Windows側の`C:\Users\kouki\Downloads\tast`には絶対にアクセスしない。列挙、検索、読み取り、hash計算も禁止。ユーザーが貼り付けたログだけを情報源として扱う。
- `git reset --hard`、無断のファイル削除、Force pushをしない。
- 動いているWindows用経路をLinux対応の都合で壊さない。
- モデル名や前処理を推測で決めない。Hugging Faceの実ファイル、モデルカード、公式推論コードを確認する。

## コーディング規則

- リポジトリの`AGENTS.md`を最初に読む。
- インデントはスペース4つ。
- Linux shell scriptはUTF-8で保存する。
- shell scriptを追加・変更する場合、`show_help()`による`-h`または`--help`を維持する。引数なしで実行不能なscriptもhelpを表示する。
- Python追加packageは必ず仮想環境へ入れる。
- 既存コードの省略、削除、無関係な整形や改変をしない。
- 日本語ログが文字化けしていないことを実際の出力で確認する。

## 現在の実装状態

主要機能はWindows 11・RTX 2070 Max-Q・DirectMLで動作確認済み。

- DBV4 metadata駆動推論
- モデル固有`preprocess.json`
- tag単位`best_threshold`
- Rating / General / Character分離
- R-00、R-15_0～4、R-17_0～4、R-18
- XMP保存とscore再利用
- Pixivフォルダ単位整理
- Server / Client metadata整合性検査
- batch推論、前処理worker、ウォームアップ、速度表示
- Hugging Face gated modelのブラウザログイン／同意ページ表示
- 外部ONNX dataを同一ディレクトリへmaterializeするultra対応
- 旧WD14 V3互換profile

利用可能profile:

- `lightweight`: `animetimm/mobilenetv4_conv_aa_large.dbv4-full`
- `balanced`: `animetimm/caformer_b36.dbv4-full`（既定）
- `high`: `animetimm/eva02_large_patch14_448.dbv4-full`（遅いため新規利用非推奨、互換用）
- `ultra`: `itterative/convnextv2_huge.dbv4-full-onnx`
- `wd14_v3`: `SmilingWolf/wd-swinv2-tagger-v3`
- `compact_manual`: 管理者承認待ち
- `medium_manual`: 管理者承認待ち
- `future_1b`: 将来予約。公式ONNX未公開のため意図的に実行不可。

Windows側では23件のunit testが成功済み。Windows実測値はREADMEに記録済みなので、Linux値と混ぜず、別表または明確な列で記録する。

## Linux検証の優先順位

### 1. 基本状態

~~~bash
git status --short
git branch --show-current
git log -5 --oneline
bash -n run_tagger.sh
./run_tagger.sh --help
python3 --version
~~~

`--help`の日本語、profile一覧、終了コードを確認する。

### 2. CPUセットアップとテスト

~~~bash
./run_tagger.sh
./venv_std/bin/python -m unittest discover -s tests -v
./venv_std/bin/python -m py_compile dbv4.py embed_tags_universal.py benchmark_model_vram.py
git diff --check
~~~

引数なし実行はセットアップ動作として許可されている。Python 3.14以上を避けるfallback、仮想環境作成、依存導入が正しく動くか確認する。

### 3. CPU推論

ユーザーがLinux側で明示した検証用画像ディレクトリだけを使用する。許可されていない実データを勝手に探さない。

最低限、次を確認する。

~~~bash
./run_tagger.sh --model-profile lightweight --force /path/to/test-images
./run_tagger.sh --model-profile balanced --force /path/to/test-images
./run_tagger.sh --model-profile wd14_v3 --force /path/to/test-images
~~~

- モデルとmetadataの取得
- 入力shape、出力label数、active provider
- WD14がNHWC、白背景square padding、BGR、0～255入力で動くこと
- XMP書き込み
- 2回目の整理時にscore再利用されること
- 日本語進捗表示と完了summary

### 4. GPU経路

最初にハードウェアとdriverを実測する。

~~~bash
lspci | grep -Ei 'vga|3d|display'
nvidia-smi
./run_tagger.sh --gpu --debug --model-profile lightweight --force /path/to/test-images
~~~

環境に応じて以下を確認する。

- NVIDIA: CUDA Execution Providerが本当にactiveになっていること。CPU fallbackを成功扱いしない。
- Intel: OpenVINOが導入され、既定`GPU.0`が利用できること。`--debug`時だけ詳細診断が出ること。
- AMD: 現在のwrapperが選択するproviderと実際のactive providerを照合する。
- `patchelf --clear-execstack`処理とUbuntu 24.04向け案内が正しいこと。
- GPU初回compile/warmupと通常推論を分けて確認する。

Linux GPUでVRAMを測定する場合、Windows表を上書きしない。GPU名、driver、provider、batch-size、測定前、常駐、ピーク、差分、ms/imgを一組としてREADMEへ追記する。`benchmark_model_vram.py`は現在`nvidia-smi`前提なので、NVIDIA以外で使う場合は既存動作を壊さず測定backendを追加する。

### 5. Pixiv整理

コピーした小規模な検証データで行い、原本を直接整理しない。

~~~bash
./run_tagger.sh --gpu --pixiv --force /path/to/copied-test-images
./run_tagger.sh --gpu --pixiv /path/to/copied-test-images
~~~

- 画像を含む末端／親フォルダが対象になること
- R-17以上を含む画像フォルダ全体が代表ratingへ移動すること
- 2回目は保存scoreを利用して推論数が0になること
- 空になった元フォルダだけ削除されること
- 進捗postfixの推論数、skip数、速度が正しいこと

### 6. Server / Client

可能ならlocalhostで最低限検証する。

~~~bash
./run_tagger.sh --server --model-profile lightweight
./run_tagger.sh --client --host 127.0.0.1 --model-profile lightweight /path/to/test-images
~~~

- Clientがモデル本体をdownloadしないこと
- model ID、metadata version、output size不一致を拒否すること
- 画像単位HTTPエラーはその画像だけskipし、接続断だけ全体中断すること

## 既知の注意点

- `compact_manual`と`medium_manual`はHugging Face管理者承認待ち。未承認停止は不具合ではない。
- `future_1b`は公式repoに`model.onnx`がないため、理由付き停止が正しい。
- `high`のEVA02を別モデルへ安易に交換しない。公開済み候補を調査した結果、balancedより高精度かつEVAより十分軽い代替ONNXは見つかっていない。
- `ultra`は`model.onnx`と`model.onnx_data`を同じ実ディレクトリに置く必要がある。Linuxのsymlinkで動いてもWindowsのmaterialize処理を削除しない。
- WD14 V3とDBV4のrating score分布を同一視しない。
- 実画像のタグ品質評価と単なる起動成功を区別する。

## 修正時の完了条件

1. 問題をログと再現手順で特定する。
2. Linux固有修正はWindows経路への影響を確認する。
3. `bash -n run_tagger.sh`、全unit test、`py_compile`、`git diff --check`を通す。
4. 変更した実行経路を実機で再試験する。
5. READMEとchangelogへ、実測環境・結果・制約を記載する。
6. commitして`feat/dbv4-migration`へ通常pushする。Force pushは禁止。
7. 最終報告では、成功項目、未検証項目、実測provider、問題と修正、test結果、commit SHAを明記する。

## 最初の報告

作業開始後、まず次を簡潔に報告すること。

- Linux distributionとversion
- kernel
- CPU / GPU
- Python version
- 現在branchとHEAD
- 最初に検証するprovider

調査だけで止めず、許可範囲内で実行、修正、再検証、commit、pushまで進めること。
