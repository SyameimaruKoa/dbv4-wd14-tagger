# DBV4 Tagger Universal

AnimeTimm / DeepGHS系の DBV4 fullモデルをONNX Runtimeで実行し、画像タグ付け・XMP保存・フォルダ整理・HTMLレポート・推論サーバー/クライアントを行うツールじゃ。

Windows (PowerShell) と Linux (Bash) に対応しており、DBV4のモデル・前処理・ラベルカテゴリ・タグ単位thresholdをモデルプロファイルから動的に扱う構成になっておるぞ。

## 特徴

- DBV4 fullの大規模マルチラベル出力をmetadataベースで復号
- selected_tags.csv の best_threshold をタグ単位で適用
- preprocess.json に従ったモデル固有前処理
- General / Character / Rating をmetadataから分離
- R-00 / R-15_0〜R-15_4 / R-17_0〜R-17_4 / R-18 を維持
- DBV4 rating scoreをXMPへ保存し、再整理時の再推論を省略
- 軽量〜超大型モデルを同じインターフェースで切り替え
- CPU / NVIDIA / Intel OpenVINO / AMD系 / DirectML の既存実行経路を維持
- Server / Client間でmodel ID・metadata version・output sizeを検証
- 既存のユーザータグを保持
- HTMLレポートとフォルダ整理を維持

## DBV4モデルプロファイル

| Profile | ONNX Repository | ONNX実ファイル | モデル容量 | 用途 |
| --- | --- | --- | ---: | --- |
| compact_manual | animetimm/repvit_m2_3.dbv4-full | [`model.onnx`](https://huggingface.co/animetimm/repvit_m2_3.dbv4-full/resolve/main/model.onnx) | 122,111,601 bytes（約116.5 MiB） | 承認制・省メモリ |
| lightweight | animetimm/mobilenetv4_conv_aa_large.dbv4-full | [`model.onnx`](https://huggingface.co/animetimm/mobilenetv4_conv_aa_large.dbv4-full/resolve/main/model.onnx) | 189,162,894 bytes（約180.4 MiB） | 軽量 |
| medium_manual | animetimm/convformer_s36.dbv4-full | [`model.onnx`](https://huggingface.co/animetimm/convformer_s36.dbv4-full/resolve/main/model.onnx) | 254,865,174 bytes（約243.1 MiB） | 承認制・軽量とbalancedの中間 |
| balanced | animetimm/caformer_b36.dbv4-full | [`model.onnx`](https://huggingface.co/animetimm/caformer_b36.dbv4-full/resolve/main/model.onnx) | 536,982,484 bytes（約512.1 MiB） | デフォルト |
| high | animetimm/eva02_large_patch14_448.dbv4-full | [`model.onnx`](https://huggingface.co/animetimm/eva02_large_patch14_448.dbv4-full/resolve/main/model.onnx) | 1,268,832,518 bytes（約1.18 GiB） | 高精度 |
| ultra | itterative/convnextv2_huge.dbv4-full-onnx | [`model.onnx`](https://huggingface.co/itterative/convnextv2_huge.dbv4-full-onnx/resolve/main/model.onnx) + [`model.onnx_data`](https://huggingface.co/itterative/convnextv2_huge.dbv4-full-onnx/resolve/main/model.onnx_data) | 324,944 + 2,770,470,128 bytes（合計約2.58 GiB） | 精度最優先 |
| wd14_v3 | SmilingWolf/wd-swinv2-tagger-v3 | [`model.onnx`](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/model.onnx) | 467,460,978 bytes（約445.8 MiB） | 旧WD14互換 |
| future_1b | animetimm/vit_giantopt_patch16_siglip_384.dbv4-full | ONNX未公開（`model.safetensors`のみ） | 4,734,142,376 bytes（約4.41 GiB） | 将来対応予約・現在実行不可 |

balanced は、精度とモデル規模のバランスから caformer_b36.dbv4-full をデフォルトとして使用する。

各プロファイルはモデル・タグCSV・前処理定義・カテゴリ定義・threshold定義を一つの論理単位として扱う。ultraだけはONNX変換済みrepoに前処理metadataがないため、metadataを`animetimm/convnextv2_huge.dbv4-full`から取得する。将来DBV4モデルを追加する場合も、このプロファイルへ定義を追加すれば共通推論経路を変更せず切り替えられる設計じゃ。

`compact_manual`と`medium_manual`はHugging Face管理者の承認後に利用できる。プリセット選択時に未ログインならブラウザOAuth認証を自動開始する。ログイン後にモデルファイルへのアクセス権を確認し、未承認・承認待ちの場合は対象モデルの申請・同意ページをブラウザで自動表示してから停止する。URLやCLIコマンドを手作業で探す必要はないが、申請ボタンの操作と管理者による承認待ちはHugging Face上で必要になる。

### 旧WD14 V3互換

DBV4移行前の既定モデル`SmilingWolf/wd-swinv2-tagger-v3`を`wd14_v3`として保持する。公式WD Tagger実装と同じく、白背景で正方形へpaddingし、448 × 448へBicubic resize、RGBからBGRへ変換した0～255のfloat32をONNXへ入力する。ラベルは10,861個で、先頭4個のratingも`selected_tags.csv`のcategoryから特定する。

~~~powershell
.\run_tagger.ps1 -g -ModelProfile wd14_v3 .
~~~

~~~bash
./run_tagger.sh --gpu --model-profile wd14_v3 .
~~~

これは旧結果との比較・再現用であり、通常は新しい12,476ラベルのDBV4を推奨する。WD14 V3にはタグ単位`best_threshold`がないため、明示的な`-Thresh`／`--thresh`がなければ従来値0.35を使う。

| Profile | Params | 入力解像度 | Macro@Best F1 | 選定理由 |
| --- | ---: | ---: | ---: | --- |
| compact_manual | 30.4M | 384 × 384 | 0.510 | lightweightの0.511とほぼ同等でONNXが約36%小さい |
| medium_manual | 63.5M | 448 × 448 | 0.532 | lightweightより高精度でbalancedより小さい中間候補 |

### DirectMLのVRAM目安

RTX 2070 Max-Q 8GB、DirectML、batch-size=4、合成640 × 480 RGB画像、3回推論での統一実測値。`nvidia-smi`を100ms間隔で監視し、測定前との差をモデル実行による増分とした。

| Profile | 入力解像度 | 測定前 | 常駐／ピーク | ピーク増分 | 推論時間 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lightweight | 448 × 448 | 843 MiB | 1,614／1,614 MiB | 771 MiB | 120.3 ms/img |
| balanced | 384 × 384 | 852 MiB | 2,051／2,051 MiB | 1,199 MiB | 458.8 ms/img |
| high | 448 × 448 | 847 MiB | 3,616／3,618 MiB | 2,771 MiB | 1,273.9 ms/img |
| ultra | 512 × 512 | 840 MiB | 6,570／6,583 MiB | 5,743 MiB | 1,735.3 ms/img |
| wd14_v3 | 448 × 448 | 1,087 MiB | 2,970／2,973 MiB | 1,886 MiB | 442.3 ms/img |

| 測定待ちProfile | 入力解像度 | 状態 | 暫定VRAM目安 |
| --- | ---: | --- | ---: |
| compact_manual | 384 × 384 | 管理者承認待ち | 約1～1.5GB（推定） |
| medium_manual | 448 × 448 | 管理者承認待ち | 約1.5～2.5GB（推定） |
| future_1b | 512 × 512 | ONNX未公開・RTX 2070では実行不可見込み | 未測定 |

> [!WARNING]
> VRAM使用量はGPU、DirectML／ONNX Runtimeのバージョン、ドライバー、batch-size、同時使用中のアプリによって変動する。ultraのピーク増分は5,743MiBだったため、8GB未満のGPUでは推奨しない。メモリ不足時は`-BatchSize 1`または`-BatchSize 2`を指定する。ただしbatch-sizeを下げてもモデル重み自体の常駐分は減らない。

承認待ち2プロファイルの値は、Hugging Faceで確認したFP32 ONNX容量、パラメータ数、公式入力解像度、および実測済みモデルから見積もった概算である。承認後は`benchmark_model_vram.py compact_manual`と`benchmark_model_vram.py medium_manual`で同じ条件を測定できる。

### 将来向け1B級

`future_1b`には`animetimm/vit_giantopt_patch16_siglip_384.dbv4-full`を予約した。公式値は1.2B parameters、2.3 TFLOPs、512角、Macro@Best F1 0.607で、ultraの0.611に近い。ただし2026-09-21時点の公式repoには4,734,142,376 bytesの`model.safetensors`とPyTorch重みしかなく、`model.onnx`は存在しない。このプロジェクトはONNX Runtimeを使うため、現在は理由を表示して停止する。公式ONNXまたは検証済み変換版が公開された時点で取得元と外部データ構成を確定し、16GB以上のGPUで実測して有効化する。

### highプロファイルの位置付け

`high`のEVA02は公式Macro@Best F1が0.599で、balancedの0.581より高い一方、統一条件のDirectML実測では1,273.9 ms/imgとなり、ultraの1,735.3 ms/imgに近い。そのため新規利用には非推奨とし、既存configと比較試験の互換性のため定義だけを保持する。

2026-09-21時点で公開済みDBV4を再調査したが、適切な代替はなかった。例えばConvFormer B36は181.5 GFLOPs・Macro@Best F1 0.574で、balancedの132.2 GFLOPs・0.581に両面で劣る。SwinV2 BaseもMacro@Best F1 0.575でbalanced未満、ViT GiantOptは2.3 TFLOPs・1.2B parametersでultraより重い。速度と精度の中間を埋める新モデルが公開されるまでは、`balanced`または精度最優先の`ultra`を選ぶ。

## 新しいモデルの探し方

Hugging Faceでは、まず次の検索語を使う。

~~~text
site:huggingface.co/animetimm ".dbv4-full" "Macro@Best"
site:huggingface.co/animetimm ".dbv4-full" "FLOPs / MACs"
site:huggingface.co "dbv4-full" ONNX "selected_tags.csv"
site:huggingface.co/SmilingWolf "tagger-v3" "model.onnx"
~~~

Hugging Face内では`animetimm` organizationのModelsを更新日順で確認し、`dbv4-full`、`Image Classification`、`ONNX`を手掛かりにする。旧WD14系は`SmilingWolf/wd-*-tagger-v3`を探す。検索結果やrepo名だけでは採用せず、次をすべて確認する。

1. モデルカードのdataset、ライセンス、入力解像度、Params、FLOPs、Macro@Best F1が明記されている。
2. `model.onnx`が実在し、表示容量を確認できる。外部データ形式なら`model.onnx_data`などの全ファイル名も確認する。
3. DBV4では`selected_tags.csv`、`preprocess.json`、`categories.json`、`thresholds.csv`の有無と取得元を確認する。WD14では公式推論コードの前処理を確認する。
4. ONNX出力末尾の要素数とCSVの行数が一致し、ratingの4ラベルとcategory IDを特定できる。
5. 入力layout（NCHW/NHWC）、dynamic batch、RGB/BGR、0～1/0～255、Normalize、padding方法を確認する。
6. 出力がlogitsかsigmoid済み確率かを公式サンプルで確認する。
7. 既存モデルに対し「同等以上の精度で軽い」または「明確に高精度」という価値がある。ParamsだけでなくFLOPs、解像度、ONNX容量も比較する。
8. gated modelなら自動承認か管理者承認かを確認し、プリセットに`requires_manual_approval`と案内を設定する。
9. 採用後は小さな検証画像群でCPU、対象GPU provider、batch-size 1と既定値、rating、タグ品質、VRAM、ms/imgを実測する。

公式の比較起点は[AnimeTimm model zoo](https://huggingface.co/animetimm)と[DBV4 ranklist](https://huggingface.co/spaces/animetimm/dbv4-full-ranklist)、WD14互換前処理は[SmilingWolf公式WD Tagger](https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py)とする。

## DBV4出力

DBV4ではモデル出力を固定の先頭4要素として扱わず、selected_tags.csv とmetadataを基準に解釈する。

~~~text
raw output
  ↓
label metadata
  ↓
General / Character / Rating
  ↓
tag-specific best_threshold
  ↓
最終タグ
~~~

Ratingもmetadataから general / sensitive / questionable / explicit の4ラベルを取得する。DBV4のscore分布をWD14 V3のscore分布と同一視しないため、R-15 / R-17用のseverity計算は別レイヤーとして扱うぞ。

### Tag threshold

通常時は各タグの best_threshold を使う。

~~~text
prediction[tag] >= tag.best_threshold
~~~

--thresh を明示した場合のみ、全タグへその値をoverrideとして適用するのじゃ。

## R-00 / R-15 / R-17 / R-18

R-15 / R-17はDBV4公式ratingではなく、このプロジェクト独自のseverity尺度じゃ。

~~~text
R-00
R-15_0
R-15_1
R-15_2
R-15_3
R-15_4
R-17_0
R-17_1
R-17_2
R-17_3
R-17_4
R-18
~~~

Sensitive帯とQuestionable帯は同じ連続severity軸上に置き、R-15_4 → R-17_0 と連続する。

R-00はGeneral thresholdの安全弁、R-18はExplicit側の基本判定として維持し、Sensitive / Questionable の内部を5段階へ分割する。

--sensitive-split-mode は旧CLI互換のため残しておるが、DBV4では5段階固定のため非推奨じゃ。

## XMP

既存の XMP:Subject 運用とExifToolを維持する。

DBV4推論時には、4 rating scoreに加えて以下のmarkerを保存する。

~~~text
dbv4_model:<repository-id>
general_score:0.XXXX
sensitive_score:0.XXXX
questionable_score:0.XXXX
explicit_score:0.XXXX
~~~

この4つのscoreとDBV4 model markerが揃っている場合、--organize では再推論せずratingを再計算できる。旧WD14 scoreだけが残っている場合はDBV4推論を実行するぞ。


## Bash / Zsh のタブ補完

`run_tagger.sh` のオプション、モデルプロファイル、ファイル/ディレクトリ引数をタブ補完できます。

### Bash

一時的に有効化する場合:

```bash
source completions/run_tagger.bash
```

常時有効化する場合は、補完ファイルをユーザー側の Bash completion ディレクトリへ配置します。

```bash
mkdir -p ~/.local/share/bash-completion/completions
cp completions/run_tagger.bash ~/.local/share/bash-completion/completions/run_tagger.sh
```

### Zsh

リポジトリ内の補完を現在のシェルで有効化する場合:

```zsh
fpath=("$PWD/completions" $fpath)
autoload -Uz compinit
compinit
```

常時有効化する場合は `completions/_run_tagger.sh` を `fpath` に含まれるディレクトリへ配置してください。

補完では短縮形・長形式の両方を候補に表示し、`--path` / `--model-file` / `--tags-file` ではファイルパス、`--model-profile` では利用可能なプロファイル、`--sensitive-split-mode` では `2 / 4 / 6` を候補として表示します。

## CLI短縮オプション

主要な実行引数には短縮形を用意している。既存の長い形式はそのまま利用できる。

| 用途 | Linux / Bash | Windows / PowerShell |
| --- | --- | --- |
| 対象パス | `-p` / `--path` | `-p` / `-Path` |
| GPU | `-g` / `--gpu` | `-g` / `-Gpu` |
| 整理 | `-o` / `--organize` | `-o` / `-Organize` |
| タグ付け | `-t` / `--tag` | `-t` / `-Tag` |
| Pixiv | `-x` / `--pixiv` | `-x` / `-Pixiv` |
| 強制再推論 | `-f` / `--force` | `-f` / `-Force` |
| 再帰検索 | `-r` / `--recursive` | `-r` / `-Recursive` |
| 再帰検索OFF | `-n` / `--no-recursive` | `-n` / `-NoRecursive` |
| バッチサイズ | `-b` / `--batch-size` | `-b` / `-BatchSize` |
| モデルプロファイル | `-m` / `--model-profile` | `-m` / `-ModelProfile` |
| Server | `-S` / `--server` | `-s` / `-Server` |
| Client | `-K` / `--client` | `-c` / `-Client` |
| Help | `-h` / `--help` | `-h` / `-Help` |

全オプションと短縮形は `./run_tagger.sh -h` または `.\\run_tagger.ps1 -h` で確認できる。

## Windows / PowerShell

初回セットアップ：

~~~powershell
.\run_tagger.ps1
~~~

通常実行：

~~~powershell
.\run_tagger.ps1 -Path "C:\Images" -Gpu
~~~

モデルプロファイル指定：

~~~powershell
.\run_tagger.ps1 -Path "C:\Images" -Gpu -ModelProfile high
~~~

フォルダ整理のみ：

~~~powershell
.\run_tagger.ps1 -Path "C:\Images" -Organize
~~~

タグ付け＋整理：

~~~powershell
.\run_tagger.ps1 -Path "C:\Images" -Tag -Organize
~~~

推論サーバー：

~~~powershell
.\run_tagger.ps1 -Server -Gpu -ModelProfile balanced
~~~

クライアント：

~~~powershell
.\run_tagger.ps1 -Client -HostIP "192.168.1.10" -Path "C:\Images" -Organize
~~~

## Linux / Bash

`run_tagger.sh`の一時ファイルは`/tmp`を使う。`~/.cache`が容量の小さいtmpfsの場合、Hugging Faceのモデルキャッシュは自動的に`~/.local/share/huggingface`へ保存する。既存の`HF_HUB_CACHE`・`HF_XET_CACHE`指定は優先される。モデルキャッシュは再利用するデータであり、`/tmp`には置かない。

初回セットアップ：

~~~bash
./run_tagger.sh
~~~

Hugging Faceのgated modelを利用する前のログイン：

~~~bash
./run_tagger.sh --login
~~~

`ultra`はONNX本体の公開リポジトリとは別に、タグと前処理データを`animetimm/convnextv2_huge.dbv4-full`から取得する。利用前に[モデルページ](https://huggingface.co/animetimm/convnextv2_huge.dbv4-full)で利用条件に同意し、同意したアカウントで`--login`を実行する。401が出る場合は、そのアカウントにアクセス権があるか確認する。
`ultra`などのアクセス確認で401が返った場合は、モデルページを開いて再ログインを促し、同じ実行内でアクセスを再確認する。利用条件への同意や管理者承認がまだ完了していない場合は、案内を表示して停止する。

既存のGPU／CPU仮想環境から`hf`を利用できる環境を再利用し、環境が一つもない場合だけCPU環境を作成する。ブラウザで作成したread権限のtokenを入力すると、認証情報はHugging Face標準の保存先へ保存され、ログイン後は推論を実行せず終了する。

通常実行：

~~~bash
./run_tagger.sh -g -p /path/to/images
~~~

モデルプロファイル指定：

~~~bash
./run_tagger.sh -g --model-profile high -p /path/to/images
~~~

フォルダ整理：

~~~bash
./run_tagger.sh --organize -p /path/to/images
~~~

推論サーバー：

~~~bash
./run_tagger.sh --server -g --model-profile balanced
~~~

クライアント：

~~~bash
./run_tagger.sh --client --host 192.168.1.10 -p /path/to/images --organize
~~~

Intel OpenVINOを使用する場合は既存の --gpu 経路を維持し、openvino_gpu_device に使用デバイスを指定できるぞ。

### Linux実機検証

Ubuntu 26.04.1 LTS、kernel 7.0.0-31-generic、Intel Core i7-8750H、GeForce RTX 2070 Mobile 8GB、NVIDIA driver 610.57.04で検証した。システムのPython 3.14.4は使用せず、wrapperがPython 3.13を検出して`venv_std`と`venv_gpu`を作成した。

| Profile | Provider | batch-size | 結果 | 合成画像1枚の実測 |
| --- | --- | ---: | --- | ---: |
| lightweight | CPUExecutionProvider | 4 | 推論・XMP書き込み成功 | 89.8 ms/img |
| lightweight | CUDAExecutionProvider | 4 | 推論・XMP書き込み成功 | 499.2 ms/img |
| balanced | CPUExecutionProvider | 4 | 推論・XMP書き込み成功 | 719.9 ms/img |
| balanced | CUDAExecutionProvider | 4 | 推論・XMP書き込み成功 | 303.9 ms/img |
| wd14_v3 | CPUExecutionProvider | 4 | 推論・XMP書き込み成功 | 919.8 ms/img |
| wd14_v3 | CUDAExecutionProvider | 4 | 推論・XMP書き込み成功 | 376.3 ms/img |

保存済みrating scoreを使った2回目の整理は、推論0枚・skip 1枚で完了した。lightweightのlocalhost Server/Client推論と、lightweight serverへ接続したbalanced clientのmodel ID不一致による全体停止も確認した。Clientは既存仮想環境を再利用し、GPUランタイムを再導入しない。上記は起動経路確認用の4 × 4合成画像による単発値であり、モデル間性能比較やタグ品質評価には使用しない。TensorRTは検証環境の`libnvinfer`が不完全だったため候補から除外され、実測providerはCUDAまたはCPUである。

## Google Colab

run_colab_server.ipynb はDBV4サーバーを起動する構成へ更新されておる。

<a href="https://colab.research.google.com/github/SyameimaruKoa/dbv4-wd14-tagger/blob/main/run_colab_server.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

初期状態では balanced プロファイルを使用する。ローカル側は通常のClientモードで接続できるぞ。

## 設定ファイル

初回実行時に config.json が自動生成される。既存の設定へ不足項目を追記する方式なので、旧設定ファイルが残っていても破壊しない。

主要項目：

| Key | 初期値 | 説明 |
| --- | --- | --- |
| model_profile | "balanced" | 使用するDBV4プロファイル |
| server_hosts | ["localhost", "google-colab", "100.xxx.xxx.xxx"] | Client接続先 |
| server_port | 5000 | Server/Clientポート |
| server_workers | 2 | Server同時推論数（GPUメモリに応じて調整） |
| client_timeout | 15 | Client timeout秒 |
| openvino_gpu_device | "GPU.0" | Intel OpenVINOデバイス |
| general_threshold | 0.40 | R-00安全弁 |
| rating_sublevel_thresholds_5way | [0.20,0.40,0.60,0.80] | R-15/R-17の5段階境界 |
| record_rating_percentages | true | rating割合をXMPへ記録 |
| record_raw_score | true | raw rating scoreをXMPへ記録 |
| folder_names | 下記 | 整理先フォルダ名 |

プロファイル自体も model_profiles へ追加・上書きできるぞ。

## Server / Clientプロトコル

Serverは次の情報をJSONで返す。

~~~json
{
    "protocol": 1,
    "model_id": "animetimm/caformer_b36.dbv4-full",
    "profile": "balanced",
    "metadata_version": "xxxxxxxxxxxxxxxx",
    "output_size": 12476,
    "rating_labels": [
        "general",
        "sensitive",
        "questionable",
        "explicit"
    ],
    "probabilities": []
}
~~~

Clientは処理開始時にServerの`/metadata`からmodel ID、profile、metadata version、output sizeを取得する。`--model-profile`を省略した場合はmodel IDに一致する既知profileを自動選択し、ONNX本体をダウンロードせずmetadataだけを読み込む。明示指定したprofileは自動変更しない。どちらの場合もmetadata versionとoutput sizeが一致しなければ、画像処理前に停止する。

Serverは画像ごとに受信時刻、Client IP、ファイル名、転送サイズ、処理開始、処理時間、完了状態を表示する。推論中にClientが切断した場合もServerは停止せず、長い`BrokenPipeError` tracebackの代わりに対象リクエストの警告だけを表示して次の接続を待機する。

Serverは複数Clientの要求を既定で2件まで並列処理する。`config.json`の`server_workers`で同時数を変更できる。ultraなど大きなモデルでGPUメモリ不足になる場合は`1`へ下げる。

## テスト

### Linux Intel GPU実機確認（2026-09-21）

Ubuntu 26.04.1、kernel 7.0.0-31-generic、Core i7-1355U内蔵 Iris Xe（8086:a7a1）、Python 3.13.15、onnxruntime-openvino 1.24.1で、`/dev/dri/renderD128`とIntel OpenCL platformを確認した。`openvino_gpu_device`は`GPU.0`。`--force-intel --model-profile wd14_v3`で合成PNGを処理し、active providerは`OpenVINOExecutionProvider`と`CPUExecutionProvider`、OpenVINO debugログはモデル全体の対応と推論成功を報告した。入力はNHWC `[batch_size, 448, 448, 3]`、ラベルは10,861件。ExifToolでXMP Subject書き込みを確認した。通常ログではOpenVINO内部診断は表示されなかった。

batch-size=4、初回warmup 2.89秒、1枚の通常推論176.0 ms/img（再試行178.1 ms/img）。合成画像1枚の測定値であり、タグ品質や安定した性能の評価ではない。GPU.0指定とOpenVINO EPの実行は確認したが、カーネル単位のGPU使用率は測定していないためCPU fallbackの完全な排除は未確認。`lightweight`はHugging Faceのモデル取得が401（gated repository、未認証）で推論前に停止した。`balanced`、保存scoreでの整理、実画像、Server/Clientの実機疎通は未検証。

DBV4 metadata / preprocessing / input layout / output probability変換の単体テストを実行できる。

~~~bash
python -m unittest discover -s tests -p "test_*.py"
~~~

実モデルのCPU/GPU速度比較やR-15/R-17キャリブレーションは、実行環境ごとのベンチマーク工程として別途評価する。

## ライセンス

DBV4モデル自体のライセンスはモデルごとに異なるため、使用するprofileのモデルカードを確認すること。

モデルファイルはリポジトリへ同梱せず、Hugging Faceから実行時に取得する方式を基本とする。

## 対応範囲

- Windows / Linux
- CPU
- NVIDIA CUDA / TensorRT
- Intel OpenVINO
- AMD ROCm / MIGraphX
- Server / Client
- Batch inference
- Nintendo Switch / ARM64向け既存クライアント経路
- XMP / ExifTool
- HTML report
