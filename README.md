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

### ベンチマーク結果

GPU・OS・実行プロバイダー別の速度、CPU比、バッチサイズの効果、メモリ使用量は [ベンチマーク結果と分析](BENCHMARKS.md) を参照。既存のDirectML VRAM実測とLinux実機確認も移動した。 保存済み178件を集約した[全測定一覧](benchmarks/all_measurements.md)では、速度・RAM・VRAM・GTT・取得できた使用率と、同条件でのモデル間の比を確認できる。未測定は未測定として記載し、GPUメモリのピークと時点値を区別している。

Windowsで`-Gpu`だけを指定した場合は、NVIDIAはTensorRT→CUDA→ncnn Vulkan、IntelはOpenVINO→ncnn Vulkan、AMDはDirectML→ncnn Vulkanを先に確認する。LinuxではNVIDIAはTensorRT→CUDA→ncnn Vulkan、IntelはOpenVINO→ncnn Vulkan、ROCm利用可能なAMDはMIGraphX→ROCm→ncnn Vulkan、旧AMD・PS4 Linux・Switch Linuxはncnn Vulkanから確認する。両OSともncnnの次はWebGPU、CPUの順に起動検証する。`--provider` / `-Provider`を指定すると選択を固定し、初期化に失敗した場合は理由を表示して停止する。TensorRTだけが利用できずCUDAが使える場合は、同じ`venv_tensorrt`をCUDA用に再利用する。各バックエンドは独立した`venv_*`を使用する。

### ncnn Vulkan

```bash
./run_tagger.sh --provider ncnn --gpu -m ultra -p /path/to/images
./run_tagger.sh --provider ncnn --gpu -m ultra --ncnn-precision fp16-storage -p /path/to/images
```

```powershell
.\run_tagger.ps1 -Provider ncnn -Gpu -ModelProfile ultra -Path "C:\Images"
.\run_tagger.ps1 -Provider ncnn -Gpu -ModelProfile ultra -NcnnPrecision fp16-storage -Path "C:\Images"
```

初回は `animetimm/convnextv2_huge.dbv4-full` の同一PyTorch重みを `timm` と `pnnx` で変換し、`.dbv4/models/ultra/model.ncnn.param` と `.bin` に保存する。2回目以降はこのキャッシュを再利用し、ONNX用ファイルは保持する。変換用 `torch`・`timm`・`pnnx` は必要時に導入する。変換にはモデルのHugging Faceアクセス権と十分なメインメモリ・空き容量が必要。小メモリ環境では別のPCで `python convert_ncnn_model.py animetimm/convnextv2_huge.dbv4-full .dbv4/models/ultra --input-size 512 --labels 12476` を実行し、生成された `.param`、`.bin`、`.json` の3ファイルを同じパスにコピーできる。2ファイルだけを配置する場合は `--model-file` / `-ModelFile` で `model.ncnn.param` のパスを明示する。

精度は `fp32`（既定）、`fp16-storage`、`fp16-packed`、`fp16-arithmetic`。モデル変換は共通で、実行時のncnn設定だけを切り替える。ncnn形式はbatch 1のグラフで、複数枚は順に推論する。VulkanドライバーとVulkan対応のncnn Python bindingが必要。Python wheelにVulkanが含まれない環境ではVulkan有効でncnnをビルドするか、WebGPUを使用する。既存のタグ、rating、XMP、Server/Client処理は同じ確率配列を利用する。実機比較の状態は[BENCHMARKS.md](BENCHMARKS.md)を参照。

Switch Linuxの4GB共有メモリでは、同じultraのFP32重みを `--ncnn-part-size-mib 128` で区間ごとに読み込み、実GPU推論とCPU参照の一致を確認した。例えば `./run_tagger.sh --provider ncnn --gpu -m ultra --ncnn-part-size-mib 128 -p /path/to/images`。元のモデル・重みは変更せず、分岐をまたがない位置で切り、1区間ずつ読み込み・解放する。128MiBは重み量の目安で、今回の最大区間は約243MiB。初回は同じ容量の分割キャッシュを追加するため約2.58GiBの空き容量が必要。毎画像で重みを再読み込みするので、メモリに余裕のあるPCでは既定の分割なし（0）を使う。Switchの初回測定は約75秒/枚だったため、Server/Clientでは `client_timeout` と `client_batch_timeout` を600秒などに設定する。旧NVIDIA ICDを選ぶ必要がある環境では、実行前に `VK_DRIVER_FILES` と `VK_ICD_FILENAMES` を `/etc/vulkan/icd.d/nvidia_icd.json` へ設定する。実測条件は[BENCHMARKS.md](BENCHMARKS.md)を参照。

分割推論をServerで使う場合は、複数Clientの推論を順番に実行する。区間の読み込み・解放が他の接続と干渉しないようにし、同時実行によるメモリ増加も抑える。待ち時間には先行するClientの処理時間も含まれるため、接続数に合わせてタイムアウトを調整する。

PS4 Linux（AMD Liverpool、VRAM 2GiB）でも、ultra FP32を同じ128MiB指定・14区間で検証した。約80〜88秒/枚で、CPU参照の確率・タグとServer/ClientのXMPが一致した。通信の `client_timeout` と `client_batch_timeout` は600秒を設定する。検証環境のExifToolはローカル配置のため、Bashでは以下のようにPATHへ追加して実行する。初回はキャッシュ生成と起動検証にも数分かかる。詳細は[BENCHMARKS.md](BENCHMARKS.md)を参照。

```bash
export PATH="$PWD/.dbv4/runtime/exiftool:$PATH"
bash run_tagger.sh --provider ncnn --gpu -m ultra --ncnn-part-size-mib 128 --gpu-index 0 -p /path/to/images
```


LinuxでAMD GPUを検出した場合、ncnnのVulkan初期化前に `RADV_DEBUG` へ `syncshaders` を追加する。AMD RADV実機で確認した連続演算時の確率差を抑える同期設定で、既存の環境変数のフラグは保持する。起動ログに適用を表示し、ベンチマークにも環境変数を記録する。

AMD実機のultraはFP32で確率・rating・タグとXMPの一致を確認した。FP16の3設定は非有限出力で停止したため、この組み合わせではFP32を使用する。

同じAMD PCのBazzite 44／Mesa 26.2.2でもFP32のbatch 1／4とXMP・Server/Clientを確認した。OS標準Python 3.14を変更せず、リポジトリ内のPython 3.13と`venv_ncnn`を使った。ExifToolを`.dbv4/runtime/exiftool`へ配置した場合は、実行前に `export PATH="$PWD/.dbv4/runtime/exiftool:$PATH"` を設定する。`--login`／`-Login`は既存の`venv_ncnn`も探索し、ログイン用に再利用する。

Windows 11のCore i7-1355U／Intel Iris Xeでは、OpenVINOの既定FP16実行でultraの全出力がNaNになるため、Intel providerはFP32を明示する。修正後は約1.6秒/枚で、全12,476確率・rating・タグとXMP・両Client転送の一致を確認した。`-Provider intel -Gpu -ModelProfile ultra -OpenVinoDevice GPU` で利用できる。同じGPUのncnn FP32も約33秒/枚で確認済み。省メモリの分割設定はPowerShellでは `-NcnnPartSizeMiB`（既定0）で指定する。[測定・失敗記録](BENCHMARKS.md)を参照。

初回変換は推論より多くのRAMを必要とする。保存するパラメーターの勾配を無効化し、pnnxが形状確認時に不要な勾配履歴を保持することを防ぐ。TorchScriptを作成するプロセスを終了してからpnnxを起動し、変換前半のメモリも解放する。pnnx自体がメモリ不足で終了する場合は、余裕のあるPCで変換してキャッシュ3ファイルを配置する。[CPUでの変換照合](validate_ncnn_conversion.py)、[ultra測定ランナー](benchmark_ncnn_matrix.sh)、[実GPUのXMP・Server/Client検証](tests/test_ncnn_integration.py)も用意している。

### 将来向け1B級

`future_1b`には`animetimm/vit_giantopt_patch16_siglip_384.dbv4-full`を予約した。公式値は1.2B parameters、2.3 TFLOPs、512角、Macro@Best F1 0.607で、ultraの0.611に近い。ただし2026-09-21時点の公式repoには4,734,142,376 bytesの`model.safetensors`とPyTorch重みしかなく、`model.onnx`は存在しない。このプロジェクトはONNX Runtimeを使うため、現在は理由を表示して停止する。公式ONNXまたは検証済み変換版が公開された時点で取得元と外部データ構成を確定し、16GB以上のGPUで実測して有効化する。

### highプロファイルの位置付け

`high`のEVA02は公式Macro@Best F1が0.599で、balancedの0.581より高い一方、[過去のDirectML実測](benchmarks/legacy_results.md)では処理時間がultraに近い。そのため新規利用には非推奨とし、既存configと比較試験の互換性のため定義だけを保持する。

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

### ポータブルな保存先

ランチャーが作成する認証情報、モデル、キャッシュ、TensorRT engine、pipキャッシュ、仮想環境はすべてこのリポジトリフォルダ内へ保存する。Hugging Faceのtokenは `.dbv4/huggingface/token` に一度保存され、CPU・CUDA・TensorRTなど別の仮想環境から共通利用される。仮想環境を作り直しても再ログインは不要であり、リポジトリフォルダを削除すれば関連データも削除される。

`.dbv4/`、`.dbv4_models/`、`venv_*` はGit管理外である。tokenを含む `.dbv4/` を共有・コミットしないこと。

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

### TensorRT

WindowsとLinuxのTensorRTプロバイダーは、初回実行時に公式PyPIのCUDA 12用 `tensorrt-cu12` を `venv_tensorrt` へ自動導入する。NVIDIAドライバーはOS側に必要だが、Python依存関係、TensorRTランタイム、engine cacheはリポジトリ内に保存される。

~~~powershell
.\run_tagger.ps1 -Provider tensorrt -GpuIndex 0 -Path "C:\Images"
~~~

PyPI版を使用できない場合は、[NVIDIA TensorRT 10ダウンロード](https://developer.nvidia.com/tensorrt/download/10x)からWindows CUDA 12版ZIPを取得して利用条件へ同意し、展開後の `TensorRT-10.x.x.x` フォルダを `.dbv4/runtime/` に配置する。ランチャーは `.dbv4/runtime/TensorRT-*/bin/nvinfer_10.dll` を自動検出する。任意の場所へ置く場合だけ `-TensorRtLibDir` を指定する。詳細は[NVIDIA公式Windows ZIP導入手順](https://docs.nvidia.com/deeplearning/tensorrt/10.16.0/installing-tensorrt/install-zip.html)を参照。

## Linux / Bash

`run_tagger.sh`の一時ファイルは`/tmp`を使う。認証情報、モデル、pipキャッシュ、TensorRT engine cacheはWindowsと同様にリポジトリ内の`.dbv4/`へ保存する。

初回セットアップ：

~~~bash
./run_tagger.sh
~~~

Hugging Faceのgated modelを利用する前のログイン：

~~~bash
./run_tagger.sh --login
~~~

Windows PowerShellでは`.\run_tagger.ps1 -Login`を実行する。

`ultra`はONNX本体の公開リポジトリとは別に、タグと前処理データを`animetimm/convnextv2_huge.dbv4-full`から取得する。利用前に[モデルページ](https://huggingface.co/animetimm/convnextv2_huge.dbv4-full)で利用条件に同意し、同意したアカウントで`--login`を実行する。401が出る場合は、そのアカウントにアクセス権があるか確認する。
`ultra`などのアクセス確認で401が返った場合は、モデルページを開いて再ログインを促し、同じ実行内でアクセスを再確認する。利用条件への同意や管理者承認がまだ完了していない場合は、案内を表示して停止する。

リポジトリ内の`venv_*`と`.venv`を探索し、PythonとHugging Face CLIが実際に起動する環境を再利用する。壊れた環境はスキップし、`hf`起動ファイルがなくてもPythonからCLIを呼び出す。CLIが不足する場合は既存の起動可能な環境へ`huggingface_hub`だけを追加・更新し、環境がなければ軽量な`venv_login`を作成する。ログインにGPU・推論ライブラリの導入は不要。ブラウザで作成したread権限のtokenを入力すると、認証情報は`.dbv4/huggingface/token`へ保存され、ログイン後は推論を実行せず終了する。

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

Clientも`--batch-size`（既定4）で複数画像を1回の通信と推論にまとめられる。ServerとClientの両方をこの機能に対応する版へ更新し、Serverを再起動すること。古いServerに接続した場合は自動的に1枚ずつ処理する。モデルの入力が固定バッチ1枚なら、Clientも1枚ずつ処理する。
Clientは最大2バッチの通信を並行させ、次のバッチのアップロードと前のバッチの推論・タグ書き込みを重ねる。バッチ推論に非対応の場合、この先読みは行わない。
既定の`client_upload_mode: "preprocessed"`では、ClientがServerと同じ前処理を実行し、モデル入力のfloat32テンソルを可逆圧縮して送る。前処理の照合結果が一致した形式は元画像送信と同じテンソルが推論に渡る。Client側のCPU負荷を抑えたい場合は`"original"`を選ぶ。既存の`config.json`に`"original"`が保存されている場合、その設定は自動変更されない。Bashでは`-U p`、PowerShellでは`-um p`で今回だけ前処理済み転送を選べる。`o`で元画像送信を選べる。JPEG等のコーデック版差がある画像は結果を保つため元画像で送る。前処理済みモードには対応するServerが必要。前処理自体が一致しない場合はClient仮想環境内のPillow・NumPy・AVIFプラグインをServerの版へ自動更新してClientを再起動する。準備・再照合・転送のいずれかで前処理済みモードを使用できなければ理由を警告し、元画像送信で処理を続ける。
新しいClientとServer間のバッチ応答は、サイズが大きい場合にgzipで圧縮する。古いServerからの非圧縮応答も引き続き利用できる。圧縮を有効にするにはServerの更新と再起動が必要。
バッチ応答の待ち時間は`client_batch_timeout`（既定120秒）と`client_timeout × バッチ枚数`の大きい方を使う。初回のGPU推論が期限切れになった場合は、その実行中の残りを1枚ずつ再試行する。
固定batch 1の単画像通信は`client_timeout`（既定15秒）を使う。WindowsのIntel UHD 630でultraをncnn実行すると1枚約140秒かかったため、クライアントの`config.json`で`client_timeout`と`client_batch_timeout`を例えば300秒に設定する。300秒設定で原画像・前処理済み転送のXMP一致を確認した。

Intel OpenVINOを使用する場合は既存の --gpu 経路を維持し、openvino_gpu_device に使用デバイスを指定できるぞ。
Intel GPUの計算精度はFP32を指定する。モデルのFP32出力形式だけでは内部FP16演算を防げないため、OpenVINOの `INFERENCE_PRECISION_HINT=f32` を設定する。
追加のGPU指定にも短縮形を使える。実行プロバイダは`-ep`、WebGPUは`-wg`、GPU番号は`-gi`、DirectML番号は`-di`、WebGPU番号は`-wi`、対象ベンダーは`-tv`、OpenVINOデバイスは`-od`、TensorRTライブラリ場所は`-td`。Bash・PowerShell・Python CLIで同じ短縮形を使える。

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

初期状態では balanced プロファイルを使用する。設定セルのプルダウンから、実行可能なDBV4プロファイルと旧WD14 V3互換プロファイルを選択できる。承認制モデルで未ログインの場合は、サーバー起動前にノートブック上でHugging Faceログインを要求する。モデルページでの利用条件への同意も必要。Colab Secretsの`HF_TOKEN`を設定済みなら、そのトークンを使用する。

Tailscaleホスト名はColab Secretsの`TAILSCALE_HOSTNAME`を優先して使用し、未登録の場合は設定セルの`HOSTNAME`へフォールバックする。Tailscale認証キーを使う場合は従来どおり`TAILSCALE_AUTHKEY`へ登録する。ローカル側は通常のClientモードで接続できるぞ。

## 設定ファイル

初回実行時に config.json が自動生成される。既存の設定へ不足項目を追記する方式なので、旧設定ファイルが残っていても破壊しない。

主要項目：

| Key | 初期値 | 説明 |
| --- | --- | --- |
| model_profile | "balanced" | 使用するDBV4プロファイル |
| server_hosts | ["localhost", "google-colab", "100.xxx.xxx.xxx"] | Client接続先 |
| server_port | 5000 | Server/Clientポート |
| server_workers | 2 | Server同時推論数（GPUメモリに応じて調整） |
| server_max_request_mib | 128 | HTTP要求本文の上限（MiB） |
| server_max_batch_images | 8 | 1バッチの画像枚数上限 |
| server_max_image_pixels | 80000000 | 1画像の画素数上限（旧既定値20000000は自動更新） |
| client_max_request_mib | 128 | Clientが組み立てるHTTP要求本文の上限（MiB） |
| client_upload_mode | "preprocessed" | Clientがモデル用前処理を実行。`"original"`でClient負荷を低減 |
| client_timeout | 15 | Client timeout秒 |
| client_startup_wait_seconds | 1800 | Serverのモデル読み込みを待つ上限（秒） |
| client_batch_timeout | 120 | バッチ推論の応答待ち時間の下限（秒） |
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
受信バッファの上限は`server_max_request_mib`（既定128 MiB）、バッチ枚数の上限は`server_max_batch_images`（既定8枚）、画像の画素数上限は`server_max_image_pixels`（既定8000万画素）で指定する。`server_workers`と合わせてServerが同時に保持する要求の規模を制限する。ClientとServerで画像コーデックの版が異なる形式は元画像を送信し、版が一致する形式は前処理済みテンソルを送信する。混在バッチは形式ごとに分けて送信する。
Serverはモデル読み込み前にポートを開き、`/metadata`へ準備状態をHTTP 503で返す。Clientは準備完了まで待機し、エラーまたは待機上限に達した場合は理由を表示して停止する。Serverは各画像・バッチ要求のHTTP本文について、受信サイズ、受信時間、受信速度（MiB/s）をログに記録する。この値はServerが本文を読み取った速度であり、Tailscaleなどの中継が本文をバッファした場合はClientからの実効回線速度とは異なる。

## テスト

実機測定・性能比較・失敗条件は [ベンチマーク結果](BENCHMARKS.md) に集約した。過去のIntel／AMD動作確認は [旧測定記録](benchmarks/legacy_results.md) を参照。

DBV4 metadata / preprocessing / input layout / output probability変換の単体テストを実行できる。

~~~bash
python -m unittest discover -s tests -p "test_*.py"
~~~

実モデルのCPU/GPU速度比較は [ベンチマーク結果](BENCHMARKS.md) を参照。R-15/R-17キャリブレーションは別途評価する。

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
- XMP / ExifTool
- HTML report

Nintendo Switchの実機調査: [Issue #18](https://github.com/SyameimaruKoa/dbv4-wd14-tagger/issues/18)
