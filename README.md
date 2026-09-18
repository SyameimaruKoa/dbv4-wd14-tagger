# WD14 Tagger Universal

<a href="https://colab.research.google.com/github/SyameimaruKoa/wd14-tagger-xmp/blob/main/run_colab_server.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

このツールは、AI (WD14 Tagger) を使用して画像認識を行い、タグ付け・フォルダ整理・レポート作成を行うツールじゃ。

Windows (PowerShell) と Linux (Bash) の両方に対応しておる。

## 特徴

- **3つの実行モード** : 通常実行、推論サーバー、クライアント送信モード。
- **柔軟なアクション** : 「タグ付けだけ」「整理だけ」「レポート作成だけ」など自由に組み合わせ可能。
- **事故防止** : 引数なしで実行しても、環境構築のみを行い、勝手にファイルを書き換えることはない。
- **完全日本語** : ヘルプもログも日本語じゃ。

## 推奨モデル

以下が現在の推奨モデルじゃ。`config.json` やコマンド引数で自由に変更できるぞ。

- **デフォルト (バランス・バッチ推論対応):** `SmilingWolf/wd-swinv2-tagger-v3`
- **最高精度 (高スペックPC向け):** `SmilingWolf/wd-eva02-large-tagger-v3` または `SmilingWolf/wd-vit-large-tagger-v3`
- **軽量・高速:** `SmilingWolf/wd-vit-tagger-v3` または `SmilingWolf/wd-convnext-tagger-v3`

### 利用可能なモデル一覧 (SmilingWolf氏作)

現在利用可能な主なモデル一覧じゃ。新しいモデルほどタグの網羅性や精度が高い傾向にあるぞ。

#### V3 (最新世代・標準サイズ)

- **`SmilingWolf/wd-swinv2-tagger-v3`** : 当ツールのデフォルト。精度と速度のバランスが良く、バッチ推論にも強いため大量の画像処理に向いておる。
- **`SmilingWolf/wd-vit-tagger-v3`** : 非常に人気のある軽量・高速モデルじゃ。
- **`SmilingWolf/wd-convnext-tagger-v3`** : V2時代からあるCNNベースの系譜。標準的な性能じゃな。

#### Large V3 (最新世代・大規模サイズ)

- **`SmilingWolf/wd-eva02-large-tagger-v3`** : 最高クラスの精度を誇る大規模モデルじゃ。細かな装飾まで拾いやすいが、ファイルサイズが大きく推論に少し時間がかかるぞ。
- **`SmilingWolf/wd-vit-large-tagger-v3`** : 同じく大サイズの高精度モデルじゃ。VRAM/RAMに余裕があるなら試してみるがよい。

#### V2 (旧世代)

- **`SmilingWolf/wd-v1-4-swinv2-tagger-v2`** : 以前のデフォルトモデルじゃ。安定しておるが、タグの種類はV3に劣るのう。
- **`SmilingWolf/wd-v1-4-moat-tagger-v2`** : V2世代での高精度モデルじゃった。
- **`SmilingWolf/wd-v1-4-convnext-tagger-v2`** : 旧標準モデルじゃ。
- **`SmilingWolf/wd-v1-4-vit-tagger-v2`** : 旧軽量モデルじゃな。

他のモデルを使う場合は `config.json` の `model_repo` やコマンド引数 `-ModelRepo` などで指定するのじゃ。

（※ `model_file` や `tags_file` の名前が通常と異なる特殊なモデルを使う場合は、そちらも合わせて指定する必要があるぞ）

## 使い方 (Windows / PowerShell)

基本的には `run_tagger.ps1` を使用する。

### 1. 初回セットアップ

引数なしで実行すると、必要な仮想環境を作成して終了する。

このとき、同じフォルダに設定ファイル config.json も生成されるぞ。

```
.\run_tagger.ps1

```

### 2. 基本的な使い方 (タグ付け + レポート)

フォルダまたは画像ファイルを指定して実行する。デフォルトではGPUを使用せずCPUで動く。

```
.\run_tagger.ps1 -Path "C:\Images"

```

GPUを使う場合は `-Gpu` をつける（推奨）。

```
.\run_tagger.ps1 -Path "C:\Images" -Gpu

```

### 3. フォルダ整理のみを行いたい場合

タグ付けはせず、既存のタグ（またはAI判定結果）に基づいてフォルダ振り分けだけを行う。

```
.\run_tagger.ps1 -Path "C:\Images" -Organize

```

※このモードでは、サブフォルダは検索されない（直下のみ）。

### 4. 両方行いたい場合

タグ付けをして、その結果でフォルダ整理もする。

```
.\run_tagger.ps1 -Path "C:\Images" -Tag -Organize

```

### 5. クライアントモード

別のPCでサーバーモード (`-Server`) を起動しておき、そこへ画像を投げる。

```
.\run_tagger.ps1 -Client -HostIP "192.168.1.10" -Path "C:\Images" -Organize

```

### 6. Google Colab サーバーモードでの実行

ローカルのPCスペックが不足している場合や、GPUを使用したい場合は、Google Colaboratory (GPU環境) で推論サーバーを起動し、ローカルから接続してタグ付けを行うことができるぞ。

<a href="https://colab.research.google.com/github/SyameimaruKoa/wd14-tagger-xmp/blob/main/run_colab_server.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

#### 接続手順

1. **Tailscale 認証キーの準備**:
   - Tailscale の管理画面の [Settings > Keys](https://login.tailscale.com/admin/settings/keys) から認証キー（Auth Key）を作成するのじゃ。
   - **【Google ドライブ保存モード（推奨）】**:
     - **Reusable: ON**, **Ephemeral: OFF（チェックを外す）**, **Tags: `tag:Guest`**
     - ★ Google ドライブに認証状態が保存されるため、初回に1度認証すれば、次回以降はシークレット登録もキー入力も不要でワンクリック起動できるようになるぞ！
   - **【一時利用（使い捨て）モード】**:
     - **Reusable: ON**, **Ephemeral: ON**, **Tags: `tag:Guest`**
2. **Colab Secrets (シークレット) の登録 (任意)**:
   - Google Colab の左メニューにある鍵マーク（Secrets）に名前 `TAILSCALE_AUTHKEY`、値にコピーした認証キーを登録しておくと、初回起動時も入力プロンプトなしで自動認証されるぞ。
   - ※未登録の場合でも、初回実行時に入力プロンプト（またはブラウザ認証リンク）が表示されるので安心じゃ。
3. **サーバーの起動**:
   - 上記の「Open In Colab」バッジからノートブックを開き、上部メニューの **「ランタイム」 > 「すべてのセルを実行」** を押すだけで全自動起動するぞ。
   - `USE_GOOGLE_DRIVE = True` の場合、初回認証後に Google ドライブへ Tailscale の認証情報が自動保存されるぞ。次回以降はシークレットやキー入力なしで即座に自動再接続されるのじゃ！
   - 正常に起動すると、Tailscale の IP アドレスと、ローカル側で実行するためのコマンドの例が表示されるぞ。
4. **ローカルからの接続実行**:
   - Tailscale の MagicDNS を利用するため、ホスト名に **`google-colab`** を指定してクライアントモードを実行するのじゃ。これにより、毎回変わる IP アドレスを入力する手間を省けるぞ！
     ```
     .\run_tagger.ps1 -Client -HostIP "google-colab" -Path "C:\Images" -Organize
     ```
   - ※MagicDNS名 `google-colab` は、あらかじめ設定ファイル `config.json` の接続先リスト（`server_hosts`）にも追加されているため、単に `-HostIP` を省略して実行し、接続先選択メニューから選ぶこともできるぞ。
     ```
     .\run_tagger.ps1 -Client -Path "C:\Images" -Organize
     ```
5. **終了手順**:
   - 推論サーバーを停止するには、Google Colab の起動セルの左側にある **停止（セル実行の中断）** ボタンをクリックするのじゃ。
   - Google ドライブ保存モード時は、次回再接続のために認証情報を保持したまま安全に停止されるぞ（完全ログアウトセルも自動スキップされるため安心じゃ）。
   - （完全に Tailscale からデバイスを削除・ログアウトしたい場合のみ、設定セルで `FORCE_LOGOUT_ON_FINISH = True` にするか、手動でログアウトを実行するのじゃ）

## 使い方 (Linux / Bash)

基本的には `run_tagger.sh` を使用する。

### 1. 初回セットアップ

引数なしで実行すると必要な仮想環境が自動構築される。

```bash
./run_tagger.sh
```

> **Python バージョンに関する注意事項:**  
> `onnxruntime` などの各種ライブラリは、現在 Python 3.14 用の公式バイナリ(wheel)がPyPIに存在しない。システム全体の標準 `python3` が 3.14 の場合、`run_tagger.sh` は自動的にシステム内の Python 3.13 以下の互換バージョン（`python3.13` など）を検出して仮想環境を作成するぞ。

### 2. NVIDIA GPU (CUDA / TensorRT) 有効化セットアップ

Linux 環境において NVIDIA GPU (`CUDAExecutionProvider` / `TensorrtExecutionProvider`) を使用して推論を最高速化する場合、`run_tagger.sh` が必要な runtime ライブラリ（`nvidia-cuda-runtime`, `nvidia-cublas`, `nvidia-cudnn`, `tensorrt` 10.x 等）を仮想環境へ自動的に組み込み、ライブラリパスを自動構成するぞ。

`-g`（または `--gpu` / `--force-nvidia`）を付けて実行するのじゃ：

```bash
./run_tagger.sh -g -p /path/to/images
```

- **TensorRT ウォームアップ処理**:  
  TensorRT 等のコンパイルを伴うプロバイダが有効な場合、初回起動時やバッチサイズ変更時にエンジンの自動事前構築（ウォームアップ推論）が行われるぞ。初回のみ準備に時間がかかるが、ウォームアップ完了後は非常に高速に推論が行われるのじゃ。
- **外れ値判定サマリー**:  
  推論処理完了時のサマリーログでは、ウォームアップ所要時間や、万が一発生したコンパイル遅延（外れ値）の自動除外・初回処理時間などの詳細データが分かりやすく報告されるぞ。

### 3. Intel GPU (OpenVINO) 有効化セットアップ

Linux 上で Intel GPU（HD Graphics / Iris Xe / Arc）を使用して推論を高速化する場合、OpenCL ドライバーとデバイスアクセス権限が必要じゃ。初回のみ以下のコマンドを実行しておくのじゃ。

```bash
# 1. OpenCL ドライバーとツールのインストール
sudo apt update && sudo apt install -y intel-opencl-icd clinfo

# 2. ユーザーに GPU アクセス権限を付与
sudo usermod -aG render,video $USER

# 3. 権限の反映（またはターミナル再起動）
newgrp render
```

設定後、`-g`（または `--gpu`）を付けて実行すると Intel GPU (OpenVINO) で高速動作するぞ！

```bash
./run_tagger.sh -g -p /path/to/images
```

### 4. AMD GPU (ROCm) 有効化・動作仕様

Linux 上で AMD GPU を使用する場合、`run_tagger.sh` は `onnxruntime-rocm` パッケージを自動インストールし、`/usr/lib/x86_64-linux-gnu` 等のシステムライブラリ（`librocm_smi64.so` や `libroctracer64.so` 等）に対する動的 SONAME 互換エイリアス（`venv_amd/lib/rocm_compat/`）を自動構築するぞ。

`-g`（または `--force-amd`）を付けて実行できるのじゃ：

```bash
./run_tagger.sh -g -p /path/to/images
```

### 5. Nintendo Switch (Switchroot L4T / Tegra X1) 動作仕様と注意事項

Switchroot Ubuntu 24.04 (Noble) 等の Nintendo Switch 上で実行する場合の動作仕様および制限事項は以下の通りじゃ。

> **Switchroot Ubuntu 24.04 の既知の仕様・制限事項:**
>
> - **No CUDA compiler support (CUDA runtime 10.0 is preinstalled and functions)**:
>   CUDA ランタイム 10.0 (`/usr/lib/aarch64-linux-gnu/tegra/libcuda.so.1`) は組み込まれており機能するが、CUDA コンパイラ (`nvcc`) および cuDNN / TensorRT 開発パッケージは含まれておらぬ。
> - **PyPI の ARM64 パッケージ制限**:
>   PyPI には ARM64 (aarch64) 向けの `onnxruntime-gpu` が存在せず、また ONNX Runtime は 1.8 以降 CUDA 10.x をサポートしておらぬ（Python 3.12 対応版は CUDA 11.8/12 専用）。
> - **スタンドアロン動作**:
>   そのため Switch 単体では、`run_tagger.sh` が自動的に ARM64 (aarch64) を判別し、クラッシュすることなく **CPU (ARM NEON SIMD 最適化マルチスレッド)** で安全・確実に推論を行う設計になっておるぞ。
> - **GPU 推論の推奨構成 (クライアント / サーバー)**:
>   Switch 上の画像を GPU で超高速に処理したい場合は、GPU 搭載 PC でサーバーを起動し、Switch 側から `--client` で接続するのが最もおすすめじゃ！
>
>   ```bash
>   # 【メイン PC (GPU 搭載)】推論サーバー起動
>   python embed_tags_universal.py --mode server --port 5000 --gpu
>
>   # 【Nintendo Switch】PC の GPU を使って Switch 内の画像を処理
>   ./run_tagger.sh --client -H <PCのIPアドレス> -P 5000 -p /path/to/images -o
>   ```

## 処理速度・統計表示について

進捗バー（tqdm）および処理完了時のサマリーログでは、以下の **2通りの速度・処理件数** が個別に分離して表示されるぞ。

1. **推論実行ファイル (AI演算あり)**:  
   実際に AI モデルに画像を入力し、メタデータ解析・タグ推論計算を行ったファイルとその平均処理速度 (`img/s`, `ms/img`)。
2. **演算スキップファイル (既存タグ)**:  
   既に画像メタデータ (XMP) にタグが存在し、AI 推論計算をスキップして既存タグで高速判定・整理を行ったファイルとその処理速度。

## オプション一覧

| **オプション (PS)** | **説明**                                                         |
| ------------------- | ---------------------------------------------------------------- |
| `-Path`             | 処理対象のファイルまたはフォルダ。                               |
| `-Gpu`              | GPUを使用して高速化する。                                        |
| `-Organize`         | フォルダ整理モードを有効にする（デフォルトでタグ付けOFF）。      |
| `-Tag`              | タグ付けを有効にする（`-Organize`と併用時に使用）。              |
| `-NoReport`         | HTMLレポートを作成しない。                                       |
| `-Recursive`        | 強制的にサブフォルダも検索する。                                 |
| `-NoRecursive`      | 強制的にサブフォルダを検索しない。                               |
| `-BatchSize`        | 推論バッチサイズ（ローカル時のみ、デフォルト: 4。非対応時は1）。 |
| `-IoWorkers`        | 画像の読み込み・前処理の並列ワーカー数（デフォルト: 自動）。     |
| `-ModelRepo`        | モデル/タグのHFリポジトリID。                                    |
| `-ModelFile`        | モデルファイル名またはパス。                                     |
| `-TagsFile`         | タグCSVファイル名またはパス。                                    |
| `-Force`            | 既にタグがあっても強制的に再解析・上書きする。                   |
| `-Server`           | サーバーモードで起動する。                                       |
| `-Client`           | クライアントモードで起動する。                                   |

## R-15 / R-17 の細分化について

WD14 v3 の基本 rating は `General → Sensitive → Questionable → Explicit` の4分類として扱う。ここでの R-15 / R-17 は WD14 公式の区分ではなく、このプロジェクト独自の派生尺度である。

基本 rating の判定は従来どおり行い、R-00 の既存General安全弁と R-18 の既存Explicit側判定は変更しない。

Sensitive または Questionable と判定された画像についてのみ、4つの rating score から連続した severity を算出する。4 rating は4択確率として合計せず、Sensitive帯では Gen↔Sen の相対位置と Que の上昇度を、Questionable帯では Sen↔Que の相対位置と Exp の上昇度を使う。上位側のscoreは対数スケールで圧縮するため、Genが極端に低いだけでseverityが最上位へ飛ぶのを防ぎながら、Que / Exp が高くなるほど上位へ連続的に寄せる。

得られた severity は同一の0〜9分割基準で細分化する。

- Sensitive → `R-15_0` 〜 `R-15_9`
- Questionable → `R-17_0` 〜 `R-17_9`

Sensitive 帯と Questionable 帯は同じ連続severity軸上に配置されるため、命名上も `R-15_9 → R-17_0` と連続する。Questionable を単独の `R-17` フォルダにはしない。

## 設定ファイル (config.json)

初回実行時に config.json が生成される。

これを書き換えることで、細かい挙動や振り分け先のフォルダ名を自由に変更できるのじゃ。

※アップデート等で新しい設定項目が増えた場合、自動的にファイルに追記されるぞ（既存の設定は消えないので安心せよ）。

### 設定項目一覧

| **項目名 (key)**                  | **デフォルト値**                    | **説明**                                                                                                                                                                                                      |
| --------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_repo`                      | `"SmilingWolf/wd-swinv2-tagger-v3"` | 使用するモデル/タグのリポジトリID。                                                                                                                                                                           |
| `model_file`                      | `"model.onnx"`                      | 使用するモデルファイル名（またはローカルパス）。                                                                                                                                                              |
| `tags_file`                       | `"selected_tags.csv"`               | 使用するタグCSVファイル名（またはローカルパス）。                                                                                                                                                             |
| `general_threshold`               | `0.40`                              | **General（全年齢）判定の安全弁**。`AIが「Generalである確率」がこの値以上なら、たとえ他のR指定スコアが高くても強制的に「General」として扱う。`誤爆（安全な画像をR指定にしてしまうこと）を防ぐための設定じゃ。 |
| `rating_sublevel_thresholds_5way` | `[0.20, 0.40, 0.60, 0.80]` | Sensitive / Questionable の各帯を共通の0〜4 suffixへ分割する境界。WD14公式基準ではなく、このプロジェクト独自のseverity尺度。 |
| `rating_severity_sensitive_upper_reference` | `25.0` | Sensitive帯で上側severityを測るQue scoreの参照上限（%）。WD14公式基準ではなく、実測出力を基にした初期キャリブレーション値。 |
| `rating_severity_questionable_upper_reference` | `40.0` | Questionable帯で上側severityを測るExp scoreの参照上限（%）。同上。 |
| `record_rating_percentages`       | `true`                              | `general:XX.X%`, `sensitive:XX.X%` 等の割合タグ（全4レーティング）を XMP に記録するか否か。CLI の `--record-ratio` / `--no-record-ratio` で上書き可。                                                         |
| `record_raw_score`                | `true`                              | `general_score:0.XXXX`, `sensitive_score:0.XXXX` 等の RAW スコアタグ（全4レーティング）を XMP に記録するか否か。`--organize` 時の高速再判定・再整理に使用される。                                             |
| `server_host`                     | `"localhost"`                       | サーバーモードやクライアントモードで使うデフォルトのIPアドレス。                                                                                                                                              |
| `server_port`                     | `5000`                              | 通信に使用するポート番号。                                                                                                                                                                                    |
| `folder_names`                    | (下記参照)                          | 整理モード (`--organize`) で振り分けられるフォルダ名の設定。                                                                                                                                                  |

### folder_names (フォルダ名設定)

整理モードで作成されるフォルダの名前を自由に変更できる。

R-15 / R-17 は、それぞれ0〜4の同一suffix規則で管理する。severityの初期キャリブレーション値は config.json で調整できる。

| **キー** | **デフォルトフォルダ名** | **対応するレーティング** |
| --- | --- | --- |
| `general` | `"R-00"` | 全年齢 (Safe) |
| `sensitive_0` ～ `sensitive_9` | `"R-15_0"` ～ `"R-15_9"` | Sensitive帯のseverity 5段階 |
| `questionable_0` ～ `questionable_9` | `"R-17_0"` ～ `"R-17_9"` | Questionable帯のseverity 5段階 |
| `explicit` | `"R-18"` | Explicit |

`sensitive_mild` / `sensitive_high` / `sensitive_lvl1` ～ `sensitive_lvl6` は旧バージョンのタグを再整理時に除去できるよう、内部的には引き続き認識されるが、新規の判定結果としては使用しない。

