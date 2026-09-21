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

balanced は、精度とモデル規模のバランスから caformer_b36.dbv4-full をデフォルトとして使用する。

各プロファイルはモデル・タグCSV・前処理定義・カテゴリ定義・threshold定義を一つの論理単位として扱う。ultraだけはONNX変換済みrepoに前処理metadataがないため、metadataを`animetimm/convnextv2_huge.dbv4-full`から取得する。将来DBV4モデルを追加する場合も、このプロファイルへ定義を追加すれば共通推論経路を変更せず切り替えられる設計じゃ。

`compact_manual`と`medium_manual`はHugging Face管理者の承認後に利用できる。各モデルページでアクセス申請を行い、承認済みアカウントで`huggingface-cli login`または`hf auth login`を実行してから指定すること。承認待ち・未承認の場合はダウンロードできない。

| Profile | Params | 入力解像度 | Macro@Best F1 | 選定理由 |
| --- | ---: | ---: | ---: | --- |
| compact_manual | 30.4M | 384 × 384 | 0.510 | lightweightの0.511とほぼ同等でONNXが約36%小さい |
| medium_manual | 63.5M | 448 × 448 | 0.532 | lightweightより高精度でbalancedより小さい中間候補 |

### DirectMLのVRAM目安

| Profile | 入力解像度 | batch-size=4のVRAM目安 | 推奨VRAM |
| --- | ---: | ---: | ---: |
| compact_manual | 384 × 384 | 約1～1.5GB（推定） | 2GB以上 |
| lightweight | 448 × 448 | 約1～2GB（推定） | 2GB以上 |
| medium_manual | 448 × 448 | 約1.5～2.5GB（推定） | 3GB以上 |
| balanced | 384 × 384 | 約2～3GB（推定） | 4GB以上 |
| high | 448 × 448 | 約4～5.5GB（推定） | 6GB以上 |
| ultra | 512 × 512 | 約6.6GB（RTX 2070での実測） | 8GB以上 |

> [!WARNING]
> VRAM使用量はGPU、DirectML／ONNX Runtimeのバージョン、ドライバー、batch-size、同時使用中のアプリによって変動する。ultraは今回のRTX 2070・batch-size=4で約6.6GBを使用したため、8GB未満のGPUでは推奨しない。メモリ不足時は`-BatchSize 1`または`-BatchSize 2`を指定する。ただしbatch-sizeを下げてもモデル重み自体の常駐分は減らない。

lightweight～highの値は実機ロード値ではない。Hugging Faceで確認したFP32 ONNX容量、パラメータ数、公式入力解像度、およびultraの実測値から見積もった安全側の概算である。

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

初回セットアップ：

~~~bash
./run_tagger.sh
~~~

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

## Google Colab

run_colab_server.ipynb はDBV4サーバーを起動する構成へ更新されておる。

<a href="https://colab.research.google.com/github/SyameimaruKoa/wd14-tagger-xmp/blob/main/run_colab_server.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

初期状態では balanced プロファイルを使用する。ローカル側は通常のClientモードで接続できるぞ。

## 設定ファイル

初回実行時に config.json が自動生成される。既存の設定へ不足項目を追記する方式なので、旧設定ファイルが残っていても破壊しない。

主要項目：

| Key | 初期値 | 説明 |
| --- | --- | --- |
| model_profile | "balanced" | 使用するDBV4プロファイル |
| server_hosts | ["localhost", "google-colab", "100.xxx.xxx.xxx"] | Client接続先 |
| server_port | 5000 | Server/Clientポート |
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

Clientは model_id、metadata_version、output_size を検証してから結果を利用する。異なるDBV4モデルやmetadataを接続した場合はエラーとして停止するのじゃ。

## テスト

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
