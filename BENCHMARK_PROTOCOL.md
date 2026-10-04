# AI向け統一測定・記録指示書

仕様バージョン: **1.0** / 作成日: **2026-10-04（Asia/Tokyo）**

## 1. この指示書を渡されたAIへの依頼

このリポジトリの画像タグ付け処理について、指定された複数の実機・OS・GPU・モデルを、以下の条件で測定し、再現・比較可能な記録を作成してください。特定のPC、OS、GPUメーカー、SSH名、保存パスを共通条件として固定しないでください。接続先とプロジェクトパスは測定依頼時の対象一覧から取得してください。

測定開始前に対象一覧、実行条件、取得可能な指標、使用するツールが本仕様を満たすかを確認してください。条件を黙って変更したり、取得不能な値を推定して実測値として保存したりしないでください。過去の記録を保持し、新しい測定は専用ディレクトリに保存してください。

**この文書を作成する作業と、文書に従って測定する作業は別です。** 本文の実行指示は、後で測定を依頼されたAIに適用されます。文書の作成・確認だけを依頼された場合、モデル変換やベンチマークを開始しないでください。

### 優先原則

1. 速度・メモリ・精度・変換時間を、定義を揃えて記録する。
2. メモリの主指標は本プログラムに帰属する量。OS全体・GPU全体の使用量で代用しない。
3. 各OSで取得定義が異なる指標は、名前を統一しても同一物とは扱わない。
4. 実行成功、精度合格、記録充足を独立して判定する。
5. 未測定、失敗、欠測、対象外、0を区別する。
6. 試行・診断・warmupを正式な反復統計に混ぜない。
7. 実装不足で仕様を満たせない工程は保留して理由を残す。仕様準拠を偽らない。

「必須」は未実施なら記録不足。「取得可能なら」は理由付き欠測を許容。「観測ピーク」は取得サンプルの最大であり、サンプル間の瞬間最大ではありません。

## 2. 対象一覧と識別単位

### 2.1 対象を確定する

測定依頼に含まれる各ホストについて、以下の表を作成してください。OSの再起動・切替を要する環境は、それぞれ別の測定環境として扱います。

| host_id | 接続先 | プロジェクトパス | OS | 物理GPU | 接続可否 | 備考 |
|---|---|---|---|---|---|---|
| 記入 | 記入 | 記入 | 実機確認 | 実機確認 | 記入 | 仮想・ソフトウェアGPUを区別 |

- 対象モデルは測定時点の `dbv4.MODEL_PROFILES` 全エントリ。名称を固定リストで置き換えない。
- batchは1・4。手動承認が必要なモデル、未公開モデルも列挙し、取得不能の理由を残す。
- CPUと、対象OS・GPUで候補となるbackendを列挙する。既存マトリクスだけに依存せず、ncnn等も確認する。
- 必要な承認・利用権がないモデルは未実施とする。別モデルで代用しない。
- 各backendの既定精度、提供される明示的な精度モード、既存記録で用いた分割・同期条件を候補表へ列挙する。探索中に見つけた設定は追加ケースとして記録する。
- 部分測定の依頼がない限り、失敗・非対応も含めて候補表の全ケースに状態を付ける。

### 2.2 単位とID

| 単位 | 定義 |
|---|---|
| campaign | 同じ仕様による一連の測定 |
| host/environment | 物理実機とOS・ドライバ構成 |
| case | 実機、OS、モデル、backend、デバイス、精度、batch、分割、CPU併用方針、キャッシュ条件の組合せ |
| attempt | caseへの試み。再試行は別attempt |
| set | 新規プロセスで初期化から正式反復終了まで |
| iteration | batch全体の処理1回 |
| conversion | 固定した変換条件でのモデル変換1回 |

IDは `campaign-YYYYMMDDTHHMMSSZ-<suffix>` と、その配下の `host-001 / case-0001 / attempt-001 / set-01` 等を使い、manifestに完全な条件を保存します。ID・ファイル名だけを条件の出典にしません。

同じcase内でコード・モデル・精度・分割・同期・電源設定を変更しません。変更した場合は新caseです。実測で判明したCPU併用状態も保存し、異なる状態を混ぜて集計しません。

## 3. 実行前チェックと環境記録

### 3.1 必須のチェック

- [ ] 接続先、作業ディレクトリ、コードの識別情報を確認した。
- [ ] 測定対象GPUと監視対象GPUを物理識別情報で照合した。
- [ ] モデル、外部重み、metadata、入力画像が揃っている。
- [ ] 出力先を新規作成でき、必要容量を確保できる。
- [ ] 計時・同期・独立プロセス・反復回数が仕様に対応する。
- [ ] 対象PID群のRAM/GPU指標の取得可否を確認した。
- [ ] 同じ実機で別の測定・変換を実行していない。
- [ ] 実行中のログ・結果・監視記録を残す方法がある。

不足する計測機能がある場合、対応表に「未対応」と記録します。ツール改修を依頼されている場合のみ必要な改修を行い、そのコードを固定してから正式測定します。存在しないオプションやAPIを手順に使いません。

### 3.2 保存する環境情報

- ISO 8601の開始・終了日時、UTCオフセット、タイムゾーン名。所要時間は壁時計でなく単調時計を使用。
- hostname、機種、CPU名・コア構成、RAM・swap容量、GPU名・容量。
- OS名・バージョン・build/kernel・アーキテクチャ。
- Python、NumPy、Pillow、ONNX Runtime、OpenVINO、ncnn、PyTorch、pnnx等の使用バージョン。別仮想環境は別に記録。
- コミット、tracked差分、関係するuntrackedファイル一覧とハッシュ。コードが異なる実機の結果を同一コードとして扱わない。
- モデル取得元・revision、元重み・外部重み・ラベル・前処理・閾値・categories等のハッシュと容量。
- 実際のコマンド、作業ディレクトリ、関連環境変数、スレッド数、provider options、分割・同期・最適化設定。
- 電源接続、電源モード、冷却、共有メモリ設定、出力ファイルシステム。取得不能なら理由を保存。
- 工程ごとのtimeout。既定はnull（制限なし）。中断は可能だが完了とは扱わない。

認証トークン、秘密鍵、パスワードを環境記録・コマンド・差分に収録しません。

### 3.3 デバイスとCPU併用

backendのGPU番号と監視ツールのGPU番号は別々に保存します。UUID、PCIアドレス、DXGI LUID、DRM node等を使って対応を記録してください。GPU名・番号の一致だけでは同一と断定しません。仮想ディスプレイ、llvmpipe等を物理GPUと区別します。

要求したbackend、利用可能provider、session登録provider、実際に演算が割り当てられたproviderを別項目にします。CPU providerが登録されているだけではCPU併用と判定しません。

CPU併用状態は `none_verified / mixed_verified / unknown`。割当を確認できない場合はunknown。profiling等は正式計時とは別の診断実行で行い、コード・設定が同じことを記録します。GPUへの入力転送等のホスト処理とCPU演算へのフォールバックも区別します。

要求GPUと実デバイスが一致しなければ停止します。別GPUやCPUに切り替えて同じcaseとして続行しません。

## 4. 共通画像と前処理

### 4.1 画像生成

campaignごとに一度だけ以下で生成し、同じPNGを各環境へ配布してください。

```python
import numpy as np
from PIL import Image
rng = np.random.Generator(np.random.PCG64(20260922))
pixels = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
image = Image.fromarray(pixels)
assert image.mode == "RGB"
image.save("benchmark-input.png", format="PNG", compress_level=6, optimize=False)
```

高さ480、幅640、RGB、uint8、seedは20260922。追加metadataを付けず再圧縮しません。NumPy/Pillowバージョン、PNGファイルSHA256、`pixels.tobytes(order="C")` のSHA256を保存します。配布後にファイル・画素のハッシュを検証します。

従来の単色画像、座標式画像、異なる乱数生成器の画像は同じ入力ではありません。

### 4.2 モデル別前処理

モデルの正式な前処理を使います。モデル間のサイズ、RGB/BGR、正規化を揃えるために変更しません。前処理コードと設定を保存します。

共通前処理直後とbackend入力直前について、shape、dtype、軸順、バイト順、C連続バイト列SHA256を保存します。backend用のレイアウト・dtype変換がある場合は両方残します。ハッシュ取得は計時外です。

batch 4は同じ画像4枚。全体処理では4画像として通常経路で読み込み・前処理し、画像1枚分をコピーして前処理時間を省略しません。推論単体は事前に作成した同じbatchを反復使用します。

## 5. 測定順序とプロセス数

1. モデル取得、必要な変換・構築、ファイル検証を事前に完了する。
2. caseごとに事前試行を実施する。
3. 同一モデル・metadata・入力のCPU FP32基準出力を準備する。
4. 推論単体を3セット実施する。
5. 全体処理を別プロセス群で3セット実施する。
6. 精度、集計、記録充足を判定する。

同じ実機では逐次実行します。別実機は、共有ストレージ・ネットワーク等の競合を起こさなければ独立に実行可能です。

### 5.1 事前試行

新規プロセスで初期化し、明示的な初回推論1回、その後3回を実行します。後者3回のms/枚中央値で区分します。

- 10,000ms/枚未満: normal。
- 10,000ms/枚以上: slow。

試行は正式統計に含めません。失敗時は区分を推定せず、失敗として記録します。区分は同じcaseの推論単体・全体処理に共通で適用します。

### 5.2 正式セット

| 工程 | normal | slow |
|---|---:|---:|
| 明示的な初回処理 | 1 | 1 |
| 初回後の追加warmup | 3 | 1 |
| 正式iteration | 20 | 5 |
| 独立set | 3 | 3 |

推論単体・全体処理それぞれでこの回数を実施します。初回処理はwarmupに含めず、初回とwarmupは正式統計から除外します。初期化内部にprobeがあれば記録し、「完全に初めてのGPU実行」と断定しません。

各setは新規主プロセスと処理用子プロセス群で実行し、終了後に終了を確認します。プロセス・モデルオブジェクトをset間で共有しません。初期化、初回、warmupに失敗したsetは正式反復に進みません。

中断したsetの残りを別プロセスで継ぎ足しません。再試行は新attemptで記録します。

## 6. 計時境界・出力形式・キャッシュ

生時間は単調高分解能時計の整数nsで保存します。時計の名前、精度、時刻原点の識別子を残します。別ホストの単調時刻を直接比較しません。

| 指標 | 開始 | 終了 |
|---|---|---|
| process_startup | 外部制御側のプロセス生成開始 | 子の初期化開始通知 |
| model_initialization | モデル・metadata読み込み開始 | 入力処理可能 |
| ready_total | 外部制御側の生成開始 | 入力処理可能の通知 |
| first_operation | 初回の処理呼出し直前 | 下記のmode別終了境界 |
| inference | 前処理済みbatchで推論を呼ぶ直前 | 全画像の確率がCPU上で参照可能 |
| pipeline | batchの最初のPNGを開く直前 | 全画像のタグJSONをcloseした直後 |

取得できない境界はnull。model_initializationをprocess_startupの代用にしません。包含する工程を二重加算しません。

### 6.1 推論単体

入力転送、backend内部の必要な割当・変換、演算、出力転送、同期を含みます。PNG読み込み、モデル共通前処理、タグ判定、結果保存は除外します。backend入力に必要なlayout/dtype変換を計時前に行う場合、その変換内容を明記します。

GPU非同期処理は終了を待ちます。enqueue時間を推論時間としません。ncnnのbatch 4が逐次4回なら、全4回を1iterationとして計時し、`batch_execution=sequential` と保存します。

### 6.2 全体処理

モデル初期化済みで、通常経路の画像読み込み、前処理、推論、タグ生成、ファイル保存を測ります。出力先はローカル専用領域。ネットワーク出力は別caseです。

画像ごとの出力は次の形式に固定します。

```json
{"image_id":"image-00","selected_tags":["tag_a","tag_b"],"rating_scores":null}
```

- UTF-8、BOMなし、改行LF、末尾LF1つ。JSONは空白なし、キー順は上記順。
- タグは正式閾値で採用した名前を重複除去しUnicodeコードポイント順に並べる。
- ratingがある場合は名前を同じ順に並べたオブジェクト。ない場合はnull。
- 全確率はこの出力に含めず、精度資料として計時外で保存する。
- iterationごとに新しいディレクトリを用意する。ディレクトリ作成と後片付けは計時外。
- 各画像に `image-00.json` 等を作成し、closeまでを含める。skipや既存ファイル再利用は禁止。
- fsyncを含めず、永続化保証の時間とは呼ばない。
- XMPや元画像の書換えは本pipelineに含めない。実アプリのXMP処理を追加測定する場合は別modeとする。

### 6.3 キャッシュ

OSファイルキャッシュは強制削除しません。初回読み込みとwarmup後の反復を区別し、反復は通常キャッシュ状態の測定として扱います。

モデル・変換結果・backend/engine・OSファイルキャッシュを別々に記録します。正式反復は必要なbackendキャッシュを事前構築した状態を基本とし、全setで同じ条件にします。構築と推論を分離できない場合は明記します。

初回構築なしの新規backendキャッシュでの起動は別case。新規プロセスを「全キャッシュなし」と呼びません。

## 7. 本プログラムに帰属するメモリ

### 7.1 対象と主指標

対象は測定主プロセスと処理用子孫プロセス。外部制御・監視、他アプリ、OS全体を除外します。プログラム初期状態のランタイム・ライブラリも対象に含め、初期値を差し引きません。増分は補助項目に限ります。

RAMの主指標は専有常駐量。共有按分、commit、RSSを別々に保存します。「総メモリ」という単一値へまとめません。

| 指標 | Linux候補 | Windows候補 | 意味・扱い |
|---|---|---|---|
| private_resident_bytes | Private_Clean + Private_Dirty | WorkingSetPrivate / Private Working Set | 専有常駐量、主指標 |
| pss_bytes | Pss | 取得不能ならnull | 共有按分込み常駐量 |
| shared_apportioned_bytes | 同一読取りのPss−専有量 | 取得不能ならnull | 共有分の按分 |
| resident_including_shared_bytes | Rss | WorkingSet | 共有を含む補助指標 |
| private_commit_bytes | 定義対応の取得手段があれば保存 | PrivateBytes | 非常駐分を含み得るcommit、常駐量の代用不可 |
| private_hugetlb_bytes | Private_Hugetlb | 対応手段があれば保存 | 通常指標と分ける |
| swap_bytes / swap_pss_bytes | Swap / SwapPss | 対応定義がなければnull | OS全体swapで代用不可 |

Linuxは `/proc/<pid>/smaps_rollup` を第一候補とし、kBを1,024 bytesで換算します。取得不能ならsmaps等の定義が一致する手段を調べ、取得元を記録します。巨大ページの包含範囲を残し、無条件で加算しません。派生共有量が丸めで負になった場合はnullとし、元値を残します。

Windowsはローカライズされたカウンタ名を決め打ちせず、CIMクラス等を能力確認します。WorkingSetPrivateとPrivateBytesは別の値です。

macOS、その他のOS、コンソール系Linuxは、利用可能な取得手段の定義を確認します。resident_size、footprint、システム全体差分をUSSと同じ名前に置換しません。対応する指標がなければ理由付き欠測とします。

### 7.2 プロセス群

PIDと生成時刻で識別し、親子関係と開始・終了を保存します。GPUカウンタのPID再利用も同様に確認します。

群合計は同一監視周期の生存対象プロセスから計算します。群ピークは各周期の群合計の最大で、各PIDの個別ピークの和ではありません。RSSの単純合計を専有量にしません。終了中で取得できないPIDがあれば、その周期の群合計を不完全として扱います。

同時読取りできない取得時刻の幅、捕捉できなかった短命子プロセスの可能性を残します。外部監視を優先し、プロセス内監視の消費量が含まれる場合は分離不能と記録します。

### 7.3 GPUメモリ

対象PID群か、重複除去された対象clientに帰属する値だけを採用します。

- 専用と共有を別々に保存し、取得元が「プロセスGPU使用量」のみなら、その定義の別指標で保存する。
- NVIDIAではNVML/nvidia-smi等のプロセス別インターフェースを確認する。compute一覧だけでVulkan等の取得を保証しない。
- WindowsではGPU Process MemoryのPID・LUID・physical adapterを照合し、DedicatedUsage、SharedUsage、TotalCommittedを別項目にする。TotalCommittedを専用＋共有の使用量に置換しない。
- DRMでは対象fdinfoのdriver、client-id、device、memory regionと単位を保存する。同じclientの複数fdを重複加算しない。
- デバイス全体のVRAM/GTT値、起動前後差分、予約容量で補完しない。
- PID/clientが見つからない、N/A、権限不足、取得失敗はnull。取得APIが明示的に返した有効な0のみ0とする。
- プロセス間共有割当の重複を除去できなければ、個別値を残し群合計はnull。
- RAMと共有GPUメモリは重複が不明なら加算しない。

GPUメモリの能力確認は `interface_available / target_value_observed / monitored` を区別します。インターフェースが動くだけで全backend対応と判定しません。取得不能でも速度・精度の測定は継続します。

### 7.4 監視周期と工程

目標周期200ms。取得開始・終了、サンプル時刻、PID、指標、取得元、欠測理由を保存します。予定時刻に間に合わなくても架空サンプルを作りません。

工程は `pre_init / initialization / first_operation / warmup / measurement / teardown`。境界をまたぐ取得はboundaryとし、set全体には残しますが工程別値へ強制割当しません。

工程別とset全体に、観測ピーク、サンプル数、最小・中央値・最大取得間隔、取得失敗数を記録します。200ms監視で200ms未満の瞬間ピークを保証しません。補間・前回値で埋めません。

GPU稼働率、温度、クロック、電力は取得可能なら補助記録とし、デバイス全体かプロセス単位かを明記します。主メモリ指標へ混ぜません。

## 8. モデル変換・エンジン構築

変換は各実機・元モデル・変換器・引数・精度・入力shape・変換方式につき**1回**。未測定変換を推論初期化の時間から推定しません。

開始状態は元ファイル取得済み、必要依存導入済み、新規空出力先、変換結果キャッシュなし。既存成果物は削除せず、専用領域を使います。

総時間は外部からの変換プロセス生成開始から、成果物書き込みを終えてプロセスが正常終了するまで。モデル読み込み、trace/export、pnnx等の変換、成果物配置を含みます。

ダウンロード、環境導入、変換後の精度検証、記録用ハッシュ計算は除外します。変換器内部に必要な検証が含まれる場合は包含関係を明記します。

段階別時間は取得できる範囲で `source_load / trace_export / converter / publish_outputs` 等のイベントを残します。重複工程を合算しません。1回なので平均・p95・標準偏差を出しません。

保存するもの: 元ファイル・成果物のハッシュと容量、変換器/関連ライブラリのバージョン、コマンド、設定、総時間、段階時間、対象プロセス群メモリ、stdout/stderr、終了コード、signal、残存成果物。

キャッシュhit、skip、既存成果物のコピーは新規変換時間として扱いません。失敗は経過時間と到達工程を残します。OOMはOSログ等で確認した場合のみ確定し、signalによる終了だけなら未確定とします。失敗成果物は利用可能として公開しません。

TensorRT/OpenVINO等のengine構築・compileは別kindで記録します。初期化内に含まれる場合、初期化総時間と構築時間の包含関係を保存します。分離不能なら個別構築時間はnullです。

## 9. 精度検証

基準は同じ元モデルのONNX Runtime CPU FP32。モデル、metadata、ラベル順、画像、共通前処理、batchが一致する基準を作成します。backendのlayout/dtype変換は別途記録します。FP32基準を準備できなければ理由を残して未確認とします。

batch 1・4はそれぞれ基準を用意し、全画像・全ラベルを比較します。推論単体・全体処理の各setの初回処理と最後の正式反復出力を保存します。全体処理でもタグ生成前の確率を保持し、保存・比較は計時外です。

合格には以下すべてが必要です。

1. shape、ラベル順、基準識別情報が一致。
2. NaN/Infなし。
3. 全確率で `abs(actual-reference) <= 1e-4 + 1e-3*abs(reference)`。
4. ratingがあるモデルは全rating絶対差が1e-4以下。
5. 同じ閾値・decode設定による採用タグ集合が一致。

低精度も同じ基準で判定し、勝手に許容差を緩めません。入力・ラベル等不一致はinvalid、数値やタグの不一致はfailed。ratingがないモデルは対象外です。

全確率、最大・平均絶対差、超過要素数、最大差の画像/ラベル/基準値/実測値、rating差、追加タグ、欠落タグを保存します。setの初回と最終で異なる場合は両方残します。

精度不合格の速度は保存し、合格表と分けます。合成画像1種の一致を実画像全般の精度保証とは記載しません。

## 10. 集計・比較規則

- batch時間: 終了ns−開始ns。
- ms/枚: batch時間ns÷1,000,000÷batch。
- 平均: 算術平均。
- 中央値: 偶数件は中央2値の平均。
- p95: 昇順n値の位置 `(n-1)*0.95` を線形補間。
- 標準偏差: 母標準偏差、分母n。
- 枚/秒: 正式反復の総画像数÷正式反復の総時間秒。

set別と3set全体を保存します。全体は生反復を結合して計算し、set中央値の平均で代用しません。set間待機、warmup、初期化を枚/秒に含めません。

外れ値は自動削除しません。競合処理が確認されたら注記し、再試行も元attemptを保持します。3set完了しなければ正式な全体集計はnullとし、完了setを参考値として掲載します。

計算前には丸めず、表示は時間2桁、MiB2桁、使用率1桁を基本とします。生bytes/nsを残します。

比較時は変える軸と固定した軸を明記します。normal/slow、mode、入力、精度、分割、CPU併用、batch実行方式、キャッシュ条件を混ぜません。モデル間比較はモデルと正式前処理の差を明記します。OS間のメモリは定義・取得範囲が異なれば列を分けます。時間差をbackend差だけに帰属させません。

## 11. 保存契約とテンプレート

```text
benchmarks/unified/<campaign_id>/
  manifest.json
  inputs/benchmark-input.png
  environment/<host_id>/
  references/<reference_id>/
  conversions/<conversion_id>/
  cases/<case_id>/<attempt_id>/
    result.json
    stdout.log
    stderr.log
    pilot/
    inference/set-01/ ... set-03/
    pipeline/set-01/ ... set-03/
  summary.md
```

各setに `events.jsonl / timings.jsonl / resources.jsonl / processes.json / accuracy.json / probabilities.npz / summary.json` を保存します。probabilitiesには画像・ラベルの軸情報を付けます。NPZを使えない環境では同じdtype/shapeを復元できる形式に変更し、manifestに形式と理由を保存します。

結果ファイルは進行中と完了を区別し、可能なら一時ファイルからrenameします。失敗時も取得済みファイルを保持します。

### 11.1 manifestの必須項目

`schema_version, campaign_id, protocol_version, created_at, timezone, hosts, models, input, cases, conversions, files`。

hostsは接続先・パス・環境情報参照、casesは完全条件・ID・予定回数・attempt一覧、filesは相対パス・SHA256・サイズ・役割を持たせます。基準出力もreference_idと完全条件で追跡します。

### 11.2 反復記録の形

```json
{"set_id":"set-01","mode":"inference","phase":"measurement","iteration":0,"batch_size":4,"start_ns":1000000000,"end_ns":1200000000,"duration_ns":200000000,"status":"completed"}
```

上記は書式例で実測値ではありません。iterationは工程ごとに0始まり。失敗反復はstatusとreasonを残し、正式統計に入れません。

### 11.3 指標記録の形

```json
{"metric":"private_resident_bytes","value":null,"unit":"bytes","scope":"process_tree","source":"proc_smaps_rollup","reason":{"code":"permission_denied","detail":"対象PIDの読取り不可"},"processes":[],"collection_start_ns":1000000000,"collection_end_ns":1001000000}
```

値を取得した場合reasonはnull。scope、source、対象PID、時刻、単位は省略しません。指標の定義はenvironmentのmetric_definitionsに保存します。OSの指標名を生データとしても残します。

欠測codeは `unsupported / permission_denied / collection_failed / not_measured / not_applicable / boundary_unavailable / attribution_unknown / target_not_observed`。取得不能を0にしません。

### 11.4 attempt結果の形

```json
{
  "schema_version": "1.0",
  "protocol_version": "1.0",
  "case_id": "case-0001",
  "attempt_id": "attempt-001",
  "execution_status": "not_started",
  "accuracy_status": "not_checked",
  "record_status": "incomplete",
  "reason": null,
  "speed_class": null,
  "reference_id": null,
  "device_mapping": null,
  "cpu_execution": "unknown",
  "sets": [],
  "aggregate": null,
  "missing_required": [],
  "missing_optional": [],
  "files": []
}
```

開始後は日時、完全条件参照、各setの実行状態・exit code・精度・ファイル参照を記入します。失敗理由とログ参照を省略しません。GPUメモリ欠測は理由を保存すれば任意項目欠測として扱えます。時間の生データ、固定条件、同期確認等の必須情報不足はrecord_status=incompleteです。

### 11.5 比較表

| 実機・OS | モデル | backend/実デバイス | 精度/分割 | batch/実行方式 | mode/区分 | 完了set | 中央値ms/枚 | p95 | 枚/秒 | 精度 | 出典 |
|---|---|---|---|---|---|---|---:|---:|---:|---|---|

| case/set/工程 | RAM専有MiB・定義 | 共有按分MiB | RSS/WS MiB | GPU専用MiB | GPU共有MiB | 観測間隔 | 欠測理由 | 出典 |
|---|---|---|---|---|---|---|---|---|

| 実機・OS | モデル | 変換器/設定 | kind | 総時間秒 | 段階時間 | 専有RAM観測ピーク | 成果物容量 | 状態 | 出典 |
|---|---|---|---|---:|---|---|---|---|---|

表には測定条件と定義を併記し、成功・不合格・未完了を分けます。

## 12. 失敗・中断と完了判定

- 非対応: 候補caseをunsupportedとして残す。
- 取得権不足: モデル/環境の不足理由を残し、別物で代用しない。
- デバイス不一致: 停止し、要求・実デバイス・対応付けの証拠を保存。
- 精度不合格: 時間と資源は残し、不合格として分ける。
- OOM: 証拠、終了コード、signal、失敗工程を保存。原因未確定も許容する。
- 中断/timeout: completedにしない。正式全体集計を作らない。
- 再試行: 新attempt。旧結果を削除・更新して成功へ読み替えない。

### 完了チェック

- [ ] 全host・model・caseに状態がある。
- [ ] 正式完了caseは両mode各3setと予定反復が揃う。
- [ ] 画像、モデル、metadata、コードが出典に追跡できる。
- [ ] 実デバイス・CPU併用・同期・batch方式が記録される。
- [ ] 精度の基準と全batch比較が残る。
- [ ] メモリは専有・共有・commit・デバイス全体を混同しない。
- [ ] 欠測理由と監視範囲・周期を保存した。
- [ ] 変換/構築は新規処理とcache再利用を分けた。
- [ ] 失敗・中断・再試行の原本を保持した。
- [ ] summaryは生データから再計算でき、リンクが有効。

全caseが成功することをcampaign完了条件にはしません。全候補に状態・理由が付き、未完了と成功が明確に区別されていれば、対象範囲の記録は完了です。

最終報告では測定完了数、非対応数、失敗数、中断数、精度不合格数、必須記録不足数、主要な取得制限と成果物へのリンクを示します。

## 13. 記録例による判断の固定

以下は規則例であり、実測結果ではありません。

| 状況 | 実行 | 精度 | 記録 | 扱い |
|---|---|---|---|---|
| normal、両mode各3set完了、精度一致 | completed | passed | complete | 合格表 |
| slow、各set5反復完了 | completed | passed | complete | slow表、normalと混ぜない |
| GPU帰属値取得不能、RAM/時間/精度取得 | completed | passed | complete | GPU=null＋理由、RAMは残す |
| 子プロセス読取り不可で群RAM不明 | completed | passed | complete | RAM群値=null＋理由、観測できた個別値を残す |
| 正式時間完了、タグ不一致 | completed | failed | complete | 精度不合格表 |
| 変換1回成功 | completed | not_checked | complete | 変換時間のみ。検証は別工程 |
| OSでOOM確認 | failed | not_checked | complete | 経過時間と証拠、成果物を公開しない |
| 2set完了、3set目中断 | interrupted | not_checked | complete | aggregate=null、完了setは参考値 |
| 要求GPUと別GPU | failed | not_checked | complete | device_mismatch、測定停止 |
| 推論完了だが同期・生時間の証拠なし | completed | not_checked | incomplete | 正式な比較結果にしない |

## 14. 既存ツールの対応上の注意

この節は文書作成時のコード確認に基づきます。測定時点で再確認してください。

| ツール | 現状と本仕様への不足 |
|---|---|
| benchmark_ncnn.py | 座標式合成画像、通常1set、主にRSS、GPU全体値、精度出力は先頭画像中心。共通画像・独立3set・全batch精度・プロセス帰属監視への対応確認が必要 |
| benchmark_nvidia_ep.py / benchmark_matrix.py | seed画像を使用するが3setは同じプロセス内。独立set、計時境界、初回、全体処理、RAM専有/GPU帰属値の対応が必要 |
| benchmark_webgpu_memory.py | 単色画像、独自反復数・監視間隔、GPU全体値。新仕様そのままには非準拠 |
| convert_ncnn_model.py | 変換・cache再利用を区別できるが、外部総時間・段階イベント・プロセス群資源記録の対応確認が必要 |
| 既存集約スクリプト | 新スキーマ・指標定義・状態区分への対応確認が必要。旧結果を新仕様値に変換しない |

上記ツール名だけを根拠に本仕様準拠としません。対応済み機能・不足機能・使用コードをcampaignの事前チェックに保存してください。

## 15. 実機での読取り確認の証拠と限界

以下は取得手段の確認例です。全実機にこの構成を要求するものではなく、後の測定値の代用にもなりません。確認日: 2026-10-04。

### Linux例

- 接続先: `komeiziaya@ubuntu-komeiziaya.bass-uaru.ts.net`。
- Ubuntu 26.04.1 LTS、kernel 7.0.0-38-generic、i7-8750H、RTX 2070 Max-Q＋UHD 630。
- NVIDIA driver 610.57.04、VRAM 8192 MiB。
- HEAD: `6b518dc75d15aaeba57accdc3a08dd5b92584435`。
- 読取り用Python自身のsmaps_rollup: Rss=12932 KiB、Pss=7084 KiB、Private_Clean=184 KiB、Private_Dirty=5764 KiB。タグ付け処理の使用量ではない。
- NVIDIA compute-apps問い合わせは成功したが稼働対象不在。CUDA/TensorRT/Vulkan対象PIDの使用量取得は未確認。
- 読取り可能な範囲にDRM clientを検出せず、DRM帰属メモリは未確認。
- 稼働率・温度・電力・クロックはデバイス全体の補助値として取得成功。
- 既存ncnn結果はbackend GPU index=1、監視NVIDIA index=0。番号の共通化は不可。

### Windows例

- 接続先: `a13m`、プロジェクト: `C:\Users\kouki\Documents\MyApp\wd14-tagger-xmp`。
- Windows 11 Home、version 10.0.26300、build 26300。Windows PowerShell 5.1。
- i7-1355U、Intel Iris Xe、driver 32.0.101.7092。複数の仮想ディスプレイも列挙された。
- HEAD: `9210a1626dc26e05f9cf663faf9a6cdd50b7b134`。Linux例とコードが異なる。
- `run_tagger.ps1` と `benchmark_matrix.ps1` のPowerShell構文解析はエラー0。
- Python 3.13.15、ONNX Runtime 1.24.1。OpenVINOExecutionProviderとCPUExecutionProviderを列挙。
- OpenVINO 2025.4.1はCPUとIris Xe GPUを列挙できた。今回、新規モデル推論は実行していない。
- CIM PerfProcで確認用PowerShell PID 32172のWorkingSet=86310912 bytes、WorkingSetPrivate=32419840 bytes、PrivateBytes=68440064 bytesを取得。タグ付け処理のメモリではない。
- CIM GPUProcessMemoryで既存PIDの実数値取得に成功。例: PID 17376、LUID `0x00000000_0x0001442B`、DedicatedUsage=0、SharedUsage=46104576、TotalCommitted=67268608 bytes。
- 上記GPU値は対象タグ付けプロセスの値ではなく、物理GPUへの完全な対応付けも今回未実施。測定時にPIDとLUIDを照合する。

この確認はOS情報、指標取得、スクリプト構文、backend導入・デバイス列挙の確認です。全backendでのモデル推論、正式ベンチマーク、モデル変換、対象プロセスの監視ピーク取得に成功した証拠ではありません。
