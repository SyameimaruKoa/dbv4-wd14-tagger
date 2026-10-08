# WindowsのWSL Linuxコンテナ

[Issue #25](https://github.com/SyameimaruKoa/dbv4-wd14-tagger/issues/25)向けに、`run_tagger.ps1`へ`-Wsl`モードを追加した。セットアップ用の`setup_tagger_wsl.ps1`を呼び出し、コンテナ内でLinux版の`run_tagger.sh`を実行する。WindowsのPython環境を変更せず、Linux用の仮想環境とライブラリでStandalone / Server / Clientを動かす。

## 前提

- WindowsでWSL 2.9.3以降と`wslc`が利用できること。WSL 3.0.1で実機確認した。
- GPUを使う場合は、そのGPUに対応するWindowsのWSL対応ドライバーが必要。
- 初回のイメージ構築・Python依存導入・モデル取得にはネットワーク接続と空き容量が必要。

```powershell
wsl --version
wslc version
.\run_tagger.ps1 -Wsl -h
.\run_tagger.ps1 -Wsl -WslAction Build
```

WSLの更新が必要な場合は`wsl --update`を実行する。Docker Desktopは不要。WSLコンテナの要件と使い方は[Microsoftの公式説明](https://github.com/MicrosoftDocs/WSL/blob/main/WSL/wsl-container.md)を参照。

## 起動

ネイティブとWSLは同じ`run_tagger.ps1`を使い、WSLで実行する場合だけ`-Wsl`（`-ws`）を付ける。処理引数は共通関数で組み立てる。`setup_tagger_wsl.ps1`はイメージ構築・GPU公開・Windowsパスのマウント・コンテナ起動を担当し、推論オプションを重複定義しない。

`run_tagger.ps1 -Wsl`だけで実行するとLinux環境のセットアップのみを行う。イメージが存在しない場合は自動構築し、既存イメージの更新は`-WslAction Build`で行う。

共通オプションと短縮引数は既存の`run_tagger.ps1`に揃えている。
`-p`（Path）、`-j`（HostIP/HostName）、`-u`（Port）、`-m`（ModelProfile）、
`-ep`（Provider）、`-gi`（GpuIndex）、`-b`（BatchSize）、`-w`（IoWorkers）、
`-um`（ClientUploadMode）、`-o`（Organize）、`-t`（Tag）、`-z`（NoReport）、
`-r`（Recursive）、`-n`（NoRecursive）、`-f`（Force）などを直接指定できる。
`-s` / `-c` / `-lo`はServer / Client / Loginを選択する。通常処理は、既存版と同じStandaloneで画像を処理する。
`-g`はWindowsのGPUからProviderを選び、`-wg`はWebGPUを選ぶ。
DirectMLの`-di`はLinux非対応のため理由を表示して停止する。

```powershell
.\run_tagger.ps1 -Wsl -p 'C:\Images' -ep intel -m ultra -b 1 -z
.\run_tagger.ps1 -Wsl -s -ep intel -m ultra -u 5000 -pa 0.0.0.0
.\run_tagger.ps1 -Wsl -c -j 192.168.1.100 -p 'C:\Images' -um p
```

WSLコンテナ設定にも短縮引数を用意している。ModeとRemainingArgsは共通オプション。

| オプション | 短縮引数 |
| --- | --- |
| WslAction | `-ac` |
| Mode | `-md` |
| WslImage | `-im` |
| WslBaseImage | `-bi` |
| WslDataPath | `-dp` |
| WslGpuRuntimePath | `-gr` |
| WslWorkspaceVolume | `-vol` |
| WslContainerName | `-cn` |
| WslPublishAddress | `-pa` |
| RemainingArgs | `-ta` |
| WslInteractive | `-it` |

全オプションの短縮引数は`-Wsl -h`に表示する。`-np`（NcnnPrecision）、
`-ns`（NcnnPartSizeMiB）、`-wmm`（WallgenMoveMinRating）、
`-ra`（RemainingArgs）は既存PowerShell版にも追加した。長いオプション名は引き続き使用できる。

Intel GPUでultraの初期化を確認する例:

```powershell
.\run_tagger.ps1 -Wsl -Probe -Provider intel -ModelProfile ultra
```

Linuxの別PCから使う推論サーバー:

```powershell
.\run_tagger.ps1 -Wsl -Server -Provider intel -ModelProfile ultra -WslPublishAddress 0.0.0.0
```

Windowsファイアウォールで利用するネットワークからTCP 5000への接続を許可し、Linux側からWindows PCのIPを指定する。

```bash
./run_tagger.sh --client --host 192.168.1.100 --port 5000 --model-profile ultra --path /path/to/images
```

`-PublishAddress`の省略時は`127.0.0.1`へ公開する。`-Port`でServer/Clientのポートを変更できる。

Windowsの画像へ直接タグを保存する例:

```powershell
.\run_tagger.ps1 -Wsl -Provider intel -ModelProfile ultra -Path 'C:\Images' -Recursive -NoReport
```

`-Path`はWindowsの既存ファイルまたはディレクトリを指定する。ディレクトリは`/images`へ、ファイルは親ディレクトリを公開して`/images/ファイル名`へ変換する。スペース・日本語を含むパスも引数として保持する。XMP書き込みや整理による移動はWindows上の元画像へ反映される。

コンテナをClientとして使う場合:

```powershell
.\run_tagger.ps1 -Wsl -Client -HostIP 192.168.1.100 -Path 'C:\Images' -ModelProfile ultra
```

コンテナ内の`localhost`はWindowsホストと異なる。Windows上の別サーバーへ接続する場合も、到達可能なIPを指定する。

## 保存先・認証・更新

既定では名前付きボリューム`dbv4-tagger-wsl-workspace`にLinuxの`venv_*`、`config.json`、`.dbv4`を保持する。起動ごとにイメージ内のコードを更新コピーし、保存済みの設定・モデル・仮想環境は保持する。ソース変更後は`-WslAction Build`で再構築する。

Windows版の既存モデルとHugging Face認証を共有する場合:

```powershell
.\run_tagger.ps1 -Wsl -Probe -Provider intel -ModelProfile ultra -DataPath .\.dbv4
```

`-DataPath`は既存の`.dbv4`ディレクトリを`/workspace/.dbv4`へ公開する。Windows版の仮想環境や`config.json`は共有しない。同じデータをWindows版とコンテナ版で同時に更新しない。

独立した保存先へログインする場合:

```powershell
.\run_tagger.ps1 -Wsl -Login
```

Loginは端末の標準入力を接続する。通常の起動では標準入力を接続しないため、自動実行でも利用できる。対話入力が必要な場合は`-Interactive`を付ける。Linuxコンテナ内からWindowsのブラウザーを自動表示できない場合は、表示された認証URLをWindowsのブラウザーで開く。gated modelはアクセス承認が必要。

同じボリュームでは1コンテナずつ実行する。別環境を同時に動かす場合は`-WorkspaceVolume`と`-ContainerName`を両方別名にする。

```powershell
wslc stop dbv4-tagger-wsl
```

停止後にコンテナ本体は削除されるが、ボリュームとWindowsの画像・共有データは残る。

同じコマンドを再実行したときに同名のServerが稼働していれば、イメージ・処理引数・マウント・公開ポートを確認し、一致する場合は既存Serverを使用して直近のログを表示する。停止済みの本ツールのコンテナは、保存ボリュームを削除せず再作成する。設定が異なる稼働中コンテナは自動停止せず、停止コマンドを案内する。同名の別アプリのコンテナは変更しない。

## GPU対応の範囲

GPU Providerを指定すると`wslc --gpus all`で全GPUをコンテナへ公開する。`-GpuIndex`はその中から推論デバイスを選ぶ。CPU、Client、LoginではGPU公開を要求しない。Providerは明示指定なので、選択したGPU経路の初期化に失敗した場合は停止する。

| Provider | コンテナ内の条件 | 実機確認 |
| --- | --- | --- |
| `intel` | OpenCLドライバーとOpenVINO。標準イメージにIntel OpenCL ICDを含む | Iris Xe / ultraで確認 |
| `cuda` / `tensorrt` | NVIDIAのWSL対応ドライバー。CUDA 12用Python依存は既存ランチャーが仮想環境へ導入 | このPCにNVIDIA GPUがないため未検証 |
| `rocm` / `migraphx` | 対応AMD GPU、WSL用ROCmユーザー空間ランタイム、対応ONNX Runtimeが必要 | AMD実機未検証。標準Ubuntuイメージだけでは利用不可 |
| `ncnn` / `webgpu` | 実GPUに対応するLinux Vulkanドライバーが必要 | このPCではVulkanはllvmpipeのみ。実GPU対応は未確認 |
| `cpu` | 標準イメージで利用可能 | GPU公開不要 |

`--gpus all`だけで全ベンダーのLinuxバックエンドが有効になるわけではない。特にWindowsのDirectML EPはLinux版ONNX Runtimeで利用できない。従来の`run_tagger.ps1`とそのGPU経路はそのまま利用できる。

AMDなどで追加ランタイムが必要な場合、対応するUbuntu 24.04ベースのイメージを`-BaseImage`で指定して構築し、`-Image`で選択する。WSL向け共有ライブラリをWindowsのディレクトリへ配置した場合は`-GpuRuntimePath`で`/gpu-runtime`へ読み取り専用で公開できる。このディレクトリ、WSLホストライブラリ、`/opt/rocm/lib`をライブラリ探索へ追加する。カーネルドライバーはコンテナへ入れない。AMDの対象GPU・ドライバー・WSLランタイムは[公式互換表](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)と[公式WSLランタイム説明](https://github.com/ROCm/librocdxg)に合わせる。これらの拡張経路は実機検証が必要。

## 検証記録

2026-10-08、Windows 10.0.26300.9550 / WSL 3.0.1 / Intel Iris Xeで実施。

- `wslc`によるUbuntu 24.04イメージの構築と、`/dev/dxg`・WSLホストライブラリの公開を確認。
- `clinfo`でIntel GPU `0xa7a1`を認識。
- `-Probe -Provider intel -ModelProfile ultra -DataPath .\.dbv4`が終了コード0。起動検証の実行EPはOpenVINO。
- 実GPUのultra全12,476確率を保存済みCPU参照と比較し、許容誤差内の一致、採用タグの差分なし、4 ratingの誤差基準を確認。
- 実GPU統合テスト2件が成功。Standaloneと元画像・前処理済みClientの3経路で同じXMPタグを保存。
- 専用ランチャーでTCP 15025へ公開したServerへWindowsのClientから接続し、元画像と前処理済みの両転送でCPU参照との確率一致を確認。
- Windowsの単体テスト148件を実行し、失敗なし（環境条件による12件はスキップ）。追加ランチャーのテストは14件すべて成功。
- Vulkan診断はllvmpipeのみ。ncnn/WebGPUの実GPU動作としては扱わない。

起動引数・全GPU公開・ポート・日本語パス・ファイルパス変換・追加引数・終了コード・ヘルプ・共有データ・追加ランタイムは`tests/test_wsl_launcher.py`で検証する。
