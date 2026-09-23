#compdef run_tagger.sh

_dbv4_wd14_tagger() {
    local -a profiles
    profiles=(
        compact_manual
        lightweight
        medium_manual
        balanced
        high
        ultra
        wd14_v3
        future_1b
    )

    _arguments -s -S \
        '--provider[実行EP]:provider:(cpu cuda tensorrt intel webgpu migraphx)' \
        '--webgpu[WebGPUを使用]' \
        '--gpu-index[CUDA/TensorRT/MIGraphXの番号]:index:' \
        '--directml-device-index[DirectML DXGI番号]:index:' \
        '--webgpu-device-index[WebGPU番号]:index:' \
        '--target-vendor[GPU製造元を検証]:vendor:(nvidia intel amd)' \
        '--openvino-device[Intel GPU指定]:device:' \
        '--tensorrt-lib-dir[TensorRT 10ライブラリ場所]:directory:_files -/' \
        '(-p --path)'{-p,--path}'[処理対象ファイル/フォルダ]:path:_files' \
        '(-g --gpu)'{-g,--gpu}'[GPUを使用する（自動判別）]' \
        '(-I --force-intel)'{-I,--force-intel}'[Intel GPUを強制使用]' \
        '(-N --force-nvidia)'{-N,--force-nvidia}'[NVIDIA GPUを強制使用]' \
        '(-A --force-amd)'{-A,--force-amd}'[AMD GPUを強制使用]' \
        '(-o --organize)'{-o,--organize}'[フォルダ整理を行う]' \
        '(-t --tag)'{-t,--tag}'[タグ付けも行う]' \
        '(-x --pixiv)'{-x,--pixiv}'[Pixiv整理モード]' \
        '(-R --no-report)'{-R,--no-report}'[レポートを作成しない]' \
        '(-r --recursive)'{-r,--recursive}'[再帰検索を有効化]' \
        '(-n --no-recursive)'{-n,--no-recursive}'[再帰検索を無効化]' \
        '(-b --batch-size)'{-b,--batch-size}'[推論バッチサイズ]:batch size:' \
        '(-w --io-workers)'{-w,--io-workers}'[前処理の並列ワーカー数]:workers:' \
        '(-m --model-profile)'{-m,--model-profile}'[モデルプロファイル]:profile:($profiles)' \
        '(-e --model-repo)'{-e,--model-repo}'[Hugging FaceリポジトリID]:repository:' \
        '(-M --model-file)'{-M,--model-file}'[モデルファイル名またはパス]:model file:_files' \
        '(-T --tags-file)'{-T,--tags-file}'[タグCSVファイル名またはパス]:tags file:_files' \
        '(-q --thresh)'{-q,--thresh}'[タグ閾値]:threshold:' \
        '(-f --force)'{-f,--force}'[既存タグを強制的に再解析・上書き]' \
        '(-s --sensitive-split-mode)'{-s,--sensitive-split-mode}'[旧CLI互換の分割モード]:mode:(2 4 6)' \
        '(-c --record-ratio)'{-c,--record-ratio}'[RAW・割合スコアタグを記録]' \
        '(-C --no-record-ratio)'{-C,--no-record-ratio}'[RAW・割合スコアタグを記録しない]' \
        '(-S --server)'{-S,--server}'[サーバーモード]' \
        '(-K --client)'{-K,--client}'[クライアントモード]' \
        '(-L --login)'{-L,--login}'[Hugging Faceログインモード]' \
        '(-H --host)'{-H,--host}'[サーバーのIPアドレス]:host:' \
        '(-P --port)'{-P,--port}'[ポート番号]:port:' \
        '(-d --debug)'{-d,--debug}'[GPU/OpenVINO詳細デバッグログを有効化]' \
        '(-h --help)'{-h,--help}'[ヘルプ表示]' \
        '*:path:_files'
}

_dbv4_wd14_tagger "$@"
