#!/bin/bash

# 下にヘルプがあるぞ

# ==========================================
# WD14 Tagger Universal (Bash Wrapper)
# ==========================================

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/embed_tags_universal.py"

# デフォルト値
USE_GPU=0
FORCE_TYPE="auto" # auto, nvidia, intel, amd
PROVIDER=""
TENSORRT_LIB_DIR_ARG=""
WEBGPU_MODE=0
PY_ARGS=()
DO_ORGANIZE=0
DO_TAG=0
DO_PIXIV=0
IS_CLIENT=0
DEBUG_MODE=0
LOGIN_MODE=0

configure_storage_paths() {
    if [ ! -d /tmp ] || [ ! -w /tmp ]; then
        echo "[ERROR] 一時フォルダ /tmp を使用できません。"
        exit 1
    fi
    export TMPDIR=/tmp
    export TMP=/tmp
    export TEMP=/tmp

    local portable_data="$SCRIPT_DIR/.dbv4"
    export DBV4_DATA_DIR="$portable_data"
    export HF_HOME="$portable_data/huggingface"
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HF_XET_CACHE="$HF_HOME/xet"
    export HF_TOKEN_PATH="$HF_HOME/token"
    export PIP_CACHE_DIR="$portable_data/pip-cache"
    export TENSORRT_ENGINE_CACHE_DIR="$portable_data/tensorrt-engine-cache"
    mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" \
        "$PIP_CACHE_DIR" "$TENSORRT_ENGINE_CACHE_DIR" "$portable_data/runtime" || {
        echo "[ERROR] リポジトリ内データディレクトリを作成できません: $portable_data"
        exit 1
    }
    echo "[INFO] ポータブルデータ保存先: $portable_data"
}

show_help() {
    echo "DBV4 Tagger Universal (日本語ヘルプ)"
    echo ""
    echo "使い方: ./run_tagger.sh [オプション] [パス]"
    echo ""
    echo "  引数なしで実行すると「環境構築モード」となり、セットアップのみを行います。"
    echo ""
    echo "主なオプション:"
    echo "    -p, --path <path>   処理対象ファイル/フォルダ"
    echo "    -g, --gpu           GPUを使用する（自動判別）"
    echo "    --webgpu            Vulkan経由のWebGPUを使用する"
    echo "    --provider NAME     cpu/cuda/tensorrt/intel/webgpu/migraphxを明示"
    echo "    --gpu-index N       CUDA/TensorRT/MIGraphXのデバイス番号"
    echo "    --webgpu-device-index N  WebGPUのデバイス番号"
    echo "    --target-vendor NAME     nvidia/intel/amdを検証"
    echo "    --openvino-device GPU.N  Intel GPUの指定"
    echo "    --tensorrt-lib-dir DIR   TensorRT 10のライブラリ場所"
    echo "    -I, --force-intel   Intel GPUを強制的に使用する"
    echo "    -N, --force-nvidia  NVIDIA GPUを強制的に使用する"
    echo "    -A, --force-amd     AMD GPUを強制的に使用する"
    echo "    -o, --organize      フォルダ整理のみ行う（タグ付けOFF）"
    echo "    -t, --tag           タグ付けも行う（--organize併用時）"
    echo "    -x, --pixiv         Pixiv整理モード（末端フォルダ単位で判定し、対象フォルダの全画像を一括移動）"
    echo "    -R, --no-report     レポート作成なし"
    echo "    -r, --recursive     再帰検索ON"
    echo "    -n, --no-recursive  再帰検索OFF"
    echo "    -b, --batch-size <n> 推論バッチサイズ（デフォルト: 4 / 非対応時は 1）"
    echo "    -w, --io-workers <n> 前処理の並列ワーカー数（デフォルト: 自動）"
    echo "    -m, --model-profile <name> モデルプロファイル (compact_manual/lightweight/medium_manual/balanced/high/ultra/wd14_v3/future_1b)"
    echo "    -e, --model-repo <repo> DBV4モデル/タグのHFリポジトリIDを明示指定"
    echo "    -M, --model-file <file> モデルファイル名またはパス"
    echo "    -T, --tags-file <file> タグCSVファイル名またはパス"
    echo "    -q, --thresh <0.0-1.0> DBV4のtag best_thresholdを一括上書き（省略時はタグ固有値）"
    echo "    -f, --force         既存タグがあっても強制的に再解析・上書きする"
    echo "    -s, --sensitive-split-mode <2|4|6>"
    echo "                        旧CLI互換（DBV4ではR-15/R-17の5段階固定・非推奨）"
    echo "    -c, --record-ratio  全レーティングのRAW・割合スコアタグを記録する"
    echo "    -C, --no-record-ratio RAW・割合スコアタグを記録しない"
    echo "    -S, --server        サーバーモード"
    echo "    -K, --client        クライアントモード"
    echo "    --client-upload-mode MODE  preprocessed=縮小・可逆圧縮 / original=元画像送信"
    echo "                        前処理済み転送が使えない場合は元画像送信で続行"
    echo "    -L, --login         Hugging Faceログインモード（認証後に終了）"
    echo "    -H, --host <ip>     サーバーのIPアドレス"
    echo "    -P, --port <port>   ポート番号"
    echo "    -d, --debug         GPU/OpenVINOの詳細デバッグログを有効化"
    echo "    -h, --help          ヘルプ表示"
    echo ""
    echo "Client転送モードの例:"
    echo "    ./run_tagger.sh -K -H google-colab -p /path/to/images --client-upload-mode preprocessed"
    echo "    ./run_tagger.sh -K -H google-colab -p /path/to/images --client-upload-mode original"
    echo ""
}

# --- GPU検出関数 ---
detect_gpu_vendor() {
    # 1. Tegra SoC (Nintendo Switch / Jetson 等) の検出
    if [ -e "/dev/nvhost-gpu" ] || [ -e "/dev/nvhost-ctrl-gpu" ] || [ -e "/etc/nv_tegra_release" ] || [ -d "/usr/lib/aarch64-linux-gnu/tegra" ]; then
        echo "nvidia"
        return
    fi

    # 2. lspciの結果からVGA/3Dコントローラを探す
    local lspci_out=""
    if command -v lspci >/dev/null 2>&1; then
        lspci_out=$(lspci | grep -E "VGA|3D|Display" | tr '[:upper:]' '[:lower:]')
    fi
    
    if [[ "$lspci_out" == *"nvidia"* ]]; then
        echo "nvidia"
    elif [[ "$lspci_out" == *"intel"* ]]; then
        echo "intel"
    elif [[ "$lspci_out" == *"amd"* ]] || [[ "$lspci_out" == *"advanced micro devices"* ]] || [[ "$lspci_out" == *"radeon"* ]]; then
        if command -v lspci >/dev/null 2>&1 && lspci -nn | grep -Eiq '1002:15e7'; then
            echo "amd_webgpu"
            return
        fi
        # 内蔵GPUアーキテクチャの互換性チェック
        if command -v rocminfo >/dev/null 2>&1; then
            local arch=$(rocminfo | grep -o "gfx[0-9a-f]\+" | head -n 1)
            if [ "$arch" = "gfx90c" ]; then
                echo "amd_unsupported"
                return
            fi
        fi
        echo "amd"
    else
        echo "none"
    fi
}

# --- CUDAバージョン検出関数 ---
detect_cuda_major() {
    local cuda_major=12
    if command -v nvcc >/dev/null 2>&1; then
        local nvcc_out=$(nvcc --version 2>/dev/null)
        if [[ "$nvcc_out" =~ release\ ([0-9]+)\. ]]; then
            cuda_major="${BASH_REMATCH[1]}"
        fi
    elif [ -e "/usr/local/cuda" ]; then
        local real_path=$(readlink -f /usr/local/cuda 2>/dev/null)
        if [[ "$real_path" =~ cuda-([0-9]+)\. ]]; then
            cuda_major="${BASH_REMATCH[1]}"
        fi
    elif [ -e "/usr/lib/aarch64-linux-gnu/tegra/libcuda.so" ] || [ -e "/usr/lib/aarch64-linux-gnu/tegra/libcuda.so.1" ]; then
        if [ -e "/etc/nv_tegra_release" ]; then
            local tegra_rel=$(cat /etc/nv_tegra_release 2>/dev/null)
            if [[ "$tegra_rel" =~ R32 ]]; then
                cuda_major="10"
            elif [[ "$tegra_rel" =~ R35 ]]; then
                cuda_major="11"
            elif [[ "$tegra_rel" =~ R36 ]]; then
                cuda_major="12"
            fi
        else
            cuda_major="10"
        fi
    fi
    echo "$cuda_major"
}

# --- サポートされているPythonバージョンの検出 ---
find_supported_python() {
    for py_candidate in python3 python3.13 python3.12 python3.11 python3.10; do
        if command -v "$py_candidate" >/dev/null 2>&1; then
            local ver_minor=$("$py_candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            local ver_major_num=$(echo "$ver_minor" | cut -d. -f1)
            local ver_minor_num=$(echo "$ver_minor" | cut -d. -f2)
            if [ "$ver_major_num" -eq 3 ] && [ "$ver_minor_num" -le 13 ]; then
                echo "$py_candidate"
                return 0
            fi
        fi
    done
    echo ""
    return 1
}

# --- 環境セットアップ ---
setup_env() {
    local backend=$1 # nvidia, intel, amd, cpu, client
    local is_client=$2
    local venv_name=""
    
    if [ "$is_client" = "1" ]; then
        if [ -d "$SCRIPT_DIR/venv_webgpu" ]; then
            venv_name="venv_webgpu"
        elif [ -d "$SCRIPT_DIR/venv_gpu" ]; then
            venv_name="venv_gpu"
        elif [ -d "$SCRIPT_DIR/venv_intel" ]; then
            venv_name="venv_intel"
        elif [ -d "$SCRIPT_DIR/venv_amd" ]; then
            venv_name="venv_amd"
        elif [ -d "$SCRIPT_DIR/venv_std" ]; then
            venv_name="venv_std"
        else
            venv_name="venv_client"
        fi
        backend="client"
    elif [ "$backend" = "cpu" ]; then
        venv_name="venv_std"
    else
        if [ "$backend" = "nvidia" ]; then
            venv_name="venv_gpu"
        elif [ "$backend" = "intel" ]; then
            venv_name="venv_intel"
        elif [ "$backend" = "webgpu" ]; then
            venv_name="venv_webgpu"
        elif [ "$backend" = "amd" ]; then
            venv_name="venv_amd"
        fi
    fi

    VENV_DIR="$SCRIPT_DIR/$venv_name"
    
    if [ -d "$VENV_DIR" ]; then
        local current_venv_py_ver=$("$VENV_DIR/bin/python" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "99")
        if [ "$current_venv_py_ver" -ge 14 ]; then
            echo "[WARN] 既存の仮想環境 ($venv_name) は PyPI非対応の Python 3.$current_venv_py_ver で作成されています。"
            echo "[INFO] 互換性のある Python バージョンで仮想環境を再作成します..."
            rm -rf "$VENV_DIR"
        fi
    fi

    if [ ! -d "$VENV_DIR" ]; then
        PY_BIN=$(find_supported_python)
        if [ -z "$PY_BIN" ]; then
            echo "[ERROR] onnxruntime 等のライブラリは Python 3.14 以降のビルド(wheel)がPyPI上に存在しません。"
            echo "        Python 3.13 以下のバージョン (例: python3.13) をインストールしてください。"
            exit 1
        fi
        echo "[INFO] 仮想環境を作成中 ($venv_name, python: $PY_BIN)..."
        "$PY_BIN" -m venv "$VENV_DIR"
    fi
    
    PIP_CMD="$VENV_DIR/bin/pip"

    if [ "$is_client" = "1" ] && "$VENV_DIR/bin/python" -c "import huggingface_hub, numpy, PIL" >/dev/null 2>&1; then
        echo "[INFO] Clientは既存の仮想環境を再利用します: $venv_name"
        return 0
    fi

    $PIP_CMD install --upgrade pip >/dev/null 2>&1
    
    # 必要なパッケージのインストール
    echo "[INFO] 依存ライブラリを確認・インストール中 ($backend)..."
    REQ_FILE="$SCRIPT_DIR/requirements.txt"
    
    # 失敗したときにすぐ止まるようにエラー処理を追加じゃ
    if [ "$backend" = "client" ]; then
        $PIP_CMD install -r "$REQ_FILE" || { echo "[ERROR] ライブラリのインストールに失敗しました。"; exit 1; }
    elif [ "$backend" = "nvidia" ]; then
        local arch_name=$(uname -m)
        if [ "$arch_name" = "aarch64" ] || [ "$arch_name" = "arm64" ]; then
            echo "[INFO] ARM64 (Tegra / Jetson / Switch) 環境を検出しました。"
            $PIP_CMD install -r "$REQ_FILE" onnxruntime-gpu 2>/dev/null || \
            $PIP_CMD install -r "$REQ_FILE" onnxruntime || { echo "[ERROR] ライブラリのインストールに失敗しました。"; exit 1; }
        else
            # Match the CUDA 12 / ORT combination validated by the benchmark.
            local nvidia_pkgs=('onnxruntime-gpu[cuda,cudnn]<1.27')
            if [ "$PROVIDER" = "tensorrt" ]; then
                nvidia_pkgs+=('tensorrt-cu12<11')
            fi
            $PIP_CMD install -r "$REQ_FILE" "${nvidia_pkgs[@]}" || { echo "[ERROR] NVIDIAライブラリのインストールに失敗しました。"; exit 1; }
        fi
    elif [ "$backend" = "intel" ]; then
        $PIP_CMD install -r "$REQ_FILE" 'onnxruntime-openvino==1.24.1' 'openvino==2025.4.1' || { echo "[ERROR] ライブラリのインストールに失敗しました。"; exit 1; }
    elif [ "$backend" = "webgpu" ]; then
        $PIP_CMD install -r "$REQ_FILE" onnxruntime onnxruntime-ep-webgpu || { echo "[ERROR] WebGPUライブラリのインストールに失敗しました。"; exit 1; }
    elif [ "$backend" = "amd" ]; then
        # AMD用 ROCm対応パッケージ（onnxruntime-rocm または onnxruntime-migraphx）をインストールするのじゃ
        if [ "$PROVIDER" = "migraphx" ]; then
            $PIP_CMD install -r "$REQ_FILE" onnxruntime-migraphx -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/ -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/ || exit 1
        else
            $PIP_CMD install -r "$REQ_FILE" onnxruntime-rocm -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/ -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/ 2>/dev/null || \
            $PIP_CMD install -r "$REQ_FILE" onnxruntime-migraphx -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/ -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/ || { echo "[ERROR] ライブラリのインストールに失敗しました。"; exit 1; }
        fi
        # Ubuntu等の新しいLinux環境では、実行可能スタックのフラグが原因でロードエラーになるため解除するのじゃ
        SO_FILE=$(find "$VENV_DIR" -name "onnxruntime_pybind11_state.so" | head -n 1)
        if [ -n "$SO_FILE" ]; then
            if command -v execstack >/dev/null 2>&1; then
                execstack -c "$SO_FILE"
            elif command -v patchelf >/dev/null 2>&1; then
                patchelf --clear-execstack "$SO_FILE"
            else
                echo "[ERROR] セキュリティ制約の回避に必要な execstack または patchelf が見つからぬ！"
                echo "        Ubuntu 24.04等では古いexecstackは削除されておるため、以下のコマンドで patchelf をインストールしてから再度実行するのじゃ。"
                echo "        sudo apt install patchelf"
                exit 1
            fi
        fi
    else
        # CPU用
        $PIP_CMD install -r "$REQ_FILE" onnxruntime || { echo "[ERROR] ライブラリのインストールに失敗しました。"; exit 1; }
    fi
}

# 引数なしチェック
if [ $# -eq 0 ]; then
    configure_storage_paths
    echo "=========================================="
    echo "   DBV4 Tagger Universal - Setup Mode"
    echo "=========================================="
    echo "引数が指定されなかったため、環境構築のみを行います。"
    # CPUのみ作っておく（クライアントフラグ0）
    setup_env "cpu" "0"
    "$VENV_DIR/bin/python" "$PYTHON_SCRIPT" --gen-config
    echo "[INFO] セットアップ完了。GPU環境は --gpu 指定時に構築されます。"
    exit 0
fi

# 引数解析
while [[ $# -gt 0 ]]; do
    case "$1" in
        --provider|--gpu-index|--directml-device-index|--webgpu-device-index|--target-vendor|--openvino-device|--tensorrt-lib-dir)
            if [ "$#" -lt 2 ] || [[ "$2" == --* ]]; then
                echo "[ERROR] $1には値が必要です。"; exit 1
            fi ;;
    esac
    case $1 in
        -S|--server) PY_ARGS+=("--mode" "server"); shift ;;
        -K|--client) PY_ARGS+=("--mode" "client"); IS_CLIENT=1; shift ;;
        --client-upload-mode) PY_ARGS+=("--client-upload-mode" "$2"); shift 2 ;;
        -L|--login) LOGIN_MODE=1; shift ;;
        -o|--organize) DO_ORGANIZE=1; shift ;;
        -t|--tag) DO_TAG=1; shift ;; 
        -x|--pixiv) PY_ARGS+=("--pixiv"); DO_PIXIV=1; shift ;;
        -R|--no-report) PY_ARGS+=("--no-report"); shift ;;
        -r|--recursive) PY_ARGS+=("--recursive"); shift ;;
        -n|--no-recursive) PY_ARGS+=("--no-recursive"); shift ;;
        -b|--batch-size) PY_ARGS+=("--batch-size" "$2"); shift 2 ;;
        -w|--io-workers) PY_ARGS+=("--io-workers" "$2"); shift 2 ;;
        -m|--model-profile) PY_ARGS+=("--model-profile" "$2"); shift 2 ;;
        -e|--model-repo) PY_ARGS+=("--model-repo" "$2"); shift 2 ;;
        -M|--model-file) PY_ARGS+=("--model-file" "$2"); shift 2 ;;
        -T|--tags-file) PY_ARGS+=("--tags-file" "$2"); shift 2 ;;
        -q|--thresh) PY_ARGS+=("--thresh" "$2"); shift 2 ;;
        -g|--gpu) USE_GPU=1; shift ;;
        --provider)
            PROVIDER="$2"; PY_ARGS+=("--provider" "$2"); USE_GPU=1
            case "$2" in
                cpu) USE_GPU=0 ;;
                cuda|tensorrt) FORCE_TYPE="nvidia" ;;
                intel) FORCE_TYPE="intel" ;;
                webgpu) WEBGPU_MODE=1 ;;
                migraphx) FORCE_TYPE="amd" ;;
                *) echo "[ERROR] Linux provider: cpu/cuda/tensorrt/intel/webgpu/migraphx"; exit 1 ;;
            esac
            shift 2 ;;
        --tensorrt-lib-dir) TENSORRT_LIB_DIR_ARG="$2"; PY_ARGS+=("$1" "$2"); shift 2 ;;
        --gpu-index|--directml-device-index|--webgpu-device-index|--target-vendor|--openvino-device)
            PY_ARGS+=("$1" "$2"); shift 2 ;;
        --webgpu) USE_GPU=1; WEBGPU_MODE=1; shift ;;
        -I|--force-intel) USE_GPU=1; FORCE_TYPE="intel"; shift ;;
        -N|--force-nvidia) USE_GPU=1; FORCE_TYPE="nvidia"; shift ;;
        -A|--force-amd) USE_GPU=1; FORCE_TYPE="amd"; shift ;;
        -f|--force) PY_ARGS+=("--force"); shift ;;
        -s|--sensitive-split-mode) PY_ARGS+=("--sensitive-split-mode" "$2"); shift 2 ;;
        -c|--record-ratio) PY_ARGS+=("--record-ratio"); shift ;;
        -C|--no-record-ratio) PY_ARGS+=("--no-record-ratio"); shift ;;
        -p|--path) PY_ARGS+=("$2"); shift 2 ;;
        -H|--host) PY_ARGS+=("--host" "$2"); shift 2 ;;
        -P|--port) PY_ARGS+=("--port" "$2"); shift 2 ;;
        -d|--debug) DEBUG_MODE=1; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) PY_ARGS+=("$1"); shift ;;
    esac
done

configure_storage_paths

if [ "$LOGIN_MODE" -eq 1 ]; then
    echo "[INFO] Hugging Faceログインモードを開始します。"
    VENV_DIR=""
    for venv_name in venv_webgpu venv_gpu venv_intel venv_amd venv_std venv_client; do
        if [ -x "$SCRIPT_DIR/$venv_name/bin/hf" ]; then
            VENV_DIR="$SCRIPT_DIR/$venv_name"
            echo "[INFO] 既存の仮想環境を使用します: $venv_name"
            break
        fi
    done
    if [ -z "$VENV_DIR" ]; then
        setup_env "cpu" "0"
    fi
    "$VENV_DIR/bin/hf" auth login
    echo "[INFO] Hugging Faceログインモードを終了します。"
    exit 0
fi

# デバッグログ制御
# OpenVINOの内部診断（Inference successful / Model is fully supported on OpenVINO 等）は
# 通常実行では抑制し、--debug 指定時のみ有効にする。
if [ "$DEBUG_MODE" -eq 1 ]; then
    echo "[INFO] GPU/OpenVINOデバッグモードを有効化します。"
    export ORT_OPENVINO_ENABLE_CI_LOG=1
    export ORT_OPENVINO_ENABLE_DEBUG=1
    export OPENVINO_LOG_LEVEL=5
else
    unset ORT_OPENVINO_ENABLE_CI_LOG
    unset ORT_OPENVINO_ENABLE_DEBUG
    unset OPENVINO_LOG_LEVEL
fi

# アクションロジック構築
if [ $DO_ORGANIZE -eq 1 ] || [ $DO_PIXIV -eq 1 ]; then
    if [[ ! " ${PY_ARGS[*]} " =~ " --organize " ]]; then
        PY_ARGS+=("--organize")
    fi
    if [ $DO_TAG -eq 0 ] && [ $DO_PIXIV -eq 0 ]; then
        PY_ARGS+=("--no-tag")
    fi
fi

# Reject incompatible selections before installing an environment.
if [ "$WEBGPU_MODE" = "1" ] && [ -n "$PROVIDER" ] && [ "$PROVIDER" != "webgpu" ]; then
    echo "[ERROR] --webgpuと--providerが矛盾しています。"; exit 1
fi
if [ "$PROVIDER" = "cpu" ]; then
    USE_GPU=0
fi
# GPUモード決定ロジック
BACKEND_MODE="cpu"
if [ $USE_GPU -eq 1 ]; then
    if [ "$WEBGPU_MODE" -eq 1 ]; then
        echo "[INFO] Vulkan経由のWebGPUモードで実行します。"
        BACKEND_MODE="webgpu"
        PY_ARGS+=("--webgpu")
    elif [ "$FORCE_TYPE" = "intel" ]; then
        echo "[INFO] Intel GPU モードを強制使用します。"
        BACKEND_MODE="intel"
    elif [ "$FORCE_TYPE" = "nvidia" ]; then
        echo "[INFO] NVIDIA GPU モードを強制使用します。"
        BACKEND_MODE="nvidia"
    elif [ "$FORCE_TYPE" = "amd" ]; then
        echo "[WARN] AMD GPU モードを強制使用しますが、アーキテクチャ未サポートによるコアダンプの危険性があります。"
        BACKEND_MODE="amd"
    else
        # 自動判別
        DETECTED=$(detect_gpu_vendor)
        if [ "$DETECTED" = "nvidia" ]; then
            echo "[INFO] NVIDIA GPU を検出しました。TensorRTモードを優先します。"
            BACKEND_MODE="nvidia"
            PROVIDER="tensorrt"
            PY_ARGS+=("--provider" "tensorrt")
        elif [ "$DETECTED" = "intel" ]; then
            echo "[INFO] Intel GPU を検出しました。OpenVINOモードで実行します。"
            BACKEND_MODE="intel"
        elif [ "$DETECTED" = "amd_webgpu" ]; then
            echo "[INFO] AMD Barceloを検出しました。WebGPUモードで実行します。"
            BACKEND_MODE="webgpu"
            PY_ARGS+=("--webgpu")
        elif [ "$DETECTED" = "amd" ]; then
            echo "[INFO] AMD GPU を検出しました。ROCmモードで実行します。"
            BACKEND_MODE="amd"
        elif [ "$DETECTED" = "amd_unsupported" ]; then
            echo "[WARN] サポート外のAMD内蔵GPU(gfx90c等)を検出しました。コアダンプ回避のため、安全なCPUモードで実行します。"
            BACKEND_MODE="cpu"
        else
            echo "[WARN] GPUが見つからない、または判別できませんでした。CPUモードで実行します。"
            BACKEND_MODE="cpu"
        fi
    fi
    if [ -z "$PROVIDER" ] && [ "$BACKEND_MODE" = "intel" ]; then
        PY_ARGS+=("--provider" "intel")
    fi
    if [ -z "$PROVIDER" ] && [ "${DETECTED:-}" = "amd_webgpu" ]; then
        PY_ARGS+=("--target-vendor" "amd")
    fi
    # Explicit providers are checked in Python; GPU failures must not become CPU success.
    if [ "$BACKEND_MODE" != "cpu" ]; then
        PY_ARGS+=("--gpu")
    fi
fi

# CPUモード時はスレッドプールのデッドロックを防ぐため並列前処理を強制OFFにするのじゃ
if [ "$BACKEND_MODE" = "cpu" ]; then
    PY_ARGS+=("--io-workers" "0")
fi

setup_env "$BACKEND_MODE" "$IS_CLIENT"

if [ -n "$TENSORRT_LIB_DIR_ARG" ]; then
    export TENSORRT_LIB_DIR="$TENSORRT_LIB_DIR_ARG"
fi
if [ "$BACKEND_MODE" = "intel" ] && [ "$IS_CLIENT" != "1" ]; then
    OPENVINO_LIBS=$("$VENV_DIR/bin/python" -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("openvino").origin).parent / "libs")') || exit 1
    export LD_LIBRARY_PATH="$OPENVINO_LIBS:${LD_LIBRARY_PATH:-}"
fi
if [ "$BACKEND_MODE" = "nvidia" ]; then
    if [ -n "${TENSORRT_LIB_DIR:-}" ]; then
        export LD_LIBRARY_PATH="$TENSORRT_LIB_DIR:${LD_LIBRARY_PATH:-}"
    fi
    EXTRA_LD_PATHS=$("$VENV_DIR/bin/python" -c 'import site, os; paths = ["/usr/local/cuda/lib64", "/usr/local/nvidia/lib64", "/usr/lib/aarch64-linux-gnu/tegra"]; [paths.append(root) for p in site.getsitepackages() if os.path.exists(p) for root, dirs, files in os.walk(p) if ("nvidia" in root or "tensorrt" in root) and any(f.endswith(".so") or ".so." in f for f in files)]; print(":".join(list(dict.fromkeys([p for p in paths if os.path.exists(p)]))))' 2>/dev/null)
    if [ -n "$EXTRA_LD_PATHS" ]; then
        export LD_LIBRARY_PATH="$EXTRA_LD_PATHS:${LD_LIBRARY_PATH:-}"
    fi
elif [ "$BACKEND_MODE" = "amd" ]; then
    EXTRA_LD_PATHS=$(VENV_DIR="$VENV_DIR" "$VENV_DIR/bin/python" -c '
import os, glob, site
vdir = os.environ.get("VENV_DIR", "")
cdir = os.path.join(vdir, "lib", "rocm_compat") if vdir else ""
if cdir:
    os.makedirs(cdir, exist_ok=True)
    sdirs = ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/local/lib", "/opt/rocm/lib", "/opt/rocm/lib64"]
    fallback_src = None
    for sdir in sdirs:
        m = glob.glob(os.path.join(sdir, "libamdhip64.so*")) or glob.glob(os.path.join(sdir, "libhsa-runtime64.so*"))
        if m:
            m.sort(key=len)
            fallback_src = m[0]
            break
    known_targets = [
        "librocm_smi64.so", "libroctracer64.so", "libroctx64.so", "libamdhip64.so",
        "librocblas.so", "libhsa-runtime64.so", "libMIOpen.so", "libmigraphx_c.so",
        "librocsolver.so", "librocsparse.so", "librocrand.so", "librccl.so", "libamd_comgr.so"
    ]
    prefixes = ("libroc", "libamd", "libhsa", "libMIOpen", "libmigraphx", "librccl", "libhip")
    bases = set(known_targets)
    for sdir in sdirs:
        if os.path.exists(sdir):
            try:
                for fname in os.listdir(sdir):
                    if fname.startswith(prefixes) and (".so" in fname):
                        base = fname.split(".so")[0] + ".so"
                        bases.add(base)
            except Exception:
                pass
    for lbase in bases:
        fsrc = None
        for sdir in sdirs:
            matches = glob.glob(os.path.join(sdir, lbase + "*"))
            if matches:
                matches.sort(key=len)
                fsrc = matches[0]
                break
        if not fsrc:
            fsrc = fallback_src
        if fsrc:
            lpath = os.path.join(cdir, lbase)
            if not os.path.exists(lpath):
                try:
                    os.symlink(fsrc, lpath)
                except Exception:
                    pass
            for ver in range(1, 8):
                vname = f"{lbase}.{ver}"
                lpath = os.path.join(cdir, vname)
                if not os.path.exists(lpath):
                    try:
                        os.symlink(fsrc, lpath)
                    except Exception:
                        pass
paths = [cdir] if cdir else []
paths.extend(["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/local/lib", "/opt/rocm/lib", "/opt/rocm/migraphx/lib", "/opt/rocm/lib64", "/opt/rocm/hsa/lib"])
for p in site.getsitepackages():
    if os.path.exists(p):
        for root, dirs, files in os.walk(p):
            if any(f.startswith(("libmigraphx", "libamd", "libhsa", "libroc", "libonnxruntime")) for f in files):
                paths.append(root)
print(":".join(list(dict.fromkeys([p for p in paths if p and os.path.exists(p)]))))
' 2>/dev/null)
    if [ -n "$EXTRA_LD_PATHS" ]; then
        export LD_LIBRARY_PATH="$EXTRA_LD_PATHS:${LD_LIBRARY_PATH:-}"
    fi
    if [ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
        if command -v rocminfo >/dev/null 2>&1; then
            ARCH=$(rocminfo 2>/dev/null | grep -o "gfx90[0-9a-f]\+" | head -n 1)
            if [ "$ARCH" = "gfx906" ]; then
                export HSA_OVERRIDE_GFX_VERSION=9.0.6
            else
                export HSA_OVERRIDE_GFX_VERSION=9.0.0
            fi
        else
            export HSA_OVERRIDE_GFX_VERSION=9.0.0
        fi
        echo "[INFO] AMD GPU用環境変数を自動設定しました: HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
    fi
    export HSA_ENABLE_SDMA=0
    echo "[INFO] AMD iGPU/APU コアダンプ回避設定を適用しました: HSA_ENABLE_SDMA=0"
fi

if [ "$BACKEND_MODE" = "amd" ] && [ "$IS_CLIENT" != "1" ]; then
    "$VENV_DIR/bin/python" -c 'import onnxruntime as ort, sys; p=ort.get_available_providers(); print("[INFO] AMD providers:", p); sys.exit(0 if any(x in p for x in ("MIGraphXExecutionProvider", "ROCMExecutionProvider")) else 1)' || {
        echo "[ERROR] AMD ONNX Runtimeをロードできません。GPU/ROCmに適合するwheelを確認してください。"; exit 1;
    }
fi
echo "[INFO] Pythonスクリプトを実行 ($BACKEND_MODE)..."
"$VENV_DIR/bin/python" "$PYTHON_SCRIPT" "${PY_ARGS[@]}"
