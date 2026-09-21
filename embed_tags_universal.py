import argparse
import ctypes
import csv
import glob
import io
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import warnings
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from huggingface_hub import (
    get_hf_file_metadata,
    get_token,
    hf_hub_download,
    hf_hub_url,
    login,
)
from huggingface_hub.errors import GatedRepoError

try:
    import pillow_avif  # noqa: F401
except ImportError:
    pillow_avif = None

try:
    import make_report
except ImportError:
    make_report = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda iterable, **kwargs: iterable

from dbv4 import (
    DBV4Metadata,
    DBV4Preprocessor,
    DBV4_RATING_NAMES,
    MODEL_PROFILES,
    adapt_input_layout,
    clone_profile,
    get_model_profile,
    infer_output_to_probabilities,
    select_output_name,
)

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")

SYSTEM_OS = platform.system()
IS_WINDOWS = SYSTEM_OS == "Windows"
IS_LINUX = SYSTEM_OS == "Linux"

if IS_WINDOWS:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")
REPORT_LOG_FILE = os.path.join(os.getcwd(), "report_log.json")
VALID_EXTS = (".webp", ".jpg", ".jpeg", ".png", ".bmp", ".avif")

RATING_TAGS = {
    "general",
    "sensitive",
    "questionable",
    "explicit",
    *(f"sensitive_{i}" for i in range(10)),
    *(f"questionable_{i}" for i in range(10)),
    "sensitive_mild",
    "sensitive_high",
    *(f"sensitive_lvl{i}" for i in range(1, 7)),
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "model_profile": "balanced",
    "model_profiles": MODEL_PROFILES,
    "server_hosts": ["localhost", "google-colab", "100.xxx.xxx.xxx"],
    "server_port": 5000,
    "client_timeout": 15,
    "openvino_gpu_device": "GPU.0",
    "general_threshold": 0.40,
    "rating_sublevel_thresholds_5way": [0.20, 0.40, 0.60, 0.80],
    "rating_severity_sensitive_upper_reference": 25.0,
    "rating_severity_questionable_upper_reference": 40.0,
    "record_rating_percentages": True,
    "record_raw_score": True,
    "raw_score_format": "{rating}_score:{raw_score:.4f}",
    "percentage_format": "{rating}:{percentage}%",
    "folder_names": {
        "general": "R-00",
        "sensitive_0": "R-15_0",
        "sensitive_1": "R-15_1",
        "sensitive_2": "R-15_2",
        "sensitive_3": "R-15_3",
        "sensitive_4": "R-15_4",
        "questionable_0": "R-17_0",
        "questionable_1": "R-17_1",
        "questionable_2": "R-17_2",
        "questionable_3": "R-17_3",
        "questionable_4": "R-17_4",
        "explicit": "R-18",
    },
}

APP_CONFIG: Dict[str, Any] = {}


class Colors:
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    GREY = "\033[90m"
    CYAN = "\033[36m"
    RESET = "\033[0m"


def safe_write(message: str, end: str = "\n") -> None:
    try:
        tqdm.write(message, end=end)
    except Exception:
        try:
            sys.stdout.write(message + end)
            sys.stdout.flush()
        except Exception:
            pass


def merge_defaults(target: Dict[str, Any], source: Dict[str, Any]) -> bool:
    changed = False
    for key, value in source.items():
        if key not in target:
            target[key] = json.loads(json.dumps(value))
            changed = True
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            changed = merge_defaults(target[key], value) or changed
    return changed


def migrate_legacy_config(config: Dict[str, Any]) -> bool:
    changed = False
    profiles = config.get("model_profiles", {})
    if not isinstance(profiles, dict):
        return changed
    migrations = {
        "lightweight": {
            "animetimm/mobilenetv4_conv_small.dbv4-full",
            "animetimm/caformer_m36.dbv4-full",
        },
        "balanced": {"animetimm/convformer_s36.dbv4-full"},
        "high": {"animetimm/swinv2_base_window8_256.dbv4-full"},
        "ultra": {"animetimm/convnextv2_huge.dbv4-full"},
    }
    for profile_name, legacy_repositories in migrations.items():
        profile = profiles.get(profile_name, {})
        if isinstance(profile, dict) and profile.get("repo_id") in legacy_repositories:
            replacement = MODEL_PROFILES[profile_name]
            profile["repo_id"] = replacement["repo_id"]
            if profile_name == "ultra":
                profile["metadata_repo_id"] = replacement["metadata_repo_id"]
                profile["model_external_files"] = clone_profile(
                    replacement["model_external_files"]
                )
                profile["vram_warning"] = replacement["vram_warning"]
            changed = True
    legacy_large = profiles.get("large", {})
    if (
        config.get("model_profile") == "large"
        and isinstance(legacy_large, dict)
        and legacy_large.get("repo_id") == "animetimm/eva02_large_patch14_448.dbv4-full"
    ):
        config["model_profile"] = "high"
        changed = True
    ultra_profile = profiles.get("ultra", {})
    current_ultra = MODEL_PROFILES["ultra"]
    if (
        isinstance(ultra_profile, dict)
        and ultra_profile.get("repo_id") == current_ultra["repo_id"]
        and ultra_profile.get("vram_warning") != current_ultra["vram_warning"]
    ):
        ultra_profile["vram_warning"] = current_ultra["vram_warning"]
        changed = True
    return changed


def load_config() -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if not os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print(f"[INFO] DBV4用コンフィグを生成しました: {CONFIG_FILE}")
        except OSError as exc:
            print(f"[WARN] config.jsonを生成できません: {exc}")
        return config
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        if not isinstance(user_config, dict):
            raise ValueError("config.json のルートがオブジェクトではありません")
        changed = merge_defaults(user_config, DEFAULT_CONFIG)
        changed = migrate_legacy_config(user_config) or changed
        if changed:
            try:
                with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(user_config, f, indent=4, ensure_ascii=False)
                print(f"[INFO] DBV4設定を更新しました: {CONFIG_FILE}")
            except OSError as exc:
                print(f"[WARN] config.jsonの更新を保存できません: {exc}")
        return user_config
    except Exception as exc:
        print(f"[WARN] config.jsonを読み込めないためデフォルト設定を使用します: {exc}")
        return config


APP_CONFIG = load_config()


def get_bar(probability: float, color: str, width: int = 5) -> str:
    filled = max(0, min(width, int(probability * width)))
    return f"{color}{'█' * filled}{Colors.GREY}{'░' * (width - filled)}{Colors.RESET}"


def resolve_exiftool_cmd(command: str) -> str:
    resolved = shutil.which(command) if not os.path.isabs(command) else command
    return resolved or command


class ExifToolWrapper:
    def __init__(self, command: str = "exiftool") -> None:
        self.raw_command = command
        self.command = resolve_exiftool_cmd(command)
        self.process = None
        self.running = False

    def start(self) -> None:
        if self.running:
            return
        try:
            startupinfo = subprocess.STARTUPINFO() if IS_WINDOWS else None
            if IS_WINDOWS:
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            self.process = subprocess.Popen(
                [
                    self.command,
                    "-stay_open",
                    "True",
                    "-@",
                    "-",
                    "-common_args",
                    "-charset",
                    "filename=utf8",
                    "-lang",
                    "en",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                startupinfo=startupinfo,
            )
            self.running = True
        except Exception as exc:
            print(f"[ERROR] ExifToolの起動に失敗しました: {exc}")

    def stop(self) -> None:
        if not self.running:
            return
        try:
            if self.process and self.process.stdin:
                self.process.stdin.write(b"-stay_open\nFalse\n")
                self.process.stdin.flush()
                self.process.stdin.close()
            if self.process:
                try:
                    self.process.wait(timeout=2)
                except Exception:
                    self.process.kill()
        except Exception:
            pass
        finally:
            self.process = None
            self.running = False

    def execute(self, arguments: Sequence[str]) -> str:
        if not self.running:
            self.start()
        if not self.running or not self.process or not self.process.stdin or not self.process.stdout:
            return ""
        try:
            for argument in arguments:
                self.process.stdin.write(str(argument).encode("utf-8") + b"\n")
            self.process.stdin.write(b"-execute\n")
            self.process.stdin.flush()
            output = []
            while True:
                line = self.process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="ignore").strip()
                if text == "{ready}":
                    break
                output.append(text)
            return "\n".join(output)
        except Exception as exc:
            safe_write(f"[ERROR] ExifTool通信エラー: {exc}")
            self.stop()
            return ""

    def get_tags(self, path: str) -> List[str]:
        output = self.execute(["-XMP:Subject", "-s3", "-sep", ", ", "-fast", path])
        if not output or any(token in output for token in ("Error", "Warning", "File not found")):
            return []
        return [tag.strip() for tag in output.split(",") if tag.strip()]

    def write_tags(self, path: str, tags: Sequence[str]) -> bool:
        if not tags:
            return False
        output = self.execute(
            [
                "-overwrite_original",
                "-P",
                "-m",
                "-sep",
                ", ",
                f"-XMP:Subject={', '.join(tags)}",
                path,
            ]
        )
        if "image files updated" in output:
            return True
        safe_write(f"[WARN] タグ書き込み失敗 ({os.path.basename(path)}): {output}")
        return False

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


et_wrapper = ExifToolWrapper()


def build_providers(use_gpu: bool) -> List[Any]:
    if not use_gpu:
        return ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    candidates: List[Any] = []
    if IS_WINDOWS:
        candidates = [
            "DmlExecutionProvider",
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
        ]
    elif IS_LINUX:
        candidates = [
            ("OpenVINOExecutionProvider", {"device_type": APP_CONFIG.get("openvino_gpu_device", "GPU.0")}),
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "MIGraphXExecutionProvider",
        ]
    providers: List[Any] = []
    for candidate in candidates:
        name = candidate[0] if isinstance(candidate, tuple) else candidate
        if name == "TensorrtExecutionProvider":
            library_name = "nvinfer_10.dll" if IS_WINDOWS else "libnvinfer.so.10"
            try:
                ctypes.CDLL(library_name)
            except OSError:
                continue
        if name in available:
            providers.append(candidate)
    providers.append("CPUExecutionProvider")
    return providers


class RuntimeModel:
    def __init__(
        self,
        metadata: DBV4Metadata,
        preprocessor: DBV4Preprocessor,
        session: ort.InferenceSession,
    ) -> None:
        self.metadata = metadata
        self.preprocessor = preprocessor
        self.session = session
        input_meta = session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = input_meta.shape
        self.output_name = select_output_name(session.get_outputs(), metadata.label_count)

    @property
    def batch_limit(self) -> Optional[int]:
        if not self.input_shape:
            return None
        value = self.input_shape[0]
        return int(value) if isinstance(value, (int, np.integer)) else None

    def preprocess_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        batch = np.stack([self.preprocessor(image) for image in images], axis=0).astype(np.float32)
        return adapt_input_layout(batch, self.input_shape)

    def predict_images(self, images: Sequence[Image.Image]) -> List[np.ndarray]:
        if not images:
            return []
        raw = self.session.run(
            [self.output_name],
            {self.input_name: self.preprocess_batch(images)},
        )[0]
        raw = np.asarray(raw)
        if raw.ndim == 1:
            raw = raw[None, :]
        if raw.ndim != 2:
            raise ValueError(f"DBV4 outputは2次元を想定しています: shape={raw.shape}")
        expected_shape = (len(images), self.metadata.label_count)
        if raw.shape != expected_shape:
            raise ValueError(
                f"DBV4 output shape mismatch: model={raw.shape}, expected={expected_shape}, "
                f"output={self.output_name}"
            )
        return [infer_output_to_probabilities(row) for row in raw]


def load_runtime_model(
    use_gpu: bool,
    profile_name: str,
    model_repo: Optional[str] = None,
    model_file: Optional[str] = None,
    tags_file: Optional[str] = None,
) -> RuntimeModel:
    profiles = APP_CONFIG.get("model_profiles", MODEL_PROFILES)
    profile = get_model_profile(profile_name, profiles)
    if not profile.get("available", True):
        raise RuntimeError(str(profile.get("unavailable_reason", "このprofileは現在利用できません。")))
    if profile.get("access_notice"):
        print(f"[WARN] {profile_name}: {profile['access_notice']}")
    if profile.get("runtime_warning"):
        print(f"[WARN] {profile['runtime_warning']}")
    ensure_profile_access(profile_name, profile)
    if use_gpu and profile.get("vram_warning"):
        print(f"[WARN] {profile['vram_warning']}")
    metadata = DBV4Metadata.load(
        profile,
        base_dir=SCRIPT_DIR,
        model_repo_override=model_repo,
        model_file_override=model_file,
        tags_file_override=tags_file,
        load_model=True,
    )
    preprocessor = DBV4Preprocessor.from_metadata(metadata)
    providers = build_providers(use_gpu)
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    provider_names = {
        item[0] if isinstance(item, tuple) else item
        for item in providers
    }
    if "DmlExecutionProvider" in provider_names:
        # DirectML requires sequential execution and memory patterns disabled.
        # DBV4 CAFormer also fails during ORT's graph rewrite with the default
        # ORT_ENABLE_ALL ("logits" output disappears), so let DirectML consume
        # the original graph with only ORT's mandatory transformations.
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.enable_mem_pattern = False
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        session = ort.InferenceSession(
            metadata.model_path,
            sess_options=session_options,
            providers=providers,
        )
    except Exception as exc:
        if use_gpu:
            raise RuntimeError(f"DBV4 GPUプロバイダの初期化に失敗しました: {exc}") from exc
        session = ort.InferenceSession(
            metadata.model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
    active = session.get_providers()
    family_label = "WD14 V3" if metadata.family == "wd14_v3" else "DBV4"
    print(f"[INFO] {family_label}モデル: {metadata.repo_id} ({metadata.profile_name})")
    print(f"[INFO] ラベル数: {metadata.label_count}")
    print(f"[INFO] metadata version: {metadata.metadata_version}")
    print(f"[INFO] 入力 shape: {session.get_inputs()[0].shape}")
    print(f"[INFO] アクティブプロバイダ: {active}")
    if use_gpu:
        active_names = {item[0] if isinstance(item, tuple) else item for item in active}
        candidate_names = {
            item[0] if isinstance(item, tuple) else item
            for item in providers
            if (item[0] if isinstance(item, tuple) else item) != "CPUExecutionProvider"
        }
        if not active_names.intersection(candidate_names):
            raise RuntimeError("DBV4モデルでGPU Execution Providerを有効化できませんでした。")
    return RuntimeModel(metadata, preprocessor, session)


def ensure_profile_access(profile_name: str, profile: Dict[str, Any]) -> None:
    if not profile.get("requires_manual_approval"):
        return
    repo_id = str(profile["repo_id"])
    model_filename = str(profile.get("model_file", "model.onnx"))
    model_page_url = f"https://huggingface.co/{repo_id}"

    if not get_token():
        print("[INFO] Hugging Faceへ未ログインのため、ブラウザ認証を開始します。")
        try:
            login(skip_if_logged_in=True)
        except Exception as exc:
            raise RuntimeError(f"Hugging Faceのブラウザ認証に失敗しました: {exc}") from exc

    try:
        get_hf_file_metadata(
            hf_hub_url(repo_id=repo_id, filename=model_filename),
            token=True,
        )
    except GatedRepoError as exc:
        opened = webbrowser.open(model_page_url, new=2)
        if opened:
            print(f"[INFO] アクセス申請・同意ページをブラウザで開きました: {model_page_url}")
        else:
            print(f"[WARN] ブラウザを開けませんでした。次のページを開いてください: {model_page_url}")
        raise SystemExit(
            f"[ERROR] {profile_name}は未承認または承認待ちです。"
            "ブラウザで申請・同意を完了し、管理者の承認後に再実行してください。"
        ) from exc


def warmup_runtime(runtime: RuntimeModel, batch_size: int) -> float:
    active = runtime.session.get_providers()
    compiling = {
        "TensorrtExecutionProvider",
        "OpenVINOExecutionProvider",
        "MIGraphXExecutionProvider",
        "DmlExecutionProvider",
    }
    if not any(
        (provider[0] if isinstance(provider, tuple) else provider) in compiling
        for provider in active
    ):
        return 0.0
    print(f"[INFO] コンパイル系EPのウォームアップ推論を実行します (batch={batch_size})...")
    started = time.time()
    image = Image.new("RGB", (512, 512), (0, 0, 0))
    runtime.predict_images([image] * max(1, batch_size))
    if batch_size > 1:
        runtime.predict_images([image])
    elapsed = time.time() - started
    print(f"[INFO] ウォームアップ完了 (所要時間: {elapsed:.2f}秒)。エンジンの準備が整いました。")
    return elapsed


def calculate_rating_severity(
    scores: Dict[str, float],
    base_rating: str,
    sensitive_reference: float,
    questionable_reference: float,
) -> float:
    values = [max(1e-6, min(1.0 - 1e-6, float(scores[name]))) for name in DBV4_RATING_NAMES]
    gen, sen, que, exp = values

    def pairwise_position(upper: float, lower: float) -> float:
        upper_logit = np.log(upper / (1.0 - upper))
        lower_logit = np.log(lower / (1.0 - lower))
        return float(1.0 / (1.0 + np.exp(-(upper_logit - lower_logit))))

    def score_position(score: float, reference_percent: float) -> float:
        return float(np.clip(np.log1p(score * 100.0) / np.log1p(reference_percent), 0.0, 1.0))

    if base_rating == "sensitive":
        lower = pairwise_position(sen, gen)
        upper = score_position(que, sensitive_reference)
        band = 1
    elif base_rating == "questionable":
        lower = pairwise_position(que, sen)
        upper = score_position(exp, questionable_reference)
        band = 2
    else:
        return 0.0 if base_rating == "general" else 1.0

    return float(np.clip((band + np.sqrt(lower * upper)) / 4.0, 0.0, 1.0))


def determine_rating_sublevel(base_rating: str, severity: float) -> str:
    if base_rating not in ("sensitive", "questionable"):
        return base_rating
    thresholds = APP_CONFIG.get("rating_sublevel_thresholds_5way", [0.20, 0.40, 0.60, 0.80])
    if len(thresholds) != 4:
        raise ValueError("rating_sublevel_thresholds_5way must contain exactly 4 values")
    band = 1 if base_rating == "sensitive" else 2
    local = float(np.clip(severity * 4.0 - band, 0.0, np.nextafter(1.0, 0.0)))
    for index, threshold in enumerate(thresholds):
        if local < float(threshold):
            return f"{base_rating}_{index}"
    return f"{base_rating}_4"


def calculate_rating(
    metadata: DBV4Metadata,
    probabilities: Sequence[float],
    rating_thresh: Optional[float],
    ignore_sensitive: bool,
    general_threshold: float,
    fname_disp: str = "",
) -> str:
    scores = metadata.get_rating_scores(probabilities)
    if scores["general"] >= general_threshold:
        base = "general"
    elif rating_thresh is not None:
        non_general = sum(scores[name] for name in DBV4_RATING_NAMES[1:])
        base = max(DBV4_RATING_NAMES[1:], key=scores.get) if non_general > rating_thresh else "general"
    else:
        base = max(DBV4_RATING_NAMES, key=scores.get)

    rating = base
    if base in ("sensitive", "questionable"):
        rating = determine_rating_sublevel(
            base,
            calculate_rating_severity(
                scores,
                base,
                float(APP_CONFIG.get("rating_severity_sensitive_upper_reference", 25.0)),
                float(APP_CONFIG.get("rating_severity_questionable_upper_reference", 40.0)),
            ),
        )
    if ignore_sensitive and rating.startswith("sensitive_"):
        rating = "general"

    if fname_disp:
        values = [scores[name] for name in DBV4_RATING_NAMES]
        bars = [
            get_bar(values[0], Colors.GREEN),
            get_bar(values[1], Colors.YELLOW),
            get_bar(values[2], Colors.MAGENTA),
            get_bar(values[3], Colors.RED),
        ]
        folder = folder_name_for_rating(rating)
        color = Colors.CYAN
        if rating == "general":
            color = Colors.GREEN
        elif rating.startswith("sensitive"):
            color = Colors.YELLOW
        elif rating.startswith("questionable"):
            color = Colors.MAGENTA
        elif rating == "explicit":
            color = Colors.RED
        safe_write(
            f"[{fname_disp}] Gen:{bars[0]}{values[0] * 100:04.1f}% "
            f"Sen:{bars[1]}{values[1] * 100:04.1f}% "
            f"Que:{bars[2]}{values[2] * 100:04.1f}% "
            f"Exp:{bars[3]}{values[3] * 100:04.1f}% "
            f"=> {color}[{folder}]{Colors.RESET}"
        )
    return rating


def format_score_tags(metadata: DBV4Metadata, probabilities: Sequence[float]) -> List[str]:
    scores = metadata.get_rating_scores(probabilities)
    result: List[str] = []
    raw_enabled = bool(APP_CONFIG.get("record_raw_score", True))
    pct_enabled = bool(APP_CONFIG.get("record_rating_percentages", True))
    raw_format = APP_CONFIG.get("raw_score_format", "{rating}_score:{raw_score:.4f}")
    pct_format = APP_CONFIG.get("percentage_format", "{rating}:{percentage}%")
    result.append(metadata.rating_tags_marker)
    for name in DBV4_RATING_NAMES:
        score = scores[name]
        if raw_enabled:
            try:
                result.append(raw_format.format(rating=name, cat=name, raw_score=score))
            except Exception:
                result.append(f"{name}_score:{score:.4f}")
        if pct_enabled:
            percentage = f"{score * 100:.1f}"
            try:
                result.append(pct_format.format(rating=name, cat=name, percentage=percentage))
            except Exception:
                result.append(f"{name}:{percentage}%")
    return result


def is_score_tag(tag: str) -> bool:
    return bool(
        re.match(r"^(general|sensitive|questionable|explicit)_score:[0-9.]+$", tag)
        or re.match(r"^(general|sensitive|questionable|explicit):[0-9.]+%$", tag)
    )


def extract_raw_rating_scores(
    metadata: DBV4Metadata,
    tags: Sequence[str],
) -> Optional[List[float]]:
    if metadata.rating_tags_marker not in tags:
        return None
    scores: Dict[str, float] = {}
    pattern = re.compile(r"^(general|sensitive|questionable|explicit)_score:([0-9.]+)$")
    for tag in tags:
        match = pattern.match(tag.strip())
        if match:
            try:
                scores[match.group(1)] = float(match.group(2))
            except ValueError:
                pass
    if any(name not in scores for name in DBV4_RATING_NAMES):
        return None
    return [scores[name] for name in DBV4_RATING_NAMES]


def preserve_existing_tags(tags: Sequence[str]) -> List[str]:
    return [
        tag.strip()
        for tag in tags
        if tag.strip()
        and tag.strip() not in RATING_TAGS
        and not is_score_tag(tag.strip())
        and not tag.strip().startswith("dbv4_model:")
    ]


def folder_name_for_rating(rating: str) -> str:
    mapping = APP_CONFIG.get("folder_names", {})
    folder_name = mapping.get(rating)
    if folder_name is None and isinstance(rating, str) and "_" in rating:
        folder_name = mapping.get(rating.split("_", 1)[0])
    return str(folder_name if folder_name is not None else rating)


def organize_file(
    file_path: str,
    rating: str,
    is_pixiv: bool = False,
    base_dirs: Optional[Sequence[str]] = None,
) -> Tuple[bool, str]:
    folder_name = folder_name_for_rating(rating)
    if is_pixiv and (rating == "general" or rating.startswith("sensitive_")):
        return False, file_path
    try:
        source = os.path.abspath(file_path)
        source_dir = os.path.dirname(source)
        filename = os.path.basename(source)
        target_dir = os.path.join(source_dir, folder_name)
        target_path = os.path.join(target_dir, filename)
        if is_pixiv and base_dirs:
            matches = [base for base in base_dirs if source.startswith(base)]
            if matches:
                base = max(matches, key=len)
                relative = os.path.relpath(source, base)
                target_path = os.path.join(
                    os.path.dirname(base),
                    folder_name,
                    relative,
                )
                target_dir = os.path.dirname(target_path)
        if os.path.abspath(source_dir) == os.path.abspath(target_dir):
            return False, source
        os.makedirs(target_dir, exist_ok=True)
        if os.path.exists(target_path):
            stem, ext = os.path.splitext(filename)
            target_path = os.path.join(target_dir, f"{stem}_{uuid.uuid4().hex[:6]}{ext}")
        shutil.move(source, target_path)
        return True, target_path
    except Exception as exc:
        safe_write(f"[WARN] 移動失敗 {file_path}: {exc}")
        return False, file_path


def collect_images(paths: Sequence[str], recursive: bool = True) -> List[str]:
    collected: List[str] = []
    for raw_path in paths:
        candidates = glob.glob(raw_path, recursive=recursive) if "*" in raw_path or "?" in raw_path else [raw_path]
        for candidate in candidates:
            if os.path.isdir(candidate):
                print(f"[INFO] ディレクトリをスキャン中 (再帰={recursive}): {candidate}")
                if recursive:
                    for root, _, files in os.walk(candidate):
                        collected.extend(
                            os.path.join(root, name)
                            for name in files
                            if name.lower().endswith(VALID_EXTS)
                        )
                else:
                    try:
                        collected.extend(
                            os.path.join(candidate, name)
                            for name in os.listdir(candidate)
                            if os.path.isfile(os.path.join(candidate, name))
                            and name.lower().endswith(VALID_EXTS)
                        )
                    except OSError:
                        pass
            elif os.path.isfile(candidate) and candidate.lower().endswith(VALID_EXTS):
                collected.append(candidate)
    return sorted(set(collected))


def collect_pixiv_image_groups(paths: Sequence[str]) -> Dict[str, List[str]]:
    groups: Dict[str, set] = {}
    excluded_dirs = {
        str(folder_name)
        for folder_name in APP_CONFIG.get("folder_names", {}).values()
        if folder_name
    }

    def add_group(directory: str, files: Sequence[str]) -> None:
        image_paths = [
            os.path.abspath(os.path.join(directory, name))
            for name in files
            if name.lower().endswith(VALID_EXTS)
        ]
        if image_paths:
            groups.setdefault(os.path.abspath(directory), set()).update(image_paths)

    for raw_path in paths:
        candidates = glob.glob(raw_path, recursive=True) if "*" in raw_path or "?" in raw_path else [raw_path]
        for candidate in candidates:
            if os.path.isdir(candidate):
                print(f"[INFO] Pixiv用の画像フォルダをスキャン中: {candidate}")
                for root, dirnames, files in os.walk(candidate):
                    current_name = os.path.basename(os.path.normpath(root))
                    if current_name in excluded_dirs:
                        dirnames[:] = []
                        continue
                    dirnames[:] = [
                        directory
                        for directory in dirnames
                        if directory not in excluded_dirs
                    ]
                    add_group(root, files)
            elif os.path.isfile(candidate) and candidate.lower().endswith(VALID_EXTS):
                add_group(os.path.dirname(os.path.abspath(candidate)), [os.path.basename(candidate)])

    return {directory: sorted(file_paths) for directory, file_paths in groups.items()}


def get_pixiv_move_rating(ratings: Sequence[str]) -> Optional[str]:
    best_rating = None
    best_priority = -1
    for rating in ratings:
        if rating == "explicit":
            priority = 100
        elif rating == "questionable":
            priority = 10
        else:
            match = re.fullmatch(r"questionable_(\d+)", str(rating))
            if not match:
                continue
            priority = 10 + int(match.group(1))
        if priority > best_priority:
            best_priority = priority
            best_rating = rating
    return best_rating


def organize_pixiv_folder(
    file_paths: Sequence[str],
    rating: str,
    base_dirs: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, str], int]:
    if not file_paths or not rating:
        return {}, 0
    source_dirs = {os.path.dirname(os.path.abspath(path)) for path in file_paths}
    if len(source_dirs) != 1:
        raise ValueError("Pixivフォルダ整理では1つの画像フォルダのみ指定してください。")
    source_dir = next(iter(source_dirs))
    moved_paths: Dict[str, str] = {}
    moved_count = 0
    for file_path in file_paths:
        moved, new_path = organize_file(file_path, rating, is_pixiv=True, base_dirs=base_dirs)
        if moved:
            moved_count += 1
            moved_paths[os.path.abspath(file_path)] = os.path.abspath(new_path)
    if moved_count == len(file_paths) and os.path.isdir(source_dir):
        try:
            if not os.listdir(source_dir):
                os.rmdir(source_dir)
                safe_write(f"[INFO] Pixiv画像フォルダを削除: {source_dir}")
        except OSError as exc:
            safe_write(f"[WARN] 空フォルダの削除に失敗しました ({source_dir}): {exc}")
    return moved_paths, moved_count


class TagServerHandler(BaseHTTPRequestHandler):
    runtime: Optional[RuntimeModel] = None

    def do_POST(self) -> None:
        try:
            if not self.runtime:
                raise RuntimeError("DBV4 runtimeが初期化されていません。")
            length = int(self.headers.get("Content-Length", "0"))
            image = Image.open(io.BytesIO(self.rfile.read(length))).convert("RGB")
            probabilities = self.runtime.predict_images([image])[0]
            payload = {
                **self.runtime.metadata.summary(),
                "probabilities": probabilities.astype(float).tolist(),
            }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def run_server(args: argparse.Namespace) -> None:
    runtime = load_runtime_model(
        args.gpu,
        args.model_profile,
        args.model_repo,
        args.model_file,
        args.tags_file,
    )
    TagServerHandler.runtime = runtime
    server = HTTPServer(("0.0.0.0", args.port), TagServerHandler)
    print(f"\n[INFO] DBV4推論サーバー稼働中 Port: {args.port}")
    print(f"[INFO] model_id={runtime.metadata.repo_id}")
    print(f"[INFO] output_size={runtime.metadata.label_count}")
    print(f"[INFO] metadata_version={runtime.metadata.metadata_version}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def load_client_metadata(args: argparse.Namespace) -> DBV4Metadata:
    profiles = APP_CONFIG.get("model_profiles", MODEL_PROFILES)
    profile = get_model_profile(args.model_profile, profiles)
    return DBV4Metadata.load(
        profile,
        base_dir=SCRIPT_DIR,
        model_repo_override=args.model_repo,
        model_file_override=args.model_file,
        tags_file_override=args.tags_file,
        load_model=False,
    )


def client_predict(
    server_url: str,
    image_path: str,
    metadata: DBV4Metadata,
    timeout: int,
) -> np.ndarray:
    with open(image_path, "rb") as f:
        data = f.read()
    request = urllib.request.Request(server_url, data=data, method="POST")
    request.add_header("Content-Type", "application/octet-stream")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, list):
        probabilities = np.asarray(payload, dtype=np.float32)
    elif isinstance(payload, dict):
        if payload.get("protocol") != 1:
            raise RuntimeError("サーバーのDBV4 protocol versionが不一致です。")
        if payload.get("model_id") != metadata.repo_id:
            raise RuntimeError("サーバーとクライアントのmodel_idが不一致です。")
        if int(payload.get("output_size", -1)) != metadata.label_count:
            raise RuntimeError("サーバーとクライアントのoutput sizeが不一致です。")
        if payload.get("metadata_version") != metadata.metadata_version:
            raise RuntimeError("サーバーとクライアントのmetadata versionが不一致です。")
        probabilities = np.asarray(payload.get("probabilities", []), dtype=np.float32)
    else:
        raise RuntimeError("サーバー応答形式が不正です。")
    if probabilities.shape[0] != metadata.label_count:
        raise RuntimeError("DBV4 output sizeがmetadataと一致しません。")
    return probabilities


def inference_timing_summary(
    inferred_count: int,
    inferred_time: float,
    batch_history: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    main_count = inferred_count
    main_time = inferred_time
    first_count = 0
    first_time = 0.0
    outlier_detected = False
    if len(batch_history) >= 2:
        first_count = int(batch_history[0]["count"])
        first_time = float(batch_history[0]["time"])
        rest_count = sum(int(batch["count"]) for batch in batch_history[1:])
        rest_time = sum(float(batch["time"]) for batch in batch_history[1:])
        first_per_image = first_time / first_count if first_count else 0.0
        rest_per_image = rest_time / rest_count if rest_count else 0.0
        if rest_count and first_time > 1.0 and first_per_image > 2.0 * rest_per_image:
            outlier_detected = True
            main_count = rest_count
            main_time = rest_time
    return {
        "count": inferred_count,
        "time": inferred_time,
        "main_count": main_count,
        "main_time": main_time,
        "fps": main_count / main_time if main_time else 0.0,
        "ms_per_image": main_time / main_count * 1000.0 if main_count else 0.0,
        "outlier_detected": outlier_detected,
        "first_count": first_count,
        "first_time": first_time,
    }


def process_images(args: argparse.Namespace) -> None:
    is_client = args.mode == "client"
    metadata = load_client_metadata(args) if is_client else None
    runtime: Optional[RuntimeModel] = None
    if not is_client:
        runtime = load_runtime_model(
            args.gpu,
            args.model_profile,
            args.model_repo,
            args.model_file,
            args.tags_file,
        )
        metadata = runtime.metadata

    assert metadata is not None
    need_exiftool = (not args.no_tag) or args.organize
    if need_exiftool:
        et_wrapper.start()

    if args.record_ratio is not None:
        APP_CONFIG["record_raw_score"] = args.record_ratio
        APP_CONFIG["record_rating_percentages"] = args.record_ratio

    is_pixiv = bool(args.pixiv)
    recursive = args.recursive if args.recursive is not None else not args.organize
    if is_pixiv:
        pixiv_groups = collect_pixiv_image_groups(args.images)
        target_files = sorted(
            file_path
            for group_files in pixiv_groups.values()
            for file_path in group_files
        )
        print(
            f"[INFO] Pixiv画像グループ: {len(pixiv_groups)}フォルダ / "
            f"{len(target_files)}枚"
        )
    else:
        pixiv_groups = {}
        target_files = collect_images(args.images, recursive)
    if not target_files:
        print("[WARN] 対象ファイルが見つかりません。")
        if need_exiftool:
            et_wrapper.stop()
        return

    base_dirs = []
    for raw_path in args.images:
        base = os.path.abspath(raw_path.split("*")[0].split("?")[0])
        if not os.path.isdir(base):
            base = os.path.dirname(base)
        base_dirs.append(base)

    requested_batch_size = max(1, args.batch_size)
    batch_size = 1 if is_client else requested_batch_size
    if is_client and requested_batch_size > 1:
        print("[WARN] クライアントモードではバッチ推論を使用できないため、batch-size=1で実行します。")
    if runtime and runtime.batch_limit is not None:
        batch_limit = max(1, runtime.batch_limit)
        if batch_size > batch_limit:
            if batch_limit == 1:
                print("[WARN] このモデルはバッチ推論に非対応のため、batch-size=1に変更します。")
            else:
                print(
                    f"[WARN] batch-sizeがモデル上限({batch_limit})を超えているため、"
                    f"{batch_limit}に変更します。"
                )
            batch_size = batch_limit
    io_workers = args.io_workers
    if io_workers == -1:
        io_workers = max(2, min(4, (os.cpu_count() or 1) // 2)) if batch_size > 1 else 0
    elif io_workers < 0:
        io_workers = 0

    if runtime and batch_size > 1:
        print(f"[INFO] DBV4 batch-size={batch_size}, io-workers={io_workers}")

    warmup_time = 0.0
    if runtime:
        try:
            warmup_time = warmup_runtime(runtime, batch_size)
        except Exception as exc:
            print(f"[WARN] ウォームアップ推論をスキップしました: {exc}")

    processed = organized = 0
    pixiv_rating_by_path: Dict[str, str] = {}
    pixiv_moved_paths: Dict[str, str] = {}
    pixiv_target_groups = 0
    pixiv_moved_groups = 0
    inferred = skipped = 0
    inferred_time = skipped_time = 0.0
    batch_history: List[Dict[str, Any]] = []
    executor = (
        ThreadPoolExecutor(max_workers=io_workers)
        if runtime and batch_size > 1 and io_workers > 0
        else None
    )
    pending: List[Dict[str, Any]] = []
    report_data: List[Dict[str, Any]] = []
    progress = tqdm(total=len(target_files), unit="img", dynamic_ncols=True)

    def update_progress_postfix() -> None:
        infer_speed = inferred / inferred_time if inferred_time else 0.0
        skip_speed = skipped / skipped_time if skipped_time else 0.0
        progress.set_postfix_str(
            f"推論:{inferred}枚({infer_speed:.1f}/s) "
            f"Skip:{skipped}枚({skip_speed:.1f}/s)"
        )

    def finalize(
        path: str,
        existing_tags: Sequence[str],
        detected_tags: Sequence[str],
        rating: str,
        probabilities: Optional[np.ndarray],
    ) -> None:
        nonlocal processed, organized
        final_path = path
        if detected_tags and (not args.no_tag or args.organize):
            if et_wrapper.write_tags(path, detected_tags):
                processed += 1
        if is_pixiv:
            if rating:
                pixiv_rating_by_path[os.path.abspath(path)] = rating
        elif args.organize:
            moved, new_path = organize_file(path, rating, False, base_dirs)
            if moved:
                organized += 1
                final_path = new_path
        if probabilities is not None and not args.no_report:
            report_data.append(
                {
                    "path": os.path.abspath(final_path),
                    "rating": rating,
                    "probs": [
                        metadata.get_rating_scores(probabilities)[name]
                        for name in DBV4_RATING_NAMES
                    ],
                    "model_id": metadata.repo_id,
                    "metadata_version": metadata.metadata_version,
                }
            )
        progress.update(1)

    def decode_and_finalize(item: Dict[str, Any], probabilities: np.ndarray) -> None:
        display = os.path.basename(item["path"])
        if len(display) > 20:
            display = display[:17] + "..."
        rating = calculate_rating(
            metadata,
            probabilities,
            args.rating_thresh,
            args.ignore_sensitive,
            float(APP_CONFIG.get("general_threshold", 0.40)),
            display,
        )
        tags = [rating]
        tags.extend(format_score_tags(metadata, probabilities))
        tags.extend(metadata.decode_tags(probabilities, args.thresh))
        for old_tag in preserve_existing_tags(item.get("existing_tags", [])):
            if old_tag not in tags:
                tags.append(old_tag)
        finalize(item["path"], item.get("existing_tags", []), tags, rating, probabilities)

    def load_image(path: str) -> Tuple[Optional[Image.Image], Optional[Exception]]:
        try:
            with Image.open(path) as image:
                return image.convert("RGB"), None
        except Exception as exc:
            return None, exc

    def run_batch(items: Sequence[Dict[str, Any]]) -> None:
        nonlocal inferred, inferred_time
        if not runtime or not items:
            return
        started = time.time()
        if executor:
            loaded = list(executor.map(lambda item: load_image(item["path"]), items))
        else:
            loaded = [load_image(item["path"]) for item in items]
        valid_items: List[Dict[str, Any]] = []
        images: List[Image.Image] = []
        for item, (image, error) in zip(items, loaded):
            if error or image is None:
                safe_write(f"エラー {os.path.basename(item['path'])}: {error}")
                progress.update(1)
                continue
            valid_items.append(item)
            images.append(image)
        if not valid_items:
            return
        try:
            predictions = runtime.predict_images(images)
        except Exception as exc:
            safe_write(f"[WARN] DBV4バッチ推論に失敗: {exc} -> 1枚ずつに切り替えます。")
            for item, image in zip(valid_items, images):
                single_started = time.time()
                try:
                    prediction = runtime.predict_images([image])[0]
                    elapsed = time.time() - single_started
                    inferred += 1
                    inferred_time += elapsed
                    batch_history.append({"count": 1, "time": elapsed})
                    update_progress_postfix()
                    decode_and_finalize(item, prediction)
                except Exception as single_exc:
                    safe_write(f"エラー {os.path.basename(item['path'])}: {single_exc}")
                    progress.update(1)
            return

        elapsed = time.time() - started
        inferred += len(valid_items)
        inferred_time += elapsed
        batch_history.append({"count": len(valid_items), "time": elapsed})
        update_progress_postfix()
        for item, prediction in zip(valid_items, predictions):
            try:
                decode_and_finalize(item, prediction)
            except Exception as exc:
                safe_write(f"エラー {os.path.basename(item['path'])}: {exc}")
                progress.update(1)

    aborted = False
    try:
        for image_path in target_files:
            started = time.time()
            try:
                existing_tags = et_wrapper.get_tags(image_path) if need_exiftool else []
                raw_scores = (
                    extract_raw_rating_scores(metadata, existing_tags)
                    if not args.force
                    else None
                )

                if raw_scores is not None and args.rating_thresh is None:
                    probabilities = np.zeros(metadata.label_count, dtype=np.float32)
                    for name, score in zip(DBV4_RATING_NAMES, raw_scores):
                        probabilities[metadata.rating_indices[name]] = score
                    rating = calculate_rating(
                        metadata,
                        probabilities,
                        None,
                        args.ignore_sensitive,
                        float(APP_CONFIG.get("general_threshold", 0.40)),
                    )
                    skipped += 1
                    skipped_time += time.time() - started
                    update_progress_postfix()
                    finalize(image_path, existing_tags, [], rating, None)
                    continue

                item = {"path": image_path, "existing_tags": existing_tags}
                if is_client:
                    prediction = client_predict(
                        f"http://{args.host}:{args.port}",
                        image_path,
                        metadata,
                        int(APP_CONFIG.get("client_timeout", 15)),
                    )
                    inferred += 1
                    inferred_time += time.time() - started
                    update_progress_postfix()
                    decode_and_finalize(item, prediction)
                else:
                    pending.append(item)
                    if len(pending) >= batch_size:
                        run_batch(pending[:batch_size])
                        pending = pending[batch_size:]
            except urllib.error.HTTPError as exc:
                safe_write(f"サーバー処理エラー {os.path.basename(image_path)}: {exc}")
                progress.update(1)
                continue
            except (urllib.error.URLError, socket.timeout) as exc:
                if is_client:
                    safe_write(f"接続エラー(タイムアウト含む): {exc}")
                    aborted = True
                    break
                safe_write(f"エラー {os.path.basename(image_path)}: {exc}")
                progress.update(1)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                safe_write(f"エラー {os.path.basename(image_path)}: {exc}")
                progress.update(1)
    except KeyboardInterrupt:
        safe_write("\n[INFO] 中断されました。")
        aborted = True
    finally:
        if runtime and pending and not aborted:
            run_batch(pending)
        if executor:
            executor.shutdown(wait=True)
        if need_exiftool:
            et_wrapper.stop()
        progress.close()

    if is_pixiv and aborted:
        safe_write("[WARN] 処理が中断されたため、Pixivフォルダの移動をスキップします。")
    elif is_pixiv:
        for source_dir, group_files in pixiv_groups.items():
            absolute_files = [os.path.abspath(path) for path in group_files]
            if not all(path in pixiv_rating_by_path for path in absolute_files):
                safe_write(
                    f"[WARN] Pixiv画像フォルダは全画像のスキャンが完了していないため移動をスキップ: {source_dir}"
                )
                continue
            target_rating = get_pixiv_move_rating(
                [pixiv_rating_by_path[path] for path in absolute_files]
            )
            if target_rating is None:
                continue
            pixiv_target_groups += 1
            moved_paths, moved_count = organize_pixiv_folder(
                absolute_files,
                target_rating,
                base_dirs,
            )
            if moved_count:
                pixiv_moved_groups += 1
            organized += moved_count
            pixiv_moved_paths.update(moved_paths)

        if pixiv_moved_paths and report_data:
            for report in report_data:
                original_path = os.path.abspath(report["path"])
                if original_path in pixiv_moved_paths:
                    report["path"] = pixiv_moved_paths[original_path]

    print("\n[完了] 処理結果サマリー:")
    if warmup_time > 0:
        print(f"  ・ウォームアップ時間 (エンジン事前構築): {warmup_time:.2f} 秒")

    timing = inference_timing_summary(inferred, inferred_time, batch_history)
    if inferred:
        print(
            f"  ・推論実行ファイル (DBV4 AI演算あり): {inferred} 枚 | "
            f"速度: {timing['fps']:.2f} img/s ({timing['ms_per_image']:.1f} ms/img)"
        )
        if timing["outlier_detected"]:
            first_count = int(timing["first_count"])
            first_time = float(timing["first_time"])
            first_ms = first_time / first_count * 1000.0 if first_count else 0.0
            all_speed = inferred / inferred_time if inferred_time else 0.0
            print(
                f"       ├─ 初回バッチ処理時間: {first_time:.2f} 秒 "
                f"({first_ms:.1f} ms/img)"
            )
            print(
                f"       ├─ 初回を含む全推論速度: {all_speed:.2f} img/s "
                f"(合計処理時間: {inferred_time:.2f} 秒)"
            )
            print(
                f"       └─ 初回バッチの遅延 ({first_time:.2f}秒) を"
                "コンパイル外れ値としてメイン速度から除外しました。"
            )
    else:
        print("  ・推論実行ファイル (DBV4 AI演算あり): 0 枚")

    if skipped:
        skip_speed = skipped / skipped_time if skipped_time else 0.0
        skip_ms = skipped_time / skipped * 1000.0
        print(
            f"  ・演算スキップファイル (DBV4 score): {skipped} 枚 | "
            f"速度: {skip_speed:.2f} img/s ({skip_ms:.1f} ms/img)"
        )
    else:
        print("  ・演算スキップファイル (DBV4 score): 0 枚")
    if is_pixiv:
        print(
            f"  ・Pixiv移動対象: {pixiv_target_groups}フォルダ / "
            f"移動完了 {pixiv_moved_groups}フォルダ"
        )
    print(f"  ・詳細: タグ書き込み {processed} 枚, 整理移動 {organized} 枚")

    if not args.no_report:
        if report_data:
            with open(REPORT_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False)
            print(f"[INFO] レポート用ログを保存: {REPORT_LOG_FILE}")
            if make_report:
                print("[INFO] HTMLレポートを生成中...")
                make_report.make_report()
            else:
                print("[WARN] make_reportモジュールがないためHTML生成をスキップします。")
        else:
            print("[INFO] レポート対象データがありませんでした。")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DBV4 Tagger Universal (日本語版)")
    parser.add_argument("images", nargs="*", help="処理対象の画像またはフォルダパス")
    parser.add_argument("--mode", choices=["standalone", "server", "client"], default="standalone")
    parser.add_argument("--no-tag", action="store_true", help="タグ付け処理を行わない")
    parser.add_argument("--organize", action="store_true", help="レーティングに基づきフォルダ整理を行う")
    parser.add_argument(
        "--pixiv",
        action="store_true",
        help="Pixiv整理モード（画像を含むフォルダ単位で判定し、R17以上を含むフォルダの全画像を一括移動。空フォルダは削除）",
    )
    parser.add_argument("--no-report", action="store_true", help="HTMLレポートを作成しない")
    parser.add_argument(
        "--thresh",
        type=float,
        default=None,
        help="DBV4のtag best_thresholdを上書きする明示的な閾値",
    )
    parser.add_argument("--gpu", action="store_true", help="GPUを使用する")
    parser.add_argument("--batch-size", type=int, default=4, help="推論バッチサイズ")
    parser.add_argument("--io-workers", type=int, default=-1, help="画像読込みの並列数（-1=自動）")
    parser.add_argument("--force", action="store_true", help="既存DBV4 scoreを使わず強制再推論")
    parser.add_argument("--recursive", action="store_const", const=True, default=None, help="再帰検索ON")
    parser.add_argument("--no-recursive", action="store_const", const=False, dest="recursive", help="再帰検索OFF")
    parser.add_argument(
        "--model-profile",
        default=None,
        metavar="NAME",
        help="DBV4モデルプロファイル（configのカスタム定義も指定可）",
    )
    parser.add_argument("--model-repo", default=None, help="DBV4モデル/metadataのHugging FaceリポジトリID")
    parser.add_argument("--model-file", default=None, help="ONNXモデルファイル名またはパス")
    parser.add_argument("--tags-file", default=None, help="selected_tags.csvのファイル名またはパス")
    parser.add_argument("--host", default=None, help="Client接続先ホスト")
    parser.add_argument("--port", type=int, default=None, help="Server/Clientポート")
    parser.add_argument(
        "--sensitive-split-mode",
        choices=[2, 4, 6],
        type=int,
        default=None,
        help="旧CLI互換（DBV4ではR-15/R-17の5段階固定）",
    )
    parser.add_argument("--record-ratio", action="store_true", default=None, help="rating scoreをXMPへ保存")
    parser.add_argument("--no-record-ratio", action="store_false", dest="record_ratio", help="rating scoreのXMP保存を無効化")
    parser.add_argument("--rating-thresh", type=float, default=None, help="非General rating判定閾値（旧CLI互換）")
    parser.add_argument("--ignore-sensitive", action="store_true", help="Sensitive判定をGeneralとして扱う")
    parser.add_argument("--gen-config", action="store_true", help="config.jsonを生成・更新")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.gen_config:
        load_config()
        return

    profiles = APP_CONFIG.get("model_profiles", MODEL_PROFILES)
    if args.model_profile is None:
        args.model_profile = APP_CONFIG.get("model_profile", "balanced")
    if args.model_profile not in profiles:
        raise SystemExit(f"[ERROR] 不明なDBV4モデルプロファイルです: {args.model_profile}")

    if args.sensitive_split_mode is not None:
        print("[WARN] --sensitive-split-mode はDBV4移行後は非推奨です。R-15/R-17は5段階固定で処理します。")

    if args.record_ratio is not None:
        APP_CONFIG["record_raw_score"] = bool(args.record_ratio)
        APP_CONFIG["record_rating_percentages"] = bool(args.record_ratio)

    if args.host is None:
        if args.mode == "client":
            hosts = APP_CONFIG.get("server_hosts", ["localhost"])
            hosts = [hosts] if isinstance(hosts, str) else hosts
            if len(hosts) == 1:
                args.host = hosts[0]
            else:
                print(f"{Colors.CYAN}[INFO] 接続先サーバーを選択してください。{Colors.RESET}")
                for index, host in enumerate(hosts, 1):
                    print(f"  {index}: {host}")
                while True:
                    try:
                        index = int(input("番号: ")) - 1
                        if 0 <= index < len(hosts):
                            args.host = hosts[index]
                            break
                    except ValueError:
                        pass
        else:
            args.host = "localhost"
    if args.port is None:
        args.port = int(APP_CONFIG.get("server_port", 5000))

    if args.pixiv:
        args.organize = True
        args.recursive = True

    if args.mode == "server":
        run_server(args)
        return
    if not args.images:
        print("[案内] 画像ファイルまたはフォルダを指定してください。")
        create_parser().print_help()
        return
    process_images(args)


if __name__ == "__main__":
    main()
