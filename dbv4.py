import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download


DBV4_RATING_NAMES = ("general", "sensitive", "questionable", "explicit")
DBV4_CATEGORY_IDS = {"general": 0, "character": 4, "rating": 9}

MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "compact_manual": {
        "repo_id": "animetimm/repvit_m2_3.dbv4-full",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "preprocess_file": "preprocess.json",
        "categories_file": "categories.json",
        "thresholds_file": "thresholds.csv",
        "access_notice": "Hugging Faceで管理者によるアクセス承認が必要です。",
        "requires_manual_approval": True,
    },
    "lightweight": {
        "repo_id": "animetimm/mobilenetv4_conv_aa_large.dbv4-full",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "preprocess_file": "preprocess.json",
        "categories_file": "categories.json",
        "thresholds_file": "thresholds.csv",
    },
    "medium_manual": {
        "repo_id": "animetimm/convformer_s36.dbv4-full",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "preprocess_file": "preprocess.json",
        "categories_file": "categories.json",
        "thresholds_file": "thresholds.csv",
        "access_notice": "Hugging Faceで管理者によるアクセス承認が必要です。",
        "requires_manual_approval": True,
    },
    "balanced": {
        "repo_id": "animetimm/caformer_b36.dbv4-full",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "preprocess_file": "preprocess.json",
        "categories_file": "categories.json",
        "thresholds_file": "thresholds.csv",
    },
    "high": {
        "repo_id": "animetimm/eva02_large_patch14_448.dbv4-full",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "preprocess_file": "preprocess.json",
        "categories_file": "categories.json",
        "thresholds_file": "thresholds.csv",
        "runtime_warning": (
            "highはDirectML実測でbalancedより約2.8倍遅く、ultraとの差も小さいため、"
            "新規利用には非推奨です。互換性と比較試験用に保持しています。"
        ),
    },
    "ultra": {
        "repo_id": "itterative/convnextv2_huge.dbv4-full-onnx",
        "metadata_repo_id": "animetimm/convnextv2_huge.dbv4-full",
        "model_file": "model.onnx",
        "model_external_files": ["model.onnx_data"],
        "vram_warning": (
            "ultraはRTX 2070 Max-Q・DirectML・batch-size=4で"
            "ピーク6,583MiB、測定前を除く増分5,743MiBを確認済みです。"
            "8GB以上のGPUを推奨し、空き容量不足時はbatch-sizeを下げてください。"
        ),
        "tags_file": "selected_tags.csv",
        "preprocess_file": "preprocess.json",
        "categories_file": "categories.json",
        "thresholds_file": "thresholds.csv",
    },
    "wd14_v3": {
        "family": "wd14_v3",
        "repo_id": "SmilingWolf/wd-swinv2-tagger-v3",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "preprocess": {
            "alpha_background": "white",
            "output_channel_order": "BGR",
            "output_scale": "raw_255",
            "test": [
                {"type": "PadToSquare", "background_color": "white"},
                {"type": "Resize", "size": [448, 448], "interpolation": "bicubic"},
            ],
        },
    },
    "future_1b": {
        "family": "dbv4",
        "repo_id": "animetimm/vit_giantopt_patch16_siglip_384.dbv4-full",
        "available": False,
        "unavailable_reason": (
            "future_1bは将来対応予約です。公式repoは現在Safetensorsのみで、"
            "ONNXが公開されていないため、このONNX Runtime版ではまだ実行できません。"
        ),
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "preprocess_file": "preprocess.json",
        "categories_file": "categories.json",
        "thresholds_file": "thresholds.csv",
    },
}


def clone_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(profile))


def get_model_profile(name: str, profiles: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    source = profiles or MODEL_PROFILES
    if name not in source:
        raise KeyError(f"未知のDBV4モデルプロファイルです: {name}")
    result = clone_profile(source[name])
    result["profile_name"] = name
    return result


def _local_path(filename: Optional[str], base_dir: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    candidates = []
    if os.path.isabs(filename):
        candidates.append(filename)
    if base_dir:
        candidates.append(os.path.join(base_dir, filename))
    candidates.append(os.path.abspath(filename))
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def resolve_model_artifact(
    repo_id: Optional[str],
    filename: str,
    base_dir: Optional[str] = None,
    required: bool = True,
) -> Optional[str]:
    path = _local_path(filename, base_dir)
    if path:
        return path
    if not repo_id:
        if required:
            raise FileNotFoundError(f"DBV4アーティファクトが見つかりません: {filename}")
        return None
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    except Exception:
        if required:
            raise
        return None


def materialize_external_onnx_bundle(
    model_path: str,
    external_paths: Sequence[str],
    repo_id: str,
    base_dir: Optional[str],
) -> str:
    if not external_paths:
        return model_path
    cache_base = os.path.abspath(
        base_dir or os.environ.get("DBV4_DATA_DIR") or os.getcwd()
    )
    snapshot_name = os.path.basename(os.path.dirname(model_path))
    bundle_key = hashlib.sha256(
        f"{repo_id}\0{snapshot_name}".encode("utf-8")
    ).hexdigest()[:16]
    bundle_dir = os.path.join(cache_base, "models", bundle_key)
    os.makedirs(bundle_dir, exist_ok=True)

    def materialize(source: str) -> str:
        destination = os.path.join(bundle_dir, os.path.basename(source))
        source_size = os.path.getsize(source)
        if os.path.isfile(destination) and os.path.getsize(destination) == source_size:
            return destination
        if os.path.lexists(destination):
            os.remove(destination)
        try:
            os.link(os.path.realpath(source), destination)
        except OSError:
            temporary = f"{destination}.{os.getpid()}.part"
            try:
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)
        return destination

    bundled_model = materialize(model_path)
    for external_path in external_paths:
        materialize(external_path)
    print(f"[INFO] ONNX外部データを同一ディレクトリに配置: {bundle_dir}")
    return bundled_model


def _load_categories(path: Optional[str]) -> Dict[int, str]:
    categories = {value: key for key, value in DBV4_CATEGORY_IDS.items()}
    if not path:
        return categories
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and "category" in item and "name" in item:
                    categories[int(item["category"])] = str(item["name"])
    except (OSError, TypeError, ValueError):
        pass
    return categories


def _load_category_thresholds(path: Optional[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not path:
        return result
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("name")
                value = row.get("threshold")
                if name and value not in (None, ""):
                    result[str(name)] = float(value)
    except (OSError, TypeError, ValueError):
        pass
    return result


def _read_selected_tags(
    path: str,
    categories: Dict[int, str],
    category_thresholds: Dict[str, float],
) -> Tuple[List[str], List[str], Dict[str, float]]:
    labels: List[str] = []
    category_names: List[str] = []
    thresholds: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "name" not in reader.fieldnames:
            raise ValueError("selected_tags.csv に name 列がありません。")
        for row in reader:
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            try:
                category_id = int(row.get("category", 0))
            except (TypeError, ValueError):
                category_id = 0
            category = categories.get(category_id, str(category_id))
            raw_threshold = row.get("best_threshold")
            try:
                threshold = float(raw_threshold) if raw_threshold not in (None, "") else None
            except (TypeError, ValueError):
                threshold = None
            if threshold is None:
                threshold = float(category_thresholds.get(category, 0.35))
            labels.append(name)
            category_names.append(category)
            thresholds[name] = float(threshold)
    return labels, category_names, thresholds


def _metadata_hash(
    paths: Iterable[Optional[str]],
    extra_values: Iterable[Any] = (),
) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path or not os.path.exists(path):
            digest.update(b"<missing>")
            continue
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                digest.update(chunk)
    for value in extra_values:
        digest.update(
            json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class DBV4Metadata:
    repo_id: str
    profile_name: str
    model_path: str
    tags_path: str
    preprocess_path: str
    categories_path: Optional[str]
    thresholds_path: Optional[str]
    labels: Tuple[str, ...]
    categories: Tuple[str, ...]
    tag_thresholds: Dict[str, float]
    rating_indices: Dict[str, int]
    metadata_version: str
    family: str = "dbv4"
    preprocess_config: Optional[Dict[str, Any]] = None

    @classmethod
    def load(
        cls,
        profile: Dict[str, Any],
        base_dir: Optional[str] = None,
        model_repo_override: Optional[str] = None,
        model_file_override: Optional[str] = None,
        tags_file_override: Optional[str] = None,
        load_model: bool = True,
    ) -> "DBV4Metadata":
        repo_id = model_repo_override or profile["repo_id"]
        metadata_repo_id = (
            repo_id
            if model_repo_override
            else profile.get("metadata_repo_id", repo_id)
        )
        model_path = ""
        if load_model:
            model_filename = model_file_override or profile.get("model_file", "model.onnx")
            model_path = resolve_model_artifact(
                repo_id,
                model_filename,
                base_dir,
                True,
            ) or ""
            external_paths = []
            if model_file_override is None:
                for external_filename in profile.get("model_external_files", []):
                    external_paths.append(
                        resolve_model_artifact(
                            repo_id,
                            external_filename,
                            base_dir,
                            True,
                        ) or ""
                    )
            if external_paths:
                model_path = materialize_external_onnx_bundle(
                    model_path,
                    external_paths,
                    repo_id,
                    base_dir,
                )
        tags_path = resolve_model_artifact(
            metadata_repo_id,
            tags_file_override or profile.get("tags_file", "selected_tags.csv"),
            base_dir,
            True,
        )
        preprocess_config = profile.get("preprocess")
        if preprocess_config is not None:
            preprocess_config = clone_profile(preprocess_config)
            preprocess_path = ""
        else:
            preprocess_path = resolve_model_artifact(
                metadata_repo_id,
                profile.get("preprocess_file", "preprocess.json"),
                base_dir,
                True,
            )
        categories_path = resolve_model_artifact(
            metadata_repo_id,
            profile.get("categories_file", "categories.json"),
            base_dir,
            False,
        )
        thresholds_path = resolve_model_artifact(
            metadata_repo_id,
            profile.get("thresholds_file", "thresholds.csv"),
            base_dir,
            False,
        )

        category_map = _load_categories(categories_path)
        category_thresholds = _load_category_thresholds(thresholds_path)
        labels, categories, thresholds = _read_selected_tags(
            tags_path,
            category_map,
            category_thresholds,
        )
        rating_indices = {
            label: index
            for index, (label, category) in enumerate(zip(labels, categories))
            if category == "rating" and label in DBV4_RATING_NAMES
        }
        missing = [name for name in DBV4_RATING_NAMES if name not in rating_indices]
        if missing:
            raise ValueError("DBV4 rating metadataが不足しています: " + ", ".join(missing))

        return cls(
            repo_id=repo_id,
            profile_name=str(profile.get("profile_name", "custom")),
            model_path=model_path,
            tags_path=tags_path,
            preprocess_path=preprocess_path,
            categories_path=categories_path,
            thresholds_path=thresholds_path,
            labels=tuple(labels),
            categories=tuple(categories),
            tag_thresholds=thresholds,
            rating_indices=rating_indices,
            metadata_version=_metadata_hash(
                [tags_path, preprocess_path, categories_path, thresholds_path],
                [preprocess_config] if preprocess_config is not None else [],
            ),
            family=str(profile.get("family", "dbv4")),
            preprocess_config=preprocess_config,
        )

    @property
    def label_count(self) -> int:
        return len(self.labels)

    @property
    def rating_tags_marker(self) -> str:
        prefix = "wd14" if self.family == "wd14_v3" else "dbv4"
        return f"{prefix}_model:{self.repo_id}"

    def threshold_for(self, label: str, override: Optional[float] = None) -> float:
        return float(override) if override is not None else float(self.tag_thresholds.get(label, 0.35))

    def get_rating_scores(self, probabilities: Sequence[float]) -> Dict[str, float]:
        values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if len(values) != self.label_count:
            raise ValueError(
                f"DBV4 output size mismatch: model={len(values)}, metadata={self.label_count}"
            )
        return {
            name: float(values[index])
            for name, index in self.rating_indices.items()
        }

    def decode_tags(
        self,
        probabilities: Sequence[float],
        threshold_override: Optional[float] = None,
    ) -> List[str]:
        values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if len(values) != self.label_count:
            raise ValueError(
                f"DBV4 output size mismatch: model={len(values)}, metadata={self.label_count}"
            )
        return [
            label
            for index, (label, category) in enumerate(zip(self.labels, self.categories))
            if float(values[index]) >= self.threshold_for(label, threshold_override)
            and category != "rating"
        ]

    def summary(self) -> Dict[str, Any]:
        return {
            "protocol": 1,
            "model_id": self.repo_id,
            "profile": self.profile_name,
            "metadata_version": self.metadata_version,
            "output_size": self.label_count,
            "rating_labels": list(DBV4_RATING_NAMES),
        }


class DBV4Preprocessor:
    def __init__(self, payload: Dict[str, Any]):
        self.steps = self._steps(payload)
        self.alpha_background = self._color(payload.get("alpha_background", "black"))
        self.output_channel_order = str(
            payload.get("output_channel_order", "RGB")
        ).upper()
        self.output_scale = str(payload.get("output_scale", "unit")).lower()

    @classmethod
    def from_metadata(cls, metadata: DBV4Metadata) -> "DBV4Preprocessor":
        if metadata.preprocess_config is not None:
            return cls(metadata.preprocess_config)
        with open(metadata.preprocess_path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    @staticmethod
    def _steps(payload: Dict[str, Any]) -> List[Any]:
        value = payload.get("test", payload)
        if isinstance(value, dict):
            value = value.get("transforms", value.get("steps", []))
        if not isinstance(value, list):
            raise ValueError("preprocess.json の test 定義を解釈できません。")
        return value

    @staticmethod
    def _step(step: Any) -> Tuple[str, Dict[str, Any]]:
        if isinstance(step, str):
            return step.lower(), {}
        if not isinstance(step, dict):
            raise ValueError(f"未知のpreprocess step形式です: {step!r}")
        name = str(step.get("name") or step.get("type") or step.get("transform") or "").lower()
        params = {
            key: value
            for key, value in step.items()
            if key not in {"name", "type", "transform", "params", "kwargs"}
        }
        for container_name in ("params", "kwargs"):
            nested_params = step.get(container_name)
            if isinstance(nested_params, dict):
                params.update(nested_params)
        return name, params

    @staticmethod
    def _size(value: Any, params: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
        if isinstance(value, (int, float)):
            size = int(value)
            return size, size
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return int(value[0]), int(value[1])
        params = params or {}
        height = params.get("height")
        width = params.get("width")
        if height is not None and width is not None:
            return int(height), int(width)
        raise ValueError(f"画像サイズを解釈できません: size={value!r}, params={params!r}")

    @staticmethod
    def _color(value: Any) -> Tuple[int, int, int]:
        colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "gray": (128, 128, 128),
            "grey": (128, 128, 128),
        }
        if isinstance(value, str) and value.lower() in colors:
            return colors[value.lower()]
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return (int(value[0]),) * 3
            if len(value) == 3:
                return tuple(int(v) for v in value)
        if isinstance(value, (int, float)):
            return (int(value),) * 3
        return (255, 255, 255)

    @staticmethod
    def _resampling(value: Any) -> int:
        table = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
            "box": Image.Resampling.BOX,
            "hamming": Image.Resampling.HAMMING,
        }
        return table.get(str(value or "bicubic").lower(), Image.Resampling.BICUBIC)

    def __call__(self, image: Image.Image) -> np.ndarray:
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, self.alpha_background + (255,))
            background.alpha_composite(rgba)
            current = background.convert("RGB")
        else:
            current = image.convert("RGB")
        normalize_steps: List[Dict[str, Any]] = []
        for raw_step in self.steps:
            name, params = self._step(raw_step)
            if name in {"padtosize", "pad_to_size"}:
                target_h, target_w = self._size(params.get("size"), params)
                width, height = current.size
                if width < target_w or height < target_h:
                    canvas = Image.new(
                        "RGB",
                        (max(width, target_w), max(height, target_h)),
                        self._color(params.get("background_color", "white")),
                    )
                    canvas.paste(
                        current,
                        (
                            (canvas.width - width) // 2,
                            (canvas.height - height) // 2,
                        ),
                    )
                    current = canvas
            elif name in {"padtosquare", "pad_to_square"}:
                width, height = current.size
                size = max(width, height)
                canvas = Image.new(
                    "RGB",
                    (size, size),
                    self._color(params.get("background_color", "white")),
                )
                canvas.paste(current, ((size - width) // 2, (size - height) // 2))
                current = canvas
            elif name == "resize":
                value = params.get("size")
                resample = self._resampling(params.get("interpolation"))
                if isinstance(value, (int, float)):
                    size = int(value)
                    width, height = current.size
                    if width == height:
                        target = (size, size)
                    elif width < height:
                        target = (size, max(1, round(height * size / width)))
                    else:
                        target = (max(1, round(width * size / height)), size)
                else:
                    h, w = self._size(value, params)
                    target = (w, h)
                current = current.resize(target, resample)
            elif name in {"centercrop", "center_crop"}:
                h, w = self._size(params.get("size"), params)
                width, height = current.size
                left = max(0, (width - w) // 2)
                top = max(0, (height - h) // 2)
                current = current.crop((left, top, left + w, top + h))
            elif name in {
                "maybetotensor",
                "maybe_to_tensor",
                "totensor",
                "to_tensor",
                "convertimagedtype",
                "convert_image_dtype",
                "identity",
                "",
            }:
                continue
            elif name == "normalize":
                normalize_steps.append(params)
            else:
                raise ValueError(f"未対応のDBV4前処理です: {name}")

        array = np.asarray(current, dtype=np.float32)
        if self.output_channel_order == "BGR":
            array = array[:, :, ::-1]
        elif self.output_channel_order != "RGB":
            raise ValueError(f"未対応の出力色順序です: {self.output_channel_order}")
        if self.output_scale not in {"raw_255", "raw", "255"}:
            array = array / 255.0
        array = np.transpose(array, (2, 0, 1))
        for params in normalize_steps:
            mean = np.asarray(
                params.get("mean", [0.485, 0.456, 0.406]),
                dtype=np.float32,
            ).reshape(3, 1, 1)
            std = np.asarray(
                params.get("std", [0.229, 0.224, 0.225]),
                dtype=np.float32,
            ).reshape(3, 1, 1)
            array = (array - mean) / std
        return array.astype(np.float32)


def infer_output_to_probabilities(raw_output: Any) -> np.ndarray:
    values = np.asarray(raw_output, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("DBV4 outputに非有限値があります。")
    if values.size == 0:
        raise ValueError("DBV4 outputが空です。")
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        values = 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))
    return np.clip(values, 0.0, 1.0)


def select_output_name(outputs: Sequence[Any], label_count: int) -> str:
    if not outputs:
        raise ValueError("DBV4 modelに出力がありません。")

    preferred_names = ("prediction", "probabilities", "probability", "probs", "logits")
    matching: List[Any] = []
    dynamic: List[Any] = []
    for output in outputs:
        shape = getattr(output, "shape", None)
        if not isinstance(shape, (list, tuple)) or not shape:
            dynamic.append(output)
            continue
        size = shape[-1]
        if isinstance(size, (int, np.integer)):
            if int(size) == label_count:
                matching.append(output)
        else:
            dynamic.append(output)

    for preferred_name in preferred_names:
        for output in matching:
            if str(getattr(output, "name", "")).lower() == preferred_name:
                return str(output.name)
    if len(matching) == 1:
        return str(matching[0].name)

    for preferred_name in preferred_names:
        for output in dynamic:
            if str(getattr(output, "name", "")).lower() == preferred_name:
                return str(output.name)

    details = [
        f"{getattr(output, 'name', '<unnamed>')}:{getattr(output, 'shape', None)}"
        for output in outputs
    ]
    raise ValueError(
        f"DBV4ラベル数 {label_count} に対応する出力を特定できません: "
        + ", ".join(details)
    )


def detect_input_layout(input_shape: Sequence[Any]) -> str:
    if len(input_shape) != 4:
        raise ValueError(f"4次元入力を想定しています: {input_shape}")
    if input_shape[1] == 3:
        return "NCHW"
    if input_shape[-1] == 3:
        return "NHWC"
    return "NCHW"


def adapt_input_layout(batch_nchw: np.ndarray, input_shape: Sequence[Any]) -> np.ndarray:
    if detect_input_layout(input_shape) == "NHWC":
        return np.transpose(batch_nchw, (0, 2, 3, 1)).astype(np.float32)
    return batch_nchw.astype(np.float32)
