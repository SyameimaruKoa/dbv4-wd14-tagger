"""Convert an AnimeTimm DBV4 model to a portable ncnn cache.

Run in a Python environment with torch, timm, pnnx, and huggingface_hub.
The source repository must be accessible to the current Hugging Face account.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".dbv4" / "huggingface"))


def _portable_grn_forward(self, x):
    """Equivalent GRN without torch.addcmul, which pnnx cannot lower."""
    x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
    x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
    return x + self.bias.view(self.wb_shape) + self.weight.view(self.wb_shape) * (x * x_n)


def replace_grn_addcmul(model) -> int:
    from timm.layers.grn import GlobalResponseNorm

    count = 0
    for layer in model.modules():
        if isinstance(layer, GlobalResponseNorm):
            layer.forward = types.MethodType(_portable_grn_forward, layer)
            count += 1
    return count


def convert(source_repo: str, destination: Path, input_size: int,
            expected_labels: int) -> tuple[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    param = destination / "model.ncnn.param"
    weights = destination / "model.ncnn.bin"
    manifest = destination / "model.ncnn.json"
    if param.is_file() and weights.is_file() and manifest.is_file():
        cached = json.loads(manifest.read_text(encoding="utf-8"))
        if (cached.get("source_repo") == source_repo
                and cached.get("input_size") == input_size
                and cached.get("label_count") == expected_labels
                and cached.get("fp16") is False):
            return param, weights

    import timm
    import torch

    # pnnx writes several large intermediate files. Keep them outside the cache
    # and publish the two runtime files only after conversion has succeeded.
    with tempfile.TemporaryDirectory(prefix="dbv4-pnnx-", dir=destination) as work:
        stem = Path(work) / "model"
        model = timm.create_model("hf-hub:" + source_repo, pretrained=True).eval()
        replace_grn_addcmul(model)
        dummy = torch.zeros((1, 3, input_size, input_size), dtype=torch.float32)
        with torch.inference_mode():
            output = model(dummy)
        if output.numel() != expected_labels:
            raise ValueError(
                f"PyTorchモデルの出力数がDBV4 metadataと異なります: "
                f"{output.numel()} != {expected_labels}"
            )
        with torch.inference_mode():
            traced = torch.jit.trace(model, (dummy,), check_trace=False)
        traced.save(str(stem.with_suffix(".pt")))
        del traced, model, output
        gc.collect()
        del dummy
        gc.collect()
        pnnx_executable = Path(sys.executable).parent / ("pnnx.exe" if os.name == "nt" else "pnnx")
        if not pnnx_executable.is_file():
            raise RuntimeError(f"pnnx実行ファイルがありません: {pnnx_executable}")
        subprocess.run([
            str(pnnx_executable), str(stem.with_suffix(".pt")),
            f"inputshape=[1,3,{input_size},{input_size}]f32", "fp16=0",
        ], cwd=work, check=True)
        generated_param = Path(str(stem) + ".ncnn.param")
        generated_weights = Path(str(stem) + ".ncnn.bin")
        if not generated_param.is_file() or not generated_weights.is_file():
            raise RuntimeError("pnnxがmodel.ncnn.param/binを生成しませんでした。")
        unsupported = sorted({line.split()[0] for line in generated_param.read_text().splitlines()[2:]
                              if line.startswith(("pnnx.", "aten::"))})
        if unsupported:
            raise RuntimeError("pnnxに未変換の演算が残っています: " + ", ".join(unsupported))
        shutil.copyfile(generated_param, param.with_suffix(".param.tmp"))
        shutil.copyfile(generated_weights, weights.with_suffix(".bin.tmp"))
        os.replace(param.with_suffix(".param.tmp"), param)
        os.replace(weights.with_suffix(".bin.tmp"), weights)
        manifest.write_text(json.dumps({
            "source_repo": source_repo,
            "input_size": input_size,
            "label_count": expected_labels,
            "converter": "pnnx CLI",
            "fp16": False,
            "grn_addcmul_replaced": True,
        }, indent=2) + "\n", encoding="utf-8")
    return param, weights


def main() -> None:
    parser = argparse.ArgumentParser(description="DBV4 PyTorch model to ncnn")
    parser.add_argument("source_repo", help="timm compatible Hugging Face repository")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--labels", type=int, required=True)
    args = parser.parse_args()
    print("\n".join(map(str, convert(args.source_repo, args.destination,
                                     args.input_size, args.labels))))


if __name__ == "__main__":
    main()
