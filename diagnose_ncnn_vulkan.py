"""Compare Vulkan kernel options against the same DBV4 CPU reference."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import ncnn
import numpy as np

from benchmark_ncnn import ResourceMonitor, benchmark_image, run
from dbv4 import DBV4Metadata, DBV4Preprocessor, MODEL_PROFILES


def compare_blobs(args):
    root = Path(__file__).resolve().parent
    metadata = DBV4Metadata.load(MODEL_PROFILES['ultra'], base_dir=str(root), load_model=False)
    tensor = np.ascontiguousarray(DBV4Preprocessor.from_metadata(metadata)(benchmark_image()))
    prefix = root / '.dbv4/models/ultra/model'
    reference = {}
    comparisons = []
    for gpu in (False, True):
        net = ncnn.Net()
        net.opt.use_vulkan_compute = gpu
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_arithmetic = False
        net.opt.num_threads = 4
        for name in args.disable:
            setattr(net.opt, name, False)
        if gpu:
            net.set_vulkan_device(0)
        if net.load_param(str(prefix.with_suffix('.ncnn.param'))) != 0 or net.load_model(
                str(prefix.with_suffix('.ncnn.bin'))) != 0:
            raise RuntimeError('ncnn model load failed')
        for blob in args.blobs:
            # Recompute each requested prefix rather than retaining every
            # activation of the huge graph across diagnostic extractions.
            extractor = net.create_extractor()
            if extractor.input('in0', ncnn.Mat(tensor)) != 0:
                raise RuntimeError('ncnn input failed')
            status, output = extractor.extract(blob)
            if status != 0:
                raise RuntimeError(f'extract {blob}: {status}')
            values = np.asarray(output).copy()
            del extractor
            if not gpu:
                reference[blob] = values
                print(f'CPU blob {blob}: {values.shape}', flush=True)
                continue
            expected = reference.pop(blob)
            if values.shape != expected.shape:
                raise ValueError(f'blob {blob}: {values.shape} != {expected.shape}')
            difference = np.abs(values - expected)
            item = dict(blob=blob, shape=list(values.shape),
                        max_absolute_difference=float(difference.max()),
                        mean_absolute_difference=float(difference.mean()),
                        cpu_min=float(expected.min()), cpu_max=float(expected.max()),
                        allclose=bool(np.allclose(values, expected, atol=1e-4, rtol=1e-3)))
            comparisons.append(item)
            print(json.dumps(item), flush=True)
        net.clear()
        del net
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparisons, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blobs', nargs='+', help='Compare named intermediate blobs on CPU/Vulkan')
    parser.add_argument('--disable', action='append', default=[],
                        choices=('use_winograd_convolution', 'use_sgemm_convolution',
                                 'use_packing_layout', 'use_subgroup_ops'))
    args = parser.parse_args()
    if args.blobs:
        compare_blobs(args)
        return
    original_net = ncnn.Net

    def configured_net():
        net = original_net()
        for name in args.disable:
            setattr(net.opt, name, False)
        return net

    options = argparse.Namespace(provider='ncnn', profile='ultra', gpu_index=0,
                                 ncnn_precision='fp32', model_file=None,
                                 batch_size=1, warmup=1, iterations=3,
                                 reference=args.reference)
    with patch.object(ncnn, 'Net', configured_net), ResourceMonitor(None) as monitor:
        result, measured_start = run(options, monitor)
    result['resources'] = monitor.summary(measured_start)
    result['disabled_options'] = args.disable
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['comparison'], indent=2))


if __name__ == '__main__':
    main()
