"""Check converted model probabilities on CPU without requiring a Vulkan GPU.

This validates conversion only. GPU throughput is measured by benchmark_ncnn.py.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from benchmark_ncnn import benchmark_image, compare, package_versions, ResourceMonitor
from dbv4 import DBV4Metadata, DBV4Preprocessor, MODEL_PROFILES, infer_output_to_probabilities


def run(args):
    import ncnn

    root = Path(__file__).resolve().parent
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    metadata = DBV4Metadata.load(MODEL_PROFILES[args.profile], base_dir=str(root), load_model=False)
    tensor = np.ascontiguousarray(DBV4Preprocessor.from_metadata(metadata)(benchmark_image()))
    input_hash = hashlib.sha256(tensor.tobytes()).hexdigest()
    if input_hash != reference.get('input_sha256'):
        raise ValueError('基準と前処理済み入力が異なります')
    prefix = args.model_prefix or root / '.dbv4/models' / args.profile / 'model'
    with ResourceMonitor() as monitor:
        started = time.perf_counter()
        net = ncnn.Net()
        net.opt.use_vulkan_compute = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_arithmetic = False
        net.opt.num_threads = args.threads
        if (net.load_param(str(prefix.with_suffix('.ncnn.param'))) != 0
                or net.load_model(str(prefix.with_suffix('.ncnn.bin'))) != 0):
            raise RuntimeError('ncnnモデルの読み込みに失敗しました')
        load_seconds = time.perf_counter() - started
        extractor = net.create_extractor()
        # Mat borrows tensor's storage; retain tensor until extraction completes.
        if extractor.input(list(net.input_names())[0], ncnn.Mat(tensor)) != 0:
            raise RuntimeError('ncnn入力に失敗しました')
        measured_start = time.perf_counter()
        status, output = extractor.extract(list(net.output_names())[0])
        inference_ms = (time.perf_counter() - measured_start) * 1000
        if status != 0:
            raise RuntimeError(f'ncnn推論に失敗しました: {status}')
        probabilities = infer_output_to_probabilities(np.asarray(output).reshape(-1))
        if probabilities.size != metadata.label_count:
            raise ValueError('ncnn出力ラベル数が一致しません')
    result = {
        'provider': 'ncnn CPU (conversion validation only)',
        'precision': 'fp32', 'profile': args.profile, 'batch_size': 1,
        'warmup': 0, 'iterations': 1, 'num_threads': args.threads,
        'load_seconds': load_seconds, 'single_cold_inference_ms': inference_ms,
        'input_sha256': input_hash, 'metadata_version': metadata.metadata_version,
        'label_count': metadata.label_count, 'package_versions': package_versions(),
        'comparison': compare(reference, probabilities, metadata),
        'rating_scores': metadata.get_rating_scores(probabilities),
        'selected_tags': metadata.decode_tags(probabilities),
        'probabilities': probabilities.tolist(),
        'resources': monitor.summary(measured_start),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['comparison'])
    np.testing.assert_allclose(probabilities, reference['probabilities'], atol=1e-4, rtol=1e-3)
    if result['comparison']['selected_tag_symmetric_difference']:
        raise ValueError('推奨閾値での採用タグが基準と異なります')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=MODEL_PROFILES, default='ultra')
    parser.add_argument('--model-prefix', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    run(args)


if __name__ == '__main__':
    main()
