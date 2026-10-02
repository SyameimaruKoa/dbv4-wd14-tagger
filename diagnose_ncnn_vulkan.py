"""Compare Vulkan kernel options against the same DBV4 CPU reference."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import ncnn

from benchmark_ncnn import ResourceMonitor, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--disable', action='append', default=[],
                        choices=('use_winograd_convolution', 'use_sgemm_convolution',
                                 'use_shader_pack8', 'use_shader_local_memory'))
    args = parser.parse_args()
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
