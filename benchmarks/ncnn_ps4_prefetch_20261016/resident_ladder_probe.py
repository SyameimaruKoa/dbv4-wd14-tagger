"""Resident cache escalation ladder on the real device.

Stages: 512 / 1024 / 1536 / 2048 MiB and full residency (all sections).
Each stage runs a fill cycle then a measured cycle over all 14 sections.
Guards: stop the ladder on allocation errors, critically low MemAvailable,
or non-finite outputs. Samples RSS/VRAM/GTT/MemAvailable/SwapFree and
pswpin/pswpout to detect swap churn.
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, '/opt/dbv4-wd14-tagger')
import numpy as np  # noqa: E402

import ncnn_backend  # noqa: E402
from ncnn_backend import StreamingNcnnRuntimeModel  # noqa: E402
ncnn_backend.configure_vulkan_environment()
import ncnn  # noqa: E402
from partition_ncnn_model import partition_model  # noqa: E402

part_directory, parts = partition_model(
    Path('/opt/dbv4-wd14-tagger/.dbv4/models/ultra/model'), 128)
total_mib = sum(p['bytes'] for p in parts) / 1048576.0
print(f'parts: {len(parts)} total {total_mib:.1f} MiB '
      f'largest {max(p["bytes"] for p in parts) / 1048576:.1f} MiB', flush=True)

runtime = StreamingNcnnRuntimeModel.__new__(StreamingNcnnRuntimeModel)
runtime.part_directory = part_directory
runtime.parts = parts
runtime.gpu_index = 0
runtime.metadata = SimpleNamespace(label_count=12476)
runtime.options = {'use_vulkan_compute': True, 'use_fp16_storage': False,
                   'use_fp16_packed': False, 'use_fp16_arithmetic': False,
                   'use_bf16_storage': False, 'use_bf16_packed': False,
                   'use_packing_layout': True, 'use_subgroup_ops': True,
                   'use_winograd_convolution': True, 'use_sgemm_convolution': True}
runtime._prepare_streaming_state()
print('auto resident budget:', runtime._resident_budget_mib, 'MiB', flush=True)

rng = np.random.default_rng(1026)
sample = np.ascontiguousarray(rng.standard_normal((3, 512, 512)).astype(np.float32))

DRM_DEVICE = next(
    (path for path in sorted(Path('/sys/class/drm').glob('card[0-9]*/device'))
     if (path / 'mem_info_vram_used').is_file()), None)


def read_system():
    result = {}
    try:
        fields = dict(line.split(':', 1) for line in
                      Path('/proc/meminfo').read_text().splitlines())
        result['available_mib'] = int(fields['MemAvailable'].split()[0]) / 1024
        result['swap_free_mib'] = int(fields['SwapFree'].split()[0]) / 1024
    except (OSError, KeyError, ValueError):
        pass
    try:
        for line in Path('/proc/vmstat').read_text().splitlines():
            if line.startswith('pswpout '):
                result['pswpout_pages'] = int(line.split()[1])
            elif line.startswith('pswpin '):
                result['pswpin_pages'] = int(line.split()[1])
    except (OSError, ValueError):
        pass
    if DRM_DEVICE:
        for name, key in (('mem_info_vram_used', 'vram_mib'),
                          ('mem_info_gtt_used', 'gtt_mib')):
            try:
                result[key] = int((DRM_DEVICE / name).read_text()) / 1048576
            except (OSError, ValueError):
                pass
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                result['rss_mib'] = int(line.split()[1]) / 1024
    except OSError:
        pass
    return result


class Sampler:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(0.5):
            self.samples.append(read_system())

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

    def mark(self):
        return len(self.samples)

    def window(self, start):
        return [s for s in self.samples[start:]]


class StageAbort(RuntimeError):
    pass


def stage_stats(window):
    stats = {}
    for key in ('available_mib', 'swap_free_mib', 'vram_mib', 'gtt_mib', 'rss_mib'):
        values = [s[key] for s in window if key in s]
        stats['min_' + key] = min(values) if values else None
        stats['max_' + key] = max(values) if values else None
    first = next((s for s in window if 'pswpout_pages' in s), None)
    last = next((s for s in reversed(window) if 'pswpout_pages' in s), None)
    if first and last:
        stats['pswpout_mib'] = (last['pswpout_pages'] - first['pswpout_pages']) * 4096 / 1048576
        stats['pswpin_mib'] = (last['pswpin_pages'] - first['pswpin_pages']) * 4096 / 1048576
    return stats


def dmesg_tail(n=6):
    try:
        output = subprocess.run(['dmesg', '--time-format', 'iso'],
                                capture_output=True, text=True, timeout=10).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return ['(dmesg unavailable)']
    lines = [line for line in output.splitlines()
             if any(token in line.lower() for token in
                    ('amdgpu', 'ring', 'reset', 'gpu fault', 'gpu reset', 'oom'))]
    return lines[-n:]


def run_cycle(label):
    current = sample
    per_section = []
    wall0 = time.perf_counter()
    for index, part in enumerate(runtime.parts):
        avail = read_system().get('available_mib', 1e9)
        if avail < 80:
            raise StageAbort(f'MemAvailable {avail:.0f} MiB < 80 MiB at section {index}')
        build0 = time.perf_counter()
        net = runtime._acquire_section_net(index)
        build_ms = (time.perf_counter() - build0) * 1000
        runtime._request_prefetch((index + 1) % len(runtime.parts))
        infer0 = time.perf_counter()
        extractor = net.create_extractor()
        mat = ncnn.Mat(current)
        if extractor.input(part['input'], mat) != 0:
            raise RuntimeError(f'input failed at section {index}')
        status, output = extractor.extract(part['output'])
        if status != 0:
            raise RuntimeError(f'extract failed at section {index}: {status}')
        current = np.ascontiguousarray(np.asarray(output))
        if not np.isfinite(current).all():
            raise RuntimeError(f'non-finite output at section {index}')
        output = extractor = None
        del mat
        infer_ms = (time.perf_counter() - infer0) * 1000
        runtime._release_section_net(index, net)
        per_section.append({'section': index, 'build_ms': build_ms, 'infer_ms': infer_ms})
        print(f'  [{label}] sec {index:02d} build {build_ms:9.1f} infer {infer_ms:8.1f} ms',
              flush=True)
    wall = time.perf_counter() - wall0
    build_sum = sum(entry['build_ms'] for entry in per_section) / 1000
    infer_sum = sum(entry['infer_ms'] for entry in per_section) / 1000
    print(f'  [{label}] wall {wall:.1f} s build {build_sum:.1f} s infer {infer_sum:.1f} s',
          flush=True)
    return {'per_section': per_section, 'wall_s': wall,
            'build_s': build_sum, 'infer_s': infer_sum}


stages = [512, 1024, 1536, 2048, int(total_mib) + 1]
report = {'total_mib': total_mib, 'auto_budget_mib': runtime._resident_budget_mib,
          'stages': []}

with Sampler() as sampler:
    for budget in stages:
        full = budget >= int(total_mib) + 1
        name = 'full' if full else str(budget)
        print(f'=== stage {name} MiB ===', flush=True)
        runtime._resident_budget_mib = budget
        stage = {'budget_mib': budget, 'status': 'ok'}
        start = sampler.mark()
        try:
            stage['fill'] = run_cycle(f'stage {name} fill')
            stage['measure'] = run_cycle(f'stage {name} measure')
        except StageAbort as exc:
            stage['status'] = f'aborted: {exc}'
            print(f'[ABORT] {exc}', flush=True)
        except Exception as exc:  # noqa: BLE001
            stage['status'] = f'failed: {exc!r}'
            print(f'[FAILED] {exc!r}', flush=True)
        window = sampler.window(start)
        stage.update(stage_stats(window))
        with runtime._resident_lock:
            stage['resident_keys'] = list(runtime._resident_cache.keys())
            stage['resident_bytes_mib'] = runtime._resident_bytes / 1048576.0
        stage['dmesg'] = dmesg_tail()
        report['stages'].append(stage)
        print(f'[stage {name}] {stage["status"]} resident {stage["resident_bytes_mib"]:.0f} MiB'
              f' vram max {stage.get("max_vram_mib")} avail min {stage.get("min_available_mib")}'
              f' swapfree min {stage.get("min_swap_free_mib")}'
              f' pswpout {stage.get("pswpout_mib")} MiB', flush=True)
        for line in stage['dmesg']:
            print(f'  dmesg: {line}', flush=True)
        Path('/tmp/opencode/resident_ladder.json').write_text(
            json.dumps(report, indent=2))
        if stage['status'] != 'ok':
            print('[LADDER STOP] stage did not complete cleanly', flush=True)
            break

status = report['stages'][-1]['status'] if report['stages'] else 'no stages'
runtime.close()
Path('/tmp/opencode/resident_ladder.json').write_text(json.dumps(report, indent=2))
print(f'LADDER DONE last_status={status}', flush=True)
