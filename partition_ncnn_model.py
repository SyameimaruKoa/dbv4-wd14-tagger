"""Partition an FP32 ncnn graph without changing layers or weight bytes.

Cuts are made only where one tensor connects the two graph sections. Native
layer loaders determine weight boundaries; unsupported weight formats fail
before a cache is published. The source model is never modified.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile


def _graph(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines[0] != '7767517':
        raise ValueError('Expected ncnn text param')
    rows = []
    for line in lines[2:]:
        tokens = line.split()
        ni, no = map(int, tokens[2:4])
        rows.append((line, tokens[4:4 + ni], tokens[4 + ni:4 + ni + no]))
    if len(rows) != int(lines[1].split()[0]):
        raise ValueError('ncnn layer count mismatch')
    return rows


def _param(rows, input_name=None):
    rows = list(rows)
    if input_name is not None:
        rows.insert(0, (f'Input partition_input 0 1 {input_name}', [], [input_name]))
    blobs = set(blob for _, ins, outs in rows for blob in ins + outs)
    return '7767517\n' + f'{len(rows)} {len(blobs)}\n' + '\n'.join(r[0] for r in rows) + '\n'


def partition_model(prefix, size_mib):
    import ncnn

    if size_mib <= 0:
        raise ValueError('Partition size must be positive')
    prefix = Path(prefix)
    param, binary = prefix.with_suffix('.ncnn.param'), prefix.with_suffix('.ncnn.bin')
    signature = {'version': 1, 'size_mib': size_mib,
                 'sources': {str(p.resolve()): [p.stat().st_size, p.stat().st_mtime_ns]
                             for p in (param, binary)}}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    cache = prefix.parent / (prefix.name + '.parts-' + key)
    manifest_path = cache / 'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest['signature'] == signature and all(
            (cache / p['bin']).stat().st_size == p['bytes'] and
            (cache / p['param']).is_file() for p in manifest['parts']):
            return cache, manifest['parts']
    rows = _graph(param.read_text())
    consumers = Counter(blob for _, ins, _ in rows for blob in ins)
    outputs = [blob for _, _, outs in rows for blob in outs if not consumers[blob]]
    if len(outputs) != 1 or rows[0][0].split()[0] != 'Input' or len(rows[0][2]) != 1:
        raise ValueError('Partitioning requires one input and one output')

    with binary.open('rb') as source:
        class CountingBin(ncnn.ModelBin):
            def load(self, w, kind):
                if kind == 0:
                    tag = source.read(4)
                    if tag != struct.pack('<I', 0):
                        raise ValueError(f'Only FP32 weights supported at offset {source.tell() - 4}')
                elif kind != 1:
                    raise ValueError(f'Unsupported ncnn weight type {kind}')
                source.seek(int(w) * 4, 1)
                if source.tell() > binary.stat().st_size:
                    raise ValueError('Weight range exceeds source file')
                mat = ncnn.Mat(int(w))
                mat.fill(0.0)
                return mat

        bounds = [0]
        for line, ins, outs in rows:
            # A fresh single-layer net releases dummy weights immediately.
            inputs = [(f'Input offset_input_{j} 0 1 {b}', [], [b])
                      for j, b in enumerate(dict.fromkeys(ins))]
            net = ncnn.Net()
            layer = None
            try:
                if net.load_param_mem(_param(inputs + [(line, ins, outs)])) != 0:
                    raise ValueError('Cannot inspect ncnn layer: ' + line)
                layer = net.layers()[-1]
                if layer.load_model(CountingBin()) != 0:
                    raise ValueError('Cannot determine weights for: ' + line)
            finally:
                layer = None
                net.clear()
            bounds.append(source.tell())
        if source.tell() != binary.stat().st_size:
            raise ValueError(f'Weight accounting mismatch: {source.tell()} != {binary.stat().st_size}')

        parts, live = [], set()
        start, start_byte, input_name = 0, 0, rows[0][2][0]
        for i, (_, ins, outs) in enumerate(rows):
            for blob in ins:
                consumers[blob] -= 1
                if consumers[blob] == 0:
                    live.discard(blob)
            live.update(blob for blob in outs if consumers[blob] or blob in outputs)
            last = i == len(rows) - 1
            if last or (len(live) == 1 and bounds[i + 1] - start_byte >= size_mib * 1048576):
                if len(live) != 1:
                    raise ValueError('Partition has multiple outputs')
                output_name = next(iter(live))
                parts.append({'start': start, 'end': i + 1, 'offset': start_byte,
                              'bytes': bounds[i + 1] - start_byte,
                              'input': input_name, 'output': output_name,
                              'param': f'part{len(parts):03d}.ncnn.param',
                              'bin': f'part{len(parts):03d}.ncnn.bin'})
                start, start_byte, input_name = i + 1, bounds[i + 1], output_name
        temporary = Path(tempfile.mkdtemp(prefix='.ncnn-parts-', dir=prefix.parent))
        try:
            for part in parts:
                text = _param(rows[part['start']:part['end']],
                              part['input'] if part['start'] else None)
                (temporary / part['param']).write_text(text)
                source.seek(part['offset'])
                remaining = part['bytes']
                with (temporary / part['bin']).open('wb') as dest:
                    while remaining:
                        data = source.read(min(8 * 1048576, remaining))
                        if not data:
                            raise ValueError('Unexpected end of ncnn weights')
                        dest.write(data)
                        remaining -= len(data)
            (temporary / 'manifest.json').write_text(json.dumps(
                {'signature': signature, 'parts': parts}, indent=2) + '\n')
            if cache.exists():
                raise RuntimeError(f'Invalid existing partition cache: {cache}')
            temporary.rename(cache)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return cache, parts
