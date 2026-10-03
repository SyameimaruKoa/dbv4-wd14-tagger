"""Render saved measurements without inventing missing memory observations."""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / 'benchmarks'


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def fmt(value):
    return f'{value:,.2f}' if number(value) else '未測定'


def ratio(value, baseline):
    return f'{value / baseline:.2f}×' if number(value) and number(baseline) and baseline > 0 else '—'


def provider(row):
    value = row.get('provider_requested', row.get('provider', '不明'))
    return '+'.join(value) if isinstance(value, list) else str(value)


def measurements():
    groups = defaultdict(list)
    for path in sorted(BENCH.rglob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict):
            continue
        # Matrix summaries duplicate their individual result files.
        if path.name in ('summary.json', 'manifest.json', 'comparison-summary.json'):
            continue
        candidates = data.get('results', [data])
        if not isinstance(candidates, list):
            continue
        for index, row in enumerate(candidates):
            if not isinstance(row, dict) or 'profile' not in row:
                continue
            ms = row.get('median_ms_per_image', row.get('inference_only_median_ms', row.get('ms_per_image', row.get('single_cold_inference_ms'))))
            if not number(ms) and 'status' not in row:
                continue
            if not any(k in row for k in ('provider', 'provider_requested')):
                continue
            resources = row.get('resources') or {}
            peak = row.get('peak') or {}
            rss = row.get('peak_rss_mib', resources.get('peak_rss_mib', peak.get('rss_mib')))
            vram = row.get('peak_vram_mib', resources.get('peak_vram_mib', peak.get('vram_used_mib')))
            increment = row.get('vram_increment_mib', row.get('vram_peak_increment_mib'))
            snapshots = [row.get(k) for k in ('gpu_memory_mib_before', 'gpu_memory_mib_loaded', 'gpu_memory_mib_after_warmup')]
            snapshots = [v for v in snapshots if number(v)]
            gtt = resources.get('peak_gtt_mib', peak.get('gtt_used_mib'))
            busy = resources.get('measured_gpu_busy_percent_mean')
            ram_total, vram_total = data.get('system_ram_total_mib'), data.get('gpu_vram_total_mib')
            if path.parent.name == 'ncnn_switch_20261003':
                env = json.loads((path.parent / 'environment.json').read_text())
                ram_total = int(env['meminfo'].split('MemTotal:')[1].split()[0]) / 1024
            if path.parent.name == 'ncnn_ps4_20261004':
                vram_total = int((path.parent / 'environment.log').read_text().splitlines()[-1]) / 1048576
            source = path.relative_to(BENCH).as_posix()
            if path.parent.name == 'nvidia_linux_20261004':
                hardware = (path.parent / 'hardware.log').read_text()
                match = re.search(r'/\s*(\d+)MiB', hardware)
                if match:
                    vram_total = int(match.group(1))
            comparison = row.get('comparison') or {}
            diff = comparison.get('max_absolute_difference')
            tags = comparison.get('selected_tag_symmetric_difference')
            quality = (f'Δ {diff:.3g}; タグ差{len(tags)}' if number(diff) and isinstance(tags, list) else '未照合')
            status = row.get('status', '記録あり')
            if tags:
                quality += '（タグ不一致）'
            item = dict(source=source, index=index, profile=row['profile'], provider=provider(row),
                        batch=row.get('batch_size', '—'), precision=row.get('precision') or '—',
                        part=row.get('ncnn_part_size_mib') or (128 if 'stream128' in path.stem else 0),
                        ms=ms, rss=rss, vram=vram, increment=increment, snapshot=max(snapshots) if snapshots else None,
                        snapshot_pct=100*max(snapshots)/vram_total if snapshots and number(vram_total) and vram_total > 0 else None,
                        gtt=gtt, busy=busy, busy_peak=resources.get('measured_gpu_busy_percent_peak'), load=row.get('load_seconds', row.get('session_seconds')),
                        ram_pct=100*rss/ram_total if number(rss) and number(ram_total) and ram_total > 0 else None,
                        vram_pct=100*vram/vram_total if number(vram) and number(vram_total) and vram_total > 0 else None,
                        status=status, quality=quality, row=row)
            group = path.parent.name if path.parent != BENCH else path.stem
            groups[group].append(item)
    # Earlier DirectML records survive only in Markdown, with a different timer.
    legacy = (BENCH / 'legacy_results.md').read_text()
    for line in legacy.splitlines():
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if len(cells) != 6 or cells[0] not in ('lightweight','balanced','high','ultra','wd14_v3'):
            continue
        ms = float(cells[5].split()[0].replace(',', ''))
        vram = float(cells[3].split('／')[1].split()[0].replace(',', ''))
        increment = float(cells[4].split()[0].replace(',', ''))
        groups['directml_legacy'].append(dict(source='legacy_results.md', index=0, profile=cells[0],
            provider='DirectML', batch=4, precision='—', part=0, ms=ms, rss=None, vram=vram,
            increment=increment, snapshot=None, snapshot_pct=None, gtt=None, busy=None, busy_peak=None, load=None,
            ram_pct=None, vram_pct=100*vram/8192, status='旧平均3回', quality='未照合', row={}))
    return groups


def render(groups):
    text = ['# 全測定の速度・RAM・VRAM一覧', '', '[総合記録](../BENCHMARKS.md) / [使い方](../README.md)', '',
            '保存済みJSONから集計（2026-10-04）。単位は時間ms/枚、メモリMiB、使用率%。未測定は0ではない。', '',
            '- RSSはプロセス常駐RAM。RAM使用率はRSS÷記録された総RAMで、システム全体の使用率ではない。',
            '- VRAMピーク・増分・スナップショットは別指標。GPU全体の値には画面表示や他プロセスも含む。スナップショット最大は推論中の連続監視ピークではない。',
            '- VRAM使用率はピーク÷記録されたVRAM総容量。Intel/AMDの共有メモリ、GTT、Switchの統合RAMは独立VRAMと合算しない。総容量未記録では割合を算出しない。',
            '- Linux AEROのGPUメモリ時点使用率は同じ測定組のhardware.logにある8192MiBを分母にする。旧DirectMLはlegacy_results.mdの8GiBを使用。',
            '- GPU稼働率は測定中の平均・最大で、VRAM使用率とは異なる。旧DirectMLは前処理込み3回の平均で、中央値と混ぜない。Windowsの最新測定ではRSS監視がなく、未測定として表示する。',
            '- 2026-09-23はwarmup3・20回×3セットのsession.run時間。最新ultraは画像が異なり、warmup/回数も各JSONに従う。ncnnのbatch4は順次4枚。PS4/Switchは測定1回。',
            '- 初期化時間は記録方式によってセッション構築のみ／取得・変換・起動検証込みが異なる。別日・別OS・別モデルの値から条件を揃えた性能差は断定しない。',
            '- AMD診断の精度不一致や失敗も残す。「記録あり」は速度が保存された意味で、精度合格の意味ではない。ΔはCPU参照との最大絶対差。全確率の合格判定は各comparison-summary.jsonを参照。', '',
            '## 測定組', '']
    for group in groups:
        text.append(f'- [{group}](#{group})')
    for group, rows in groups.items():
        text += ['', f'## {group}', '', '### 速度と精度', '',
                 '| 条件・出典 | モデル | batch | 実行方式 | 精度・分割MiB | ms/枚 | 初期化秒 | 状態・精度 |',
                 '|---|---|---:|---|---|---:|---:|---|']
        for r in rows:
            label = Path(r['source']).stem
            timing = '（cold 1回）' if 'single_cold_inference_ms' in r['row'] else ''
            text.append('| ' + ' | '.join([f"[{label}]({r['source']})", str(r['profile']), str(r['batch']), r['provider'],
                f"{r['precision']} / {r['part']}", fmt(r['ms']) + timing, fmt(r['load']),
                f"{r['status']}; {r['quality']}"]) + ' |')
        text += ['', '### メモリとGPU稼働率', '',
                 '| 条件・モデル・batch | RSS MiB | RAM % | VRAMピーク MiB | VRAM % | VRAM増分 MiB | GPUメモリ時点最大 MiB (%) | GTTピーク MiB | GPU稼働 平均 / 最大 % |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            label = f"{Path(r['source']).stem} / {r['profile']} / b{r['batch']}"
            text.append('| ' + ' | '.join([f"[{label}]({r['source']})",
                *[fmt(r[k]) for k in ('rss','ram_pct','vram','vram_pct','increment')],
                fmt(r['snapshot']) + (f" ({fmt(r['snapshot_pct'])}%)" if number(r['snapshot_pct']) else ''),
                fmt(r['gtt']), fmt(r['busy']) + ' / ' + fmt(r['busy_peak'])]) + ' |')
    text += ['', '## 同じ測定組でのモデル差', '',
             'balancedを基準とした対象モデル÷balanced。時間比が1より大きければ遅く、メモリ比が1より大きければ多く使用する。実行方式・batchが同じ保存値だけを比較する。VRAMは同じ指標同士（ピーク優先、なければ増分）。モデルごとに画像サイズ・処理が異なるため品質の比較ではない。最新ultraと9月の他モデルは測定日・入力・精度設定が異なり、直接の倍率表を作らない。', '',
             '| 測定組 | 実行方式 | batch | 対象モデル | 時間比 | RSS比 | VRAM比・指標 | 出典 |', '|---|---|---:|---|---:|---:|---|---|']
    for group, rows in groups.items():
        lookup={(r['profile'],r['provider'],r['batch']):r for r in rows if number(r['ms']) and r['status'] in ('ok','記録あり','旧平均3回')}
        for (profile, ep, batch), r in lookup.items():
            base=lookup.get(('balanced',ep,batch))
            if profile=='balanced' or not base:
                continue
            field='vram' if number(r['vram']) and number(base['vram']) else 'increment'
            text.append(f"| {group} | {ep} | {batch} | {profile} | {ratio(r['ms'],base['ms'])} | {ratio(r['rss'],base['rss'])} | {ratio(r[field],base[field])} / {'ピーク' if field=='vram' else '増分'} | [対象]({r['source']}) / [balanced]({base['source']}) |")
    text += ['', '## 補助観測と旧記録', '',
             '- Switchの分割実行について、[外部メモリ監視](ncnn_switch_20261003/ultra-stream128-memory-watch.json)も保存。下表は全サンプルから集計し、プロセス内監視と測定範囲が異なる。', '',
             '| 観測 | 終了コード | RSS最大 MiB | 空きRAM最小 MiB | 空きswap最小 MiB | GPU稼働最大 % |', '|---|---:|---:|---:|---:|---:|']
    for p in sorted((BENCH/'ncnn_switch_20261003').glob('*memory-watch.json')):
        d=json.loads(p.read_text()); samples=d.get('samples',[])
        def extreme(key, fn):
            values=[s[key] for s in samples if number(s.get(key))]
            return fn(values) if values else None
        busy=extreme('gpu_load_per_mille',max)
        text.append(f"| [{p.stem}]({p.relative_to(BENCH).as_posix()}) | {d.get('exit_code','—')} | {fmt(extreme('rss_mib',max))} | {fmt(extreme('MemAvailable',min))} | {fmt(extreme('SwapFree',min))} | {fmt(busy/10 if number(busy) else None)} |")
    text += ['', '[9月の全100条件・ばらつき・p95](matrix_details_20260923.md)、[旧測定の速度・VRAM記録](legacy_results.md)、[AMD 4GiBの説明](amd_barcelo_4gb.md)も参照。古いMarkdownだけの記録は元の条件とともに保持する。', '']
    return '\n'.join(text)


if __name__ == '__main__':
    groups=measurements()
    (BENCH/'all_measurements.md').write_text(render(groups),encoding='utf-8')
    print(f'{sum(map(len,groups.values()))} records, {len(groups)} groups')
