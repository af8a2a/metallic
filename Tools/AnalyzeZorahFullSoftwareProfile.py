"""Extract exact Nsight metric names/TSV rows for the frozen Full SW investigation."""
import csv
import json
import re
import sys
from pathlib import Path

METRICS = {
    'activeLanes': 'Top_Level_Triage.sm__average_thread_inst_executed_pred_on_per_inst_executed_realtime.ratio',
    'computeWarpOccupancyPercent': 'tpc__warps_active_shader_cs_queue_sync_realtime.avg.pct_of_peak_sustained_elapsed',
    'computeRegisterFilePercent': 'tpc__sm_rf_registers_allocated_shader_cs_queue_sync_realtime.avg.pct_of_peak_sustained_elapsed',
    'smThroughputPercent': 'GPUTrace.sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'l2ThroughputPercent': 'GPUTrace.lts__throughput.avg.pct_of_peak_sustained_elapsed',
    'dramSectorsPercent': 'dram__sectors.avg.pct_of_peak_sustained_elapsed',
    'l2AtomicInputActivePercent': 'lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'globalAtomOrRedWriteSectors': 'l1tex__m_l1tex2xbar_write_sectors_mem_global_op_atom_or_red_realtime.sum',
    'globalAtomOrRedWritePercent': 'l1tex__m_l1tex2xbar_write_sectors_mem_global_op_atom_or_red_realtime.avg.pct_of_peak_sustained_elapsed',
    'atomicL2HitPercent': 'Top_Level_Triage.lts__average_t_sector_hit_rate_srcunit_tex_op_atom_realtime.pct',
}


def analyze(directory):
    directory = Path(directory)
    table = directory / 'trace/BASE_UNLOCKED/GPUTRACE_REGIMES.xls'
    with table.open(encoding='utf-8-sig') as f:
        rows = list(csv.reader(f, delimiter='\t'))
    header = rows[0]
    stalls = list(dict.fromkeys(k for k in header if 'warps_issue_stalled_' in k and k.endswith('.avg.per_cycle_elapsed')))

    def value(row, name):
        # Nsight repeats export columns; preserve provenance and reject an
        # ambiguous/non-numeric cell instead of inventing a min/mean/max alias.
        cells = [row[i] for i, k in enumerate(header) if k == name]
        assert cells, f'Missing metric: {name}'
        numbers = [float(x) for x in cells]
        assert all(x == numbers[0] for x in numbers), f'Ambiguous repeated metric: {name}'
        return numbers[0]

    ranges = []
    marker = 'RenderGraphPass: VBuffer (VisibilityBufferPass)/Hybrid raster: stable cluster bins'
    for line, row in enumerate(rows[1:], 2):
        if row[0] != marker:
            continue
        samples = {k: value(row, k) for k in stalls}
        total = sum(samples.values())
        ranges.append({'tsvLine': line, 'marker': marker,
            'phase': 'early' if len(ranges) % 2 == 0 else 'late',
            'metrics': {k: value(row, v) for k, v in METRICS.items()},
            'warpStateSamplesRaw': samples,
            'warpStateSamplePercent': {k: v / total * 100 if total else None for k, v in samples.items()}})
    assert len(ranges) == 6, 'Expected early/late ranges in three frames'
    ready = json.loads((directory / 'app/ProfileReady.json').read_text(encoding='utf-8-sig'))
    return {'directory': directory.as_posix(), 'sourceTable': table.as_posix(), 'metricNames': METRICS,
        'scope': 'Exported stable-bins range includes adjacent raster work; not isolated SW. Warp-state fractions include selected/not-selected, not time fractions.',
        'snapshot': ready, 'ranges': ranges}


def pipeline_statistics(path):
    results = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        start = re.search(r'\[SW Pipeline Probe\] entry=(\S+) debugMode=(\d+)', line)
        if start:
            results.append({'entry': start[1], 'debugMode': int(start[2]), 'rawStats': {}})
        stat = re.search(r'spirv=(\w+) executable=(\S+) subgroup=(\d+) (.+?)=([^ ]+) \((.*)\)', line)
        if stat and results:
            results[-1].update(spirvHash=stat[1], executable=stat[2], subgroupSize=int(stat[3]))
            results[-1]['rawStats'][stat[4]] = {'value': stat[5], 'description': stat[6]}
    expected = {(entry, mode) for entry in (
        'streamClusterRasterLegacyMain', 'streamClusterRasterMain',
        'streamClusterRasterPlaneMain') for mode in (0, 1)}
    actual = [(item['entry'], item['debugMode']) for item in results]
    assert expected.issubset(actual), 'Missing reference SW pipeline statistics'
    assert len(actual) == len(set(actual)), 'Duplicate pipeline statistics'
    return {'source': str(path), 'pipelines': results,
        'localMemoryCaution': 'Driver reports ~64 GiB per thread, including tiny kernels. Raw values retained; do not infer spill bytes or silently mask high bits.'}


if __name__ == '__main__':
    # Arguments: output.json pipeline-statistics.log trace-directory [...]
    output, stats, *directories = sys.argv[1:]
    assert directories
    result = {'traces': [analyze(p) for p in directories], 'compiler': pipeline_statistics(stats)}
    Path(output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for trace in result['traces']:
        early = [r for r in trace['ranges'] if r['phase'] == 'early']
        print(trace['directory'], [(r['tsvLine'], r['metrics']['activeLanes']) for r in early])
