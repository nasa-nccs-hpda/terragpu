"""Audit one completed GPU I/O run and export plot-ready, same-run metrics.

No cross-run pooling, reconstructed end-to-end timings, or inferred GDS claims.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

PHASES = ('read_seconds', 'compute_seconds', 'write_seconds',
          'finalization_and_loop_seconds')
MODES = {'numpy', 'cupy', 'kvikio-compat', 'kvikio-cufile'}


def number(value, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError('Invalid timing value')
    return value


def audit(report):
    if report.get('schema_version') != 1 or report.get('status') not in ('complete', 'complete_available_modes'):
        raise ValueError('Expected a completed version 1 GPU I/O report')
    if report.get('git_dirty') is not False:
        raise ValueError('Publication export requires a clean source revision')
    repeat = report['repeat']
    if type(repeat) is not int or repeat < 1:
        raise ValueError('Invalid repetition count')
    tiles = report['tiles']
    if not tiles or any(type(t) is not int or t < 1 for t in tiles) or len(set(tiles)) != len(tiles):
        raise ValueError('Invalid tile matrix')
    modes = set(report['probes'])
    skipped = set(report['skipped_modes'])
    if not modes or not modes <= MODES or not skipped <= MODES or modes & skipped:
        raise ValueError('Invalid probe/mode matrix')
    if bool(skipped) != (report['status'] == 'complete_available_modes'):
        raise ValueError('Completion status disagrees with skipped modes')
    records = report['records']
    seen = {}
    summary, samples = [], []
    for record in records:
        key = (record['tile_size'], record['mode'])
        if key in seen or record.get('correctness_passed') is not True:
            raise ValueError('Duplicate or unvalidated benchmark record')
        seen[key] = record
        values, phases = record['samples_seconds'], record['phase_samples']
        if len(values) != repeat or len(phases) != repeat:
            raise ValueError('Incomplete timing repetitions')
        for index, (total, phase) in enumerate(zip(values, phases)):
            number(total, positive=True)
            number(phase['total_seconds'], positive=True)
            if not math.isclose(total, phase['total_seconds'], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError('Total and phase sample disagree')
            for name in PHASES:
                number(phase[name])
            if not math.isclose(total, sum(phase[name] for name in PHASES), rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError('Timing phases do not sum to total')
            samples.append(dict(tile_size=key[0], mode=key[1], repetition=index,
                                total_seconds=total, **{name: phase[name] for name in PHASES}))
        median = statistics.median(values)
        if not math.isclose(median, number(record['median_seconds'], positive=True), rel_tol=1e-9):
            raise ValueError('Stored median disagrees with raw samples')
        summary.append(dict(tile_size=key[0], mode=key[1], repetitions=repeat,
                            median_seconds=median, min_seconds=min(values), max_seconds=max(values),
                            **{name.replace('_seconds', '_median_seconds'):
                               statistics.median(p[name] for p in phases) for name in PHASES},
                            scope=record['scope']))
    if set(seen) != {(tile, mode) for tile in tiles for mode in modes}:
        raise ValueError('Incomplete tile/backend matrix')
    ratios = []
    for row in summary:
        cpu = seen.get((row['tile_size'], 'numpy'))
        if cpu is None or row['mode'] == 'numpy':
            continue
        gpu = seen[(row['tile_size'], row['mode'])]
        for field in ('workload', 'shape', 'scope', 'logical_input_bytes', 'physical_input_bytes'):
            if field not in cpu or field not in gpu or cpu[field] != gpu[field]:
                raise ValueError('Unmatched CPU/backend comparison: ' + field)
        ratios.append(dict(tile_size=row['tile_size'], mode=row['mode'],
                           cpu_over_mode=statistics.median(cpu['samples_seconds']) / row['median_seconds']))
    preparation = report['preparation']
    if len(preparation) != len(tiles) or {p['tile_size'] for p in preparation} != set(tiles):
        raise ValueError('Incomplete preparation matrix')
    costs = []
    for prep in preparation:
        costs.append(dict(tile_size=prep['tile_size'],
                          conversion_seconds=number(prep['conversion_seconds']),
                          geotiff_export_seconds=number(prep['geotiff_export_seconds'])))
    return summary, samples, ratios, costs


def export(source, destination):
    source, destination = Path(source), Path(destination)
    if source.with_suffix('.partial.json').exists():
        raise ValueError('Incomplete run: partial report remains')
    report = json.loads(source.read_text())
    summary, samples, ratios, costs = audit(report)
    # Validate everything before creating output; never overwrite an earlier audit.
    destination.mkdir(parents=True, exist_ok=False)
    datasets = [('summary.csv', summary), ('samples.csv', samples),
                ('preparation.csv', costs)]
    if ratios:
        datasets.append(('same-tile-ratios.csv', ratios))
    for name, rows in datasets:
        with (destination / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    provenance = {key: report[key] for key in ('git_commit', 'source_sha256', 'repeat', 'warmup', 'tiles', 'focal_sizes', 'threads')}
    provenance.update(
        skipped_modes=sorted(report['skipped_modes']),
        gds_verified=False,
        gds_note='This exporter does not verify transfer telemetry, regardless of input flags.',
        native_preparation_seconds=report.get('native_preparation_seconds'),
        limitations=['Same-run, same-tile ratios only; CPU thread settings recorded, not tuned by this exporter.',
                     'Cold and warmup samples excluded; repetitions are within one job, not independent jobs.',
                     'Phase medians do not necessarily sum to median total.',
                     'Preparation/export are separate measurements, not directly timed end-to-end results.',
                     'Filesystem caches uncontrolled; writes have no fsync durability barrier.'])
    (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    export(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
