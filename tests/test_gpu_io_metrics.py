import copy
import csv
import json

import pytest

from terragpu.gpu_io_metrics import audit, export


@pytest.fixture
def report():
    result = dict(schema_version=1, status='complete', git_dirty=False, git_commit='fixture',
                  source_sha256='fixture', repeat=2, warmup=1, tiles=[16], focal_sizes=[3], threads={},
                  probes={'numpy': {}, 'cupy': {}}, skipped_modes={}, records=[],
                  preparation=[dict(tile_size=16, conversion_seconds=5, geotiff_export_seconds=6)])
    for mode, totals in [('numpy', [4., 8.]), ('cupy', [1., 3.])]:
        result['records'].append(dict(tile_size=16, mode=mode, correctness_passed=True,
            workload='spatial', shape=[1, 32, 32], scope='cache pipeline',
            logical_input_bytes=4096, physical_input_bytes=8192,
            samples_seconds=totals, median_seconds=sum(totals)/2,
            phase_samples=[dict(total_seconds=t, read_seconds=t/4, compute_seconds=t/2,
                                write_seconds=t/8, finalization_and_loop_seconds=t/8) for t in totals]))
    return result


def test_exports_recomputed_metrics_without_inventing_gds(tmp_path, report):
    report['gds_verified'] = True  # An input flag alone is not evidence.
    source=tmp_path/'gpu-io.json';source.write_text(json.dumps(report))
    dest=export(source,tmp_path/'audit')
    rows=list(csv.DictReader((dest/'summary.csv').open()))
    assert [float(row['median_seconds']) for row in rows]==[6.,2.]
    assert len(list(csv.DictReader((dest/'samples.csv').open())))==4
    ratio=list(csv.DictReader((dest/'same-tile-ratios.csv').open()))
    assert float(ratio[0]['cpu_over_mode'])==3.
    provenance=json.loads((dest/'provenance.json').read_text())
    assert provenance['gds_verified'] is False
    assert 'end_to_end_seconds' not in provenance
    with pytest.raises(FileExistsError):export(source,dest)


@pytest.mark.parametrize('fault', ['missing_record', 'duplicate', 'nan', 'boolean', 'phases',
                                    'stored_median', 'scope', 'shape', 'repeat', 'unclean',
                                    'unvalidated', 'preparation', 'status'])
def test_rejects_inconsistent_reports_before_output(tmp_path, report, fault):
    r=report['records'][0]
    if fault=='missing_record':report['records'].pop()
    elif fault=='duplicate':report['records'].append(copy.deepcopy(r))
    elif fault=='nan':r['samples_seconds'][0]=float('nan')
    elif fault=='boolean':r['samples_seconds'][0]=True
    elif fault=='phases':r['phase_samples'][0]['write_seconds']=500.
    elif fault=='stored_median':r['median_seconds']=100.
    elif fault=='scope':r['scope']='resident computation only'
    elif fault=='shape':r['shape']=[1,64,64]
    elif fault=='repeat':r['samples_seconds'].pop()
    elif fault=='unclean':report['git_dirty']=True
    elif fault=='unvalidated':r['correctness_passed']=False
    elif fault=='preparation':report['preparation']=[]
    elif fault=='status':report['status']='running'
    source=tmp_path/'gpu-io.json';source.write_text(json.dumps(report))
    with pytest.raises(ValueError):export(source,tmp_path/'audit')
    assert not (tmp_path/'audit').exists()


def test_explicit_skipped_mode_and_partial_file(tmp_path, report):
    report['status']='complete_available_modes'
    report['skipped_modes']={'kvikio-cufile': {'error_type': 'RuntimeError'}}
    assert len(audit(report)[0])==2
    source=tmp_path/'gpu-io.json';source.write_text(json.dumps(report))
    source.with_suffix('.partial.json').write_text('{}')
    with pytest.raises(ValueError,match='partial'):export(source,tmp_path/'audit')
