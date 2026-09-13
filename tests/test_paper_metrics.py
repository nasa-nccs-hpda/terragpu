import json
from pathlib import Path

import pytest
pytest.importorskip('scipy')
from terragpu.paper_benchmark import _summary
from scripts.export_benchmark_metrics import export


def test_plot_ready_metrics(tmp_path):
    record=lambda backend,samples:dict(workload='test',backend=backend,samples_seconds=samples,correctness_passed=True)
    (tmp_path/'suite.json').write_text(json.dumps({'records':[record('numpy',[2,4,6]),record('cupy',[1,2,3])]}))
    export(tmp_path)
    assert json.loads((tmp_path/'speedups.json').read_text())==[{'workload':'test','cpu_over_gpu':2.0}]
    assert len((tmp_path/'samples.csv').read_text().splitlines())==7
    stats=_summary([2,4,6])
    assert stats['median_seconds']==4
    assert stats['median_bootstrap_ci95_seconds'][0]<=4<=stats['median_bootstrap_ci95_seconds'][1]


@pytest.mark.parametrize('bad', [[], [0.], [-1.], [float('nan')], [float('inf')], [True]])
def test_invalid_timings_never_export(tmp_path, bad):
    (tmp_path/'suite.json').write_text(json.dumps({'records':[dict(workload='bad',backend='numpy',samples_seconds=bad,correctness_passed=True)]}))
    with pytest.raises(ValueError,match='Invalid timing'):
        export(tmp_path)
    assert not (tmp_path/'summary.csv').exists()


def test_complete_products_export_and_mismatch_rejection(tmp_path):
    def record(name,backend):
        return dict(workload=name,backend=backend,samples_seconds=[2 if backend=='numpy' else 1],correctness_passed=True,shape=[4,4])
    (tmp_path/'suite.json').write_text(json.dumps({'git_commit':'same','records':[record('resident',b) for b in ('numpy','cupy')]}))
    product=dict(git_commit='same',requested_backends=['numpy','cupy'],records=[record(w,b) for w in ('hls_ndvi_io','pace_spectral_mean_io','viirs_spectral_mean_io') for b in ('numpy','cupy')])
    (tmp_path/'products.json').write_text(json.dumps(product))
    export(tmp_path,require_products=True)
    assert len(json.loads((tmp_path/'speedups.json').read_text()))==4
    product['records'][-1]['shape']=[8,8]
    (tmp_path/'products.json').write_text(json.dumps(product))
    with pytest.raises(ValueError,match='shape'):
        export(tmp_path,require_products=True)
    product['records'].pop()
    (tmp_path/'products.json').write_text(json.dumps(product))
    with pytest.raises(ValueError,match='Incomplete'):
        export(tmp_path,require_products=True)


def test_io_export_requires_matched_devices_inputs_and_tiles(tmp_path):
    backends=('numpy','cupy')
    (tmp_path/'suite.json').write_text(json.dumps({'git_commit':'same','records':[
        dict(workload='resident',backend=b,samples_seconds=[1.],correctness_passed=True) for b in backends]}))
    def write(device,**changes):
        report=dict(git_commit='same',device=device,samples_seconds={'streaming':[2.,4.],'dask':[4.,8.]},
                    correctness_passed=True,shape=[2,8,8],scope='read/compute/write',tile_size=4,
                    input_array_sha256=['same-input'])
        report.update(changes)
        (tmp_path/f'io-{device}.json').write_text(json.dumps(report))
    write('numpy')
    with pytest.raises(ValueError,match='Incomplete I/O backend'):
        export(tmp_path,require_io=True)
    write('cupy',samples_seconds={'streaming':[1.,2.],'dask':[2.,4.]})
    export(tmp_path,require_io=True)
    ratios={r['workload']:r['cpu_over_gpu'] for r in json.loads((tmp_path/'speedups.json').read_text())}
    assert ratios['synthetic_geotiff_ndvi_streaming']==2.
    assert ratios['synthetic_geotiff_ndvi_dask']==2.
    write('cupy',input_array_sha256=['different-input'])
    with pytest.raises(ValueError,match='input_array_sha256'):
        export(tmp_path,require_io=True)
    write('cupy',tile_size=8)
    with pytest.raises(ValueError,match='tile_size'):
        export(tmp_path,require_io=True)
    write('cupy',git_commit='different')
    with pytest.raises(ValueError,match='source revisions'):
        export(tmp_path,require_io=True)
