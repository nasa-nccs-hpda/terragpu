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
