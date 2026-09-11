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
