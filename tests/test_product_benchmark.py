import logging
from types import SimpleNamespace

import pytest
pytest.importorskip('scipy')
from scripts import benchmark_products as bench


def test_missing_token_fails_before_network(tmp_path, monkeypatch):
    monkeypatch.delenv('EARTHDATA_TOKEN', raising=False)
    monkeypatch.setattr(bench, 'nasa_data', lambda **kw: pytest.fail('Network must not be called'))
    with pytest.raises(RuntimeError, match='EARTHDATA_TOKEN'):
        bench.prepare(tmp_path)


def test_download_diagnostics_cannot_leak_token(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('EARTHDATA_TOKEN', 'secret-test-token')
    def download(**kw):
        print('secret-test-token')
        logging.error('secret-test-token')
        raise RuntimeError('secret-test-token')
    monkeypatch.setattr(bench, 'nasa_data', download)
    before = logging.root.manager.disable
    with pytest.raises(RuntimeError) as error:
        bench.prepare(tmp_path)
    captured = capsys.readouterr()
    assert 'secret-test-token' not in captured.out + captured.err + str(error.value)
    assert logging.root.manager.disable == before


def test_each_output_validated_and_backend_order_alternates(tmp_path, monkeypatch):
    calls=[]; validated=[]
    module=SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda:None)))
    monkeypatch.setattr(bench,'array_module',lambda backend:module)
    def operation(source, target, backend):
        calls.append(backend); target.write_bytes(b'output')
    def validator(source, target):
        validated.append(target.read_bytes())
        return dict(valid_pixels=12,shape=[3,4],max_absolute_error=0.)
    records=bench.measure('test',None,operation,validator,'.tif',('numpy','cupy'),2,1,tmp_path)
    assert calls==['numpy','cupy','cupy','numpy']*2
    assert len(validated)==8
    assert all(len(r['samples_seconds'])==2 and r['correctness_passed'] for r in records)
    assert not list(tmp_path.iterdir())


def test_bad_output_aborts_before_reporting_timings(tmp_path, monkeypatch):
    monkeypatch.setattr(bench,'array_module',lambda backend:None)
    def operation(source,target,backend):target.write_bytes(b'bad')
    def validator(source,target):raise AssertionError('wrong science')
    with pytest.raises(AssertionError,match='wrong science'):
        bench.measure('test',None,operation,validator,'.tif',('numpy',),1,0,tmp_path)
    assert not list(tmp_path.iterdir())


def test_failure_preserves_stage_and_http_status_without_secrets(tmp_path,monkeypatch,capsys):
    monkeypatch.setenv('EARTHDATA_TOKEN','secret-test-token')
    def download(**kw):
        kw['progress']('NASA file download')
        error=RuntimeError('https://provider/?token=secret-test-token')
        error.response=SimpleNamespace(status_code=403)
        raise error
    monkeypatch.setattr(bench,'nasa_data',download)
    with pytest.raises(RuntimeError,match='NASA file download: HTTP status 403') as caught:
        bench.prepare(tmp_path)
    output=capsys.readouterr()
    assert 'secret-test-token' not in str(caught.value)+output.out+output.err
    assert 'https://' not in str(caught.value)


@pytest.mark.parametrize('error,expected',[
    (ValueError('Cached query differs; choose a new output directory'),'different scene/query'),
    (FileExistsError('Use a new output directory or a completed TerraGPU cache'),'without a manifest'),
    (ModuleNotFoundError('secret-path'),'data extra'),
    (PermissionError('secret-path'),'permission denied'),
    (OSError(28,'secret-path'),'error number 28'),
])
def test_safe_failure_categories(error,expected):
    assert expected in bench.safe_failure(error)
    assert 'secret-path' not in bench.safe_failure(error)
