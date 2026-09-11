import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from terragpu import datasets


@pytest.fixture
def sample(monkeypatch):
    content = b'fixture raster bytes'
    monkeypatch.setattr(datasets, 'SAMPLE', {**datasets.SAMPLE, 'size_bytes': len(content),
                        'sha256': hashlib.sha256(content).hexdigest()})
    monkeypatch.setattr(datasets, 'urlopen', lambda *a, **k: io.BytesIO(content))
    return content


def test_sample_download_and_offline_cache(tmp_path, monkeypatch, sample):
    path = datasets.fetch_sample(tmp_path)
    assert path.read_bytes() == sample
    def offline(*a, **k):
        raise AssertionError('Verified cache should not use the network')
    monkeypatch.setattr(datasets, 'urlopen', offline)
    assert datasets.fetch_sample(tmp_path) == path
    assert json.loads((tmp_path/'sample-manifest.json').read_text())['sha256'] == datasets.sha256(path)


@pytest.mark.parametrize('content', [b'x', b'x' * 100])
def test_sample_incomplete_or_oversized_download(tmp_path, monkeypatch, sample, content):
    monkeypatch.setattr(datasets, 'urlopen', lambda *a, **k: io.BytesIO(content))
    with pytest.raises(ValueError):
        datasets.fetch_sample(tmp_path)
    assert not (tmp_path / datasets.SAMPLE['filename']).exists()
    assert not list(tmp_path.glob('.download-*'))


QUERY = dict(short_name='HLSL30', version='2.0', bbox=(-77.1, 38.8, -76.9, 39.0),
             start='2024-06-01', end='2024-06-05')


@pytest.fixture
def nasa(monkeypatch):
    client = SimpleNamespace(__version__='test')
    def search_data(**kw):
        assert isinstance(kw['bounding_box'], tuple)
        assert isinstance(kw['temporal'], tuple)
        return [{'meta': {'concept-id': 'G-example'}, 'umm': {'GranuleUR': 'example'}}]
    client.search_data = search_data
    client.login = lambda **kw: SimpleNamespace(authenticated=True)
    def download(granules, local_path, **kw):
        path = Path(local_path) / 'B04.tif'
        path.write_bytes(b'native file')
        return [path]
    client.download = download
    monkeypatch.setattr(datasets, '_earthaccess', lambda: client)
    return client


def test_nasa_search_never_logs_in(nasa):
    def forbidden(**kw):
        raise AssertionError('Search must not authenticate')
    nasa.login = forbidden
    assert datasets.nasa_data(**QUERY, search_only=True)['granules'][0]['concept_id'] == 'G-example'


def test_nasa_cache_download_and_tampering(tmp_path, nasa, monkeypatch):
    output = tmp_path / 'nasa'
    result = datasets.nasa_data(**QUERY, output=output)
    assert result['files'][0]['sha256'] == datasets.sha256(output/'B04.tif')
    def offline():
        raise AssertionError('Completed cache must work without earthaccess/network')
    monkeypatch.setattr(datasets, '_earthaccess', offline)
    assert datasets.nasa_data(**QUERY, output=output) == result
    (output/'B04.tif').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='corrupt'):
        datasets.nasa_data(**QUERY, output=output)


def test_nasa_failed_download_not_published(tmp_path, nasa):
    nasa.download = lambda *a, **kw: []
    output = tmp_path / 'nasa'
    with pytest.raises(RuntimeError, match='no downloaded'):
        datasets.nasa_data(**QUERY, output=output)
    assert not output.exists()
    assert not list(tmp_path.glob('.nasa-download-*'))


def test_nasa_missing_auth(tmp_path, nasa):
    nasa.login = lambda **kw: SimpleNamespace(authenticated=False)
    with pytest.raises(RuntimeError, match='authentication'):
        datasets.nasa_data(**QUERY, output=tmp_path/'nasa')


@pytest.mark.parametrize('overrides', [{'limit': 0}, {'limit': 11}, {'bbox': (1, 1, 0, 0)}, {'end': '2020-01-01'}])
def test_nasa_query_validation(overrides):
    with pytest.raises(ValueError):
        datasets.nasa_data(**{**QUERY, **overrides}, search_only=True)
