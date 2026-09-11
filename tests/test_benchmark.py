import pytest
from terragpu.benchmark import run


@pytest.mark.parametrize('backend', ['numpy', 'dask'])
def test_benchmark_schema_and_correctness(backend):
    result = run(backend=backend, size=16, chunk=7, repeat=2, warmup=0)
    assert result['correctness_passed']
    assert len(result['samples_seconds']) == 2
    assert all(t > 0 for t in result['samples_seconds'])
    assert result['max_absolute_error'] < 1e-6


def test_invalid_size():
    with pytest.raises(ValueError):
        run(size=0)


def test_io_benchmark_correctness():
    from terragpu.benchmark_io import run as run_io
    result = run_io(size=19, tile_size=8, repeat=1)
    assert result['correctness_passed']
    assert all(len(samples) == 1 for samples in result['samples_seconds'].values())
