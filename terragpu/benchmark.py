"""Reproducible synthetic NDVI benchmark; no claimed real-scene speedups."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import xarray as xr
from terragpu.engine import array_module
from terragpu.indices.wv_indices import ndvi


def _git(*args):
    try:
        return subprocess.check_output(['git', *args], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(backend='numpy', size=2048, chunk=1024, repeat=10, warmup=2, seed=42):
    if min(size, chunk, repeat) < 1 or warmup < 0:
        raise ValueError('size, chunk, repeat must be positive; warmup must be nonnegative')
    if backend not in {'numpy', 'cupy', 'dask', 'dask-cupy'}:
        raise ValueError('Unknown backend')
    gpu = 'cupy' in backend
    xp = array_module('cupy' if gpu else 'numpy')
    def sync():
        if gpu:
            xp.cuda.runtime.deviceSynchronize()
    rng = np.random.default_rng(seed)
    host = rng.uniform(0.01, 1, (2, size, size)).astype('float32')
    # Independent float64 reference, outside timing.
    red, nir = host.astype('float64')
    expected = (nir - red) / (nir + red)
    sync()
    start = time.perf_counter()
    resident = xp.asarray(host)
    sync()
    transfer_seconds = time.perf_counter() - start
    if backend.startswith('dask'):
        from .io import _require_dask
        _require_dask()
        import dask.array as da
        data = da.from_array(resident, chunks=(2, chunk, chunk), asarray=False)
    else:
        data = resident
    raster = xr.DataArray(data, dims=('band', 'y', 'x'), attrs={'band_names': ['red', 'nir1']})
    start = time.perf_counter()
    lazy_result = ndvi(raster) if backend.startswith('dask') else None
    graph_seconds = time.perf_counter() - start if lazy_result is not None else 0.0

    def operation():
        if lazy_result is not None:
            # One GPU / one execution thread: predictable synchronization and residency.
            return lazy_result.compute(scheduler='single-threaded').data
        return ndvi(raster).data

    sync()
    start = time.perf_counter()
    result = operation()
    sync()
    cold_seconds = time.perf_counter() - start
    for _ in range(warmup):
        result = operation()
        sync()
    samples = []
    for _ in range(repeat):
        sync()
        start = time.perf_counter()
        result = operation()
        sync()
        samples.append(time.perf_counter() - start)
    start = time.perf_counter()
    actual = xp.asnumpy(result) if gpu else result
    sync()
    output_transfer_seconds = time.perf_counter() - start
    np.testing.assert_allclose(actual[0], expected, rtol=1e-5, atol=1e-6)
    packages = {}
    for package in ('terragpu', 'numpy', 'xarray', 'dask', 'distributed', 'rasterio', 'rioxarray'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    hardware = {'platform': platform.platform(), 'machine': platform.machine(), 'cpu_count': os.cpu_count()}
    if gpu:
        props = xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)
        hardware.update(gpu_name=props['name'].decode() if isinstance(props['name'], bytes) else props['name'],
                        gpu_memory_bytes=props['totalGlobalMem'], cuda_runtime=xp.cuda.runtime.runtimeGetVersion(),
                        cuda_driver=xp.cuda.runtime.driverGetVersion(), cupy=xp.__version__)
    median = statistics.median(samples)
    return {
        'schema_version': 1, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git('rev-parse', 'HEAD'), 'git_dirty': bool(_git('status', '--porcelain')),
        'workload': 'synthetic_ndvi', 'backend': backend, 'shape': list(host.shape), 'dtype': str(host.dtype),
        'chunk': chunk if backend.startswith('dask') else None, 'seed': seed, 'warmup': warmup, 'repeat': repeat,
        'timing_scope': 'resident inputs; includes xarray arithmetic (eager) or prebuilt graph execution (dask); excludes I/O and transfers',
        'scheduler': 'single-threaded' if backend.startswith('dask') else None,
        'input_transfer_seconds': transfer_seconds, 'output_transfer_seconds': output_transfer_seconds,
        'graph_build_seconds': graph_seconds, 'cold_seconds': cold_seconds,
        'samples_seconds': samples, 'median_seconds': median,
        'pixels_per_second': size * size / median,
        'max_absolute_error': float(np.max(np.abs(actual[0] - expected))),
        'correctness_passed': True, 'hardware': hardware, 'python': platform.python_version(), 'packages': packages,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['numpy', 'cupy', 'dask', 'dask-cupy'], default='numpy')
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--chunk', type=int, default=1024)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = vars(parser.parse_args())
    output = args.pop('output')
    result = json.dumps(run(**args), indent=2, allow_nan=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result + '\n')
    print(result)


if __name__ == '__main__':
    main()
