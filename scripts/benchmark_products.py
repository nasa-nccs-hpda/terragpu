"""Download fixed NASA examples and benchmark validated full-product CPU/GPU I/O.

Run as python -m scripts.benchmark_products from the checkout. EARTHDATA_TOKEN
is read only by earthaccess; credentials and access URLs are never report fields.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
import time

from terragpu.benchmark import _git
from terragpu.datasets import nasa_data, fetch_worldview_sample, fetch_satstereo, _write_json
from terragpu.engine import array_module
from terragpu.hls import process_hls_ndvi
from terragpu.pace import process_pace
from terragpu.viirs import process_viirs
from terragpu.paper_benchmark import _summary
from scripts.validate_hls import validate as validate_hls
from scripts.validate_pace import validate as validate_swath

SAMPLES = {
    'hls': dict(short_name='HLSL30', version='2.0', bbox=[-77.1,38.8,-76.9,39.0],
                start='2024-06-01', end='2024-07-01',
                granule_name='HLS.L30.T18SUJ.2024158T154508.v2.0'),
    'pace': dict(short_name='PACE_OCI_L2_AOP', version='3.2', bbox=[-76,34,-74,36],
                 start='2024-06-01', end='2024-06-03',
                 granule_name='PACE_OCI.20240601T165222.L2.OC_AOP.V3_2.nc'),
    'viirs': dict(short_name='VIIRSJ2_L2_OC', version='2025.0', bbox=[-76,34,-74,36],
                  start='2024-06-01', end='2024-06-03',
                  granule_name='VIIRSJ2_L2_OC_JPSS2_VIIRS.20240601T173601.L2.OC.nc_2025.0'),
}


def safe_failure(error):
    """Keep diagnostic categories/status codes, never provider messages or URLs."""
    known = {
        'Use a new output directory or a completed TerraGPU cache': 'Cache directory exists without a manifest; choose a new TERRAGPU_DATA_ROOT or move the incomplete cache aside.',
        'Cached query differs; choose a new output directory': 'Existing cache selects a different scene/query; choose a new TERRAGPU_DATA_ROOT.',
        'Cached manifest contains no files': 'Cache manifest is empty; use a new cache directory.',
        'Cached file missing/corrupt; use a new output directory': 'Cache checksum verification failed; use a new cache directory.',
        'Invalid cached filename': 'Cache manifest contains an invalid filename.',
        'No granules found; adjust dates, region or collection/version': 'NASA catalog returned no matches for the fixed granule.',
        'NASA returned no downloaded files': 'NASA returned no files; check token validity, product access and compute-node network access.',
        'Unexpected or empty downloaded file': 'Downloader returned an empty or unexpected file.',
        'Earthdata authentication required; authenticate locally, not in chat': 'Earthdata authentication was rejected; check EARTHDATA_TOKEN.',
    }
    if str(error) in known:
        return known[str(error)]
    # Inspect structured metadata only; exception strings may embed credentials.
    current = error; seen = set(); statuses = set(); classes = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        classes.update(c.__name__ for c in type(current).__mro__)
        status = getattr(getattr(current, 'response', None), 'status_code', None)
        if type(status) is int and 100 <= status <= 599:
            statuses.add(status)
        current = current.__cause__ or current.__context__
    if statuses:
        codes = ', '.join(map(str, sorted(statuses)))
        return f'HTTP status {codes}; 401/403 indicate authentication or access denial; 429/5xx indicate a service/rate-limit failure.'
    for category, advice in [
        ('SSLError', 'TLS certificate verification failed; check the node certificate/proxy configuration.'),
        ('Timeout', 'Network request timed out; try download-only on a network-enabled node.'),
        ('ConnectionError', 'Network connection failed; check DNS, proxy and compute-node outbound access.'),
        ('ImportError', 'Downloader dependency could not be imported; rerun setup_prism_uv.sh to install the data extra.'),
        ('PermissionError', 'Filesystem permission denied; check cache directory access.'),
        ('JSONDecodeError', 'Invalid JSON was returned or read from the cache manifest.'),
    ]:
        if category in classes:
            return advice
    if isinstance(error, OSError) and type(error.errno) is int:
        return f'Filesystem/network OS error number {error.errno}; check available space and directory access.'
    return 'Exception type '+type(error).__name__+'; provider message omitted to protect credentials.'


def prepare(data_root):
    root = Path(data_root)
    manifests = {}
    for name, query in SAMPLES.items():
        destination = root/(name+'-example')
        if not destination.exists() and not os.environ.get('EARTHDATA_TOKEN', '').strip():
            raise RuntimeError('Export EARTHDATA_TOKEN before downloading NASA examples, or prefetch the complete cache')
        print(f'Verifying/downloading {name} example', flush=True)
        # Suppress third-party download diagnostics, including signed URLs. Never
        # propagate provider exception text which may contain authorization data.
        previous_logging = logging.root.manager.disable
        phase = 'initialization'
        def progress(value):
            nonlocal phase
            phase = value
        try:
            logging.disable(logging.CRITICAL)
            with open(os.devnull, 'w') as sink, redirect_stdout(sink), redirect_stderr(sink):
                manifest = nasa_data(**query, output=destination, progress=progress)
        except Exception as error:
            raise RuntimeError(f'{name} failed during {phase}: {safe_failure(error)}') from None
        finally:
            logging.disable(previous_logging)
        if len(manifest['granules']) != 1 or manifest['granules'][0]['granule_ur'] != query['granule_name']:
            raise ValueError(f'{name}: downloaded granule does not match the fixed benchmark input')
        manifests[name] = manifest
    fetch_worldview_sample(root/'worldview-example')
    fetch_satstereo(root/'satstereo')
    return manifests


def measure(name, source, operation, validator, suffix, backends, repeat, warmup, directory):
    """Alternate backend order; validate EVERY output outside the measured region."""
    modules = {b: array_module(b) for b in backends}
    samples = {b: [] for b in backends}
    cold = {}; checks = {}; sizes = {}; shapes = {}
    with tempfile.TemporaryDirectory(prefix='product-', dir=directory) as temp:
        for iteration in range(1+warmup+repeat):
            for backend in (backends if iteration % 2 == 0 else tuple(reversed(backends))):
                target = Path(temp)/(backend+suffix)
                def sync():
                    if backend == 'cupy':
                        modules[backend].cuda.runtime.deviceSynchronize()
                sync()
                start = time.perf_counter()
                operation(source, target, backend=backend)
                sync()
                elapsed = time.perf_counter()-start
                check = validator(source, target)
                if check['valid_pixels'] <= 0:
                    raise ValueError(f'{name}: empty scientific output')
                if iteration == 0:
                    cold[backend] = elapsed
                elif iteration > warmup:
                    samples[backend].append(elapsed)
                # Preserve the worst numerical error across all repetitions.
                previous = checks.get(backend, {})
                check['max_absolute_error'] = max(check['max_absolute_error'], previous.get('max_absolute_error', 0.))
                checks[backend] = check
                sizes[backend] = target.stat().st_size
                shapes[backend] = check['shape']
                target.unlink()
    return [dict(workload=name, backend=b, scope='open/read/QA/transfers/compute/compressed write/close; validation excluded; filesystem cache uncontrolled',
                 shape=shapes[b], samples_seconds=samples[b], cold_seconds=cold[b],
                 correctness_passed=True, validation=checks[b], output_bytes=sizes[b],
                 valid_pixels_per_second=checks[b]['valid_pixels']/_summary(samples[b])['median_seconds'],
                 **_summary(samples[b])) for b in backends]


def run(output, data_root='data', backends=('numpy','cupy'), repeat=7, warmup=2, allow_dirty=False):
    if repeat < 1 or warmup < 0 or not backends or len(set(backends)) != len(backends):
        raise ValueError('Invalid repetition/backend configuration')
    output = Path(output)
    if output.exists() or output.with_suffix('.partial.json').exists():
        raise FileExistsError(output)
    dirty = bool(_git('status','--porcelain'))
    if dirty and not allow_dirty:
        raise RuntimeError('Commit changes before benchmarking (or use --allow-dirty for development)')
    for backend in backends:
        array_module(backend)
    manifests = prepare(data_root)  # All downloads and hashes outside timing.
    report = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  git_commit=_git('rev-parse','HEAD'), git_dirty=dirty, repeat=repeat, warmup=warmup,
                  requested_backends=list(backends), inputs=manifests, records=[],
                  notes=['Single fixed granule per NASA product, not scene generalization.',
                         'Hardware/software and thread policy recorded by the companion suite.json and PRISM runner.',
                         'PACE dense and VIIRS sparse spectral means are different products, not interchangeable retrievals.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix('.partial.json')
    _write_json(partial, report)
    for product, name, operation, validator, suffix in [
            ('hls','hls_ndvi_io',process_hls_ndvi,validate_hls,'.tif'),
            ('pace','pace_spectral_mean_io',process_pace,validate_swath,'.nc'),
            ('viirs','viirs_spectral_mean_io',process_viirs,validate_swath,'.nc')]:
        source = Path(data_root)/(product+'-example')
        if product != 'hls':
            source, = source.glob('*.nc')
        print(f'Benchmarking {product}: '+', '.join(backends), flush=True)
        records = measure(name,source,operation,validator,suffix,backends,repeat,warmup,output.parent)
        for record in records:
            record['input_sha256'] = {v['name']: v['sha256'] for v in manifests[product]['files']}
        report['records'].extend(records)
        _write_json(partial, report)
    _write_json(output, report)
    partial.unlink()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--backends', nargs='+', choices=['numpy','cupy'], default=['numpy','cupy'])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--allow-dirty', action='store_true')
    parser.add_argument('--download-only', action='store_true')
    args = vars(parser.parse_args())
    if args.pop('download_only'):
        prepare(args['data_root'])
    elif args['output'] is None:
        parser.error('--output is required unless --download-only is used')
    else:
        run(**args)


if __name__ == '__main__':
    main()
