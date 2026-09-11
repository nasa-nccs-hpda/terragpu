"""Public-data CPU/GPU benchmark suite; JSON raw samples are the primary artifact."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
import hashlib
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import rasterio

from .benchmark import _git
from .datasets import fetch_satstereo, fetch_worldview_sample, sha256, _write_json
from .engine import array_module
from .workloads import focal_mean, spectral_angle, stereo_census, disparity_metrics
from .worldview import process_worldview


def _summary(samples):
    rng=np.random.default_rng(82)
    medians=np.median(rng.choice(samples,(2000,len(samples)),replace=True),axis=1)
    return dict(median_seconds=float(np.median(samples)), min_seconds=float(min(samples)),
                p95_seconds=float(np.percentile(samples,95)),
                median_bootstrap_ci95_seconds=np.percentile(medians,[2.5,97.5]).tolist())


def run(output, data_root='data', backends=('numpy','cupy'), repeat=5, warmup=1,
        smoke=False, allow_dirty=False):
    if repeat<1 or warmup<0 or not backends or any(b not in ('numpy','cupy') for b in backends):
        raise ValueError('Invalid benchmark configuration')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    dirty=bool(_git('status','--porcelain'))
    if dirty and not allow_dirty:raise RuntimeError('Commit changes before benchmarking (or use --allow-dirty for development)')
    # Fail before downloads if the requested GPU is unavailable.
    modules={b:array_module(b) for b in backends}
    data_root=Path(data_root)
    stereo=fetch_satstereo(data_root/'satstereo')
    item=fetch_worldview_sample(data_root/'worldview-example')
    meta=json.loads(item.read_text());asset=meta['assets']['ms_analytic']
    wv_path=item.parent/asset['href']
    images=sorted((stereo/'Rectified_Chips').glob('*.tif'))
    left_path=images[0]
    first,second=left_path.stem.removesuffix('_Rectified').split('_and_')
    right_path=stereo/'Rectified_Chips'/f'{second}_and_{first}_Rectified.tif'
    truth_path=stereo/'Disparity'/f'{first}_and_{second}_disp.tif'
    mask_path=stereo/'Masks'/f'{first}_and_{second}_mask.png'
    def read(path, band=1, dtype='float32'):
        with rasterio.open(path) as ds:return ds.read(band,masked=True,out_dtype=dtype).filled(np.nan)
    left,right=map(read,(left_path,right_path))
    truth=read(truth_path,dtype='float64')
    building=read(mask_path)>0
    bands=[b['common_name'] for b in asset['eo:bands']]
    spatial=read(wv_path,bands.index('red')+1)*np.float32(.0001)
    if smoke:
        left,right,truth,building=[a[:128,:256] for a in (left,right,truth,building)]
        spatial=spatial[:256,:256]
    rng=np.random.default_rng(14)
    side=64 if smoke else 512
    cube=rng.uniform(.001,.05,(side,side,136)).astype('float32')
    reference=np.linspace(.003,.04,136,dtype='float32')
    disparity_bounds=(-16,16) if smoke else (-128,128)
    tasks=[('worldview_focal15', [spatial], lambda a,xp:focal_mean(a[0],15,xp=xp)),
           ('spectral_angle136', [cube,reference], lambda a,xp:spectral_angle(a[0],a[1],xp=xp)),
           ('satstereo_census', [left,right], lambda a,xp:stereo_census(a[0],a[1],*disparity_bounds,xp=xp))]
    report=dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                git_commit=_git('rev-parse','HEAD'), git_dirty=dirty, smoke=smoke,
                repeat=repeat,warmup=warmup, cpu_policy='one controlling thread; thread limits recorded',
                configuration=dict(synthetic_seed=14,reference_spectrum='linspace(0.003,0.04,136)',
                                   focal_window=15,census_window=5,cost_window=5,disparity_bounds=list(disparity_bounds)),
                hardware=dict(platform=platform.platform(),processor=platform.processor(),
                              cpu_count=os.cpu_count(),cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None),
                threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
                packages={d.metadata['Name']:d.version for d in distributions()},
                inputs=[dict(name=p.name,sha256=sha256(p),size_bytes=p.stat().st_size)
                        for p in [left_path,right_path,truth_path,mask_path,wv_path,item,
                                  item.parent/meta['assets']['cloud-mask-raster']['href'],
                                  item.parent/meta['assets']['ms-saturation-mask-raster']['href']]],
                records=[], notes=['Timing repetitions are not independent scenes.',
                    'Filesystem caches are uncontrolled. Validation and downloads are outside timings.',
                    'Stereo is matching only, not bundle adjustment or georeferenced 3-D reconstruction.',
                    'No per-workload peak-memory or energy measurement is claimed.'])
    if 'cupy' in modules:
        cp=modules['cupy'];props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        report['hardware']['gpu']=dict(name=props['name'].decode() if isinstance(props['name'],bytes) else props['name'],
            total_memory_bytes=props['totalGlobalMem'],cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            cuda_driver=cp.cuda.runtime.driverGetVersion(),device_id=cp.cuda.Device().id)
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_suffix('.partial.json')
    for name,host,operation in tasks:
        expected=operation(host,np)
        # Independent CPU arithmetic checks supplement the GPU parity check.
        if name=='worldview_focal15':
            for y,x in [(8,8),(min(100,spatial.shape[0]-1),min(100,spatial.shape[1]-1))]:
                patch=spatial[max(0,y-7):y+8,max(0,x-7):x+8]
                target=np.nanmean(patch.astype('float64')) if np.isfinite(patch).any() else np.nan
                np.testing.assert_allclose(expected[y,x],target,rtol=2e-5,atol=1e-6,equal_nan=True)
        elif name=='spectral_angle136':
            double=cube.astype('float64');ref=reference.astype('float64')
            target=np.arccos(np.clip(np.sum(double*ref,axis=-1)/np.sqrt(np.sum(double*double,axis=-1)*np.sum(ref*ref)),-1,1))
            np.testing.assert_allclose(expected,target,rtol=2e-5,atol=2e-6)
            del double,target
        quality=None
        if name=='satstereo_census':
            quality=dict(all_reference=disparity_metrics(expected,truth), buildings=disparity_metrics(expected,truth,building),
                         disparity_min=disparity_bounds[0],disparity_max=disparity_bounds[1],
                         convention='x_right=x_left+disparity',reference_outside_search=int((np.isfinite(truth)&((truth<disparity_bounds[0])|(truth>disparity_bounds[1]))).sum()))
        for backend in backends:
            xp=modules[backend]
            def sync():
                if backend=='cupy':xp.cuda.runtime.deviceSynchronize()
            sync();start=time.perf_counter();resident=[xp.asarray(a) for a in host];sync()
            transfer=time.perf_counter()-start
            samples=[];cold=None
            for iteration in range(1+warmup+repeat):
                sync();start=time.perf_counter();result=operation(resident,xp);sync();elapsed=time.perf_counter()-start
                if iteration==0:cold=elapsed
                elif iteration>warmup:samples.append(elapsed)
                # Every measured output checked, outside timing.
                actual=xp.asnumpy(result) if backend=='cupy' else result
                if name=='satstereo_census':np.testing.assert_equal(actual,expected)
                else:np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6,equal_nan=True)
                del result,actual
            record=dict(workload=name,backend=backend,shape=list(host[0].shape),input_bytes=sum(a.nbytes for a in host),
                        input_array_sha256=[hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for a in host],
                        scope='resident input; allocation and computation included; H2D/D2H excluded',
                        input_transfer_seconds=transfer,cold_seconds=cold,samples_seconds=samples,
                        correctness_passed=True,quality=quality,**_summary(samples))
            report['records'].append(record);_write_json(partial,report)
            print(f'{name} {backend}: {record["median_seconds"]:.4f}s',flush=True)
            del resident
        del expected
    # Real-data end-to-end workflow: identical compressed output settings on CPU/GPU.
    if not smoke:
        with tempfile.TemporaryDirectory(prefix='terragpu-paper-',dir=output.parent) as temp:
            baseline=Path(temp)/'baseline.tif';process_worldview(item,baseline)
            with rasterio.open(baseline) as ds:expected=ds.read()
            records={b:[] for b in backends}
            for iteration in range(warmup+repeat):
                order=backends if iteration%2==0 else tuple(reversed(backends))
                for backend in order:
                    path=Path(temp)/f'{backend}-{iteration}.tif'
                    start=time.perf_counter();process_worldview(item,path,backend=backend);elapsed=time.perf_counter()-start
                    with rasterio.open(path) as ds:
                        np.testing.assert_allclose(ds.read(),expected,rtol=1e-5,atol=2e-7,equal_nan=True)
                    path.unlink()
                    if iteration>=warmup:records[backend].append(elapsed)
            for backend,samples in records.items():
                report['records'].append(dict(workload='worldview_ndvi_ndwi_io',backend=backend,
                    scope='open/read/QA alignment/transfers/compute/compressed write/close; warm filesystem cache',
                    shape=list(expected.shape),samples_seconds=samples,correctness_passed=True,**_summary(samples)))
    _write_json(output,report)
    if partial.exists():partial.unlink()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',default='data')
    parser.add_argument('--backends',nargs='+',choices=['numpy','cupy'],default=['numpy','cupy'])
    parser.add_argument('--repeat',type=int,default=5)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--allow-dirty',action='store_true')
    run(**vars(parser.parse_args()))


if __name__=='__main__':main()
