"""Compare complete spatial-feature pipelines over the same uncompressed cache.

Conversion and native GeoTIFF export are reported separately, never hidden in
an end-to-end native-product claim. No KvikIO mode is automatically called GDS.
"""
import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

from .benchmark import _git
from .datasets import fetch_worldview_sample, sha256, _write_json
from .gpu_io import RasterCache, CacheWriter, pack_raster, io_mode, io_evidence, module, synchronize, MODES
from .paper_benchmark import _summary
from .worldview import process_worldview


def features(tile, sizes, xp):
    """Multi-scale finite-neighbor means and variances; one device residency."""
    filt = uniform_filter
    if xp is not np:
        from cupyx.scipy.ndimage import uniform_filter as filt
    result=[]
    for band in tile:
        # Float64 moments avoid cancellation in low-variance neighborhoods.
        band=band.astype(xp.float64)
        valid=xp.isfinite(band)
        values=xp.where(valid,band,0.)
        for size in sizes:
            count=xp.rint(filt(valid.astype(xp.float64),size,mode='constant')*size**2)
            denominator=xp.where(count>0,count,1.)
            mean=filt(values,size,mode='constant')*size**2/denominator
            second=filt(values*values,size,mode='constant')*size**2/denominator
            result.extend((xp.where(count>0,mean,xp.nan),
                           xp.where(count>0,xp.maximum(second-mean*mean,0),xp.nan)))
    return xp.stack(result).astype(xp.float32)


def reference(tile,sizes):
    """Independent float64 SciPy reference, outside benchmark timing."""
    result=[]
    for band in tile.astype('float64'):
        valid=np.isfinite(band)
        data=np.where(valid,band,0.)
        for size in sizes:
            count=np.rint(uniform_filter(valid.astype('float64'),size,mode='constant')*size**2)
            mean=np.full(band.shape,np.nan);second=mean.copy()
            np.divide(uniform_filter(data,size,mode='constant')*size**2,count,out=mean,where=count>0)
            np.divide(uniform_filter(data*data,size,mode='constant')*size**2,count,out=second,where=count>0)
            result.extend((mean,np.maximum(second-mean*mean,0)))
    return np.stack(result)


def output_metadata(cache,sizes):
    bands=[f'{b}_{stat}_{size}' for b in cache.meta['bands'] for size in sizes for stat in ('mean','variance')]
    return {**cache.meta,'count':len(bands),'bands':bands,'halo':0,
            'operation':'finite-neighbor multi-scale means and variances','focal_sizes':list(sizes)}


def process(cache,destination,sizes,mode):
    if not sizes or any(type(s) is not int or s<1 or s%2==0 for s in sizes):
        raise ValueError('Expected positive odd focal sizes')
    if max(sizes)//2 > cache.meta['halo']:
        raise ValueError('Cache halo is too small for the requested focal sizes')
    xp=module(mode);meta=output_metadata(cache,sizes);halo=cache.meta['halo']
    timing=dict(read_seconds=0.,compute_seconds=0.,write_seconds=0.)
    synchronize(mode);started=time.perf_counter()
    with io_mode(mode),CacheWriter(destination,meta,mode) as writer:
        for row,col in cache.tiles():
            start=time.perf_counter();tile=cache.read_tile(row,col,mode);synchronize(mode)
            timing['read_seconds']+=time.perf_counter()-start
            start=time.perf_counter();result=features(tile,sizes,xp)
            h,w=cache.core_shape(row,col)
            core=xp.full((meta['count'],meta['tile_size'],meta['tile_size']),xp.nan,dtype=xp.float32)
            core[:,:h,:w]=result[:,halo:halo+h,halo:halo+w]
            synchronize(mode);timing['compute_seconds']+=time.perf_counter()-start
            start=time.perf_counter();writer.write_tile(row,col,core);synchronize(mode)
            timing['write_seconds']+=time.perf_counter()-start
            del tile,result,core
    synchronize(mode)
    timing['total_seconds']=time.perf_counter()-started
    timing['finalization_and_loop_seconds']=max(0.,timing['total_seconds']-sum(timing[k] for k in ('read_seconds','compute_seconds','write_seconds')))
    return timing


def validate(cache,output,sizes):
    actual=RasterCache(output);halo=cache.meta['halo'];maximum=0.;count=0
    expected_meta=output_metadata(cache,sizes)
    for key in ('height','width','crs_wkt','transform','bands','count','halo'):
        if actual.meta[key] != expected_meta[key]:raise AssertionError('Output metadata differs: '+key)
    for row,col in cache.tiles():
        h,w=cache.core_shape(row,col)
        expected=reference(cache.read_tile(row,col),sizes)[:,halo:halo+h,halo:halo+w]
        data=actual.read_tile(row,col)[:,:h,:w]
        np.testing.assert_array_equal(np.isfinite(data),np.isfinite(expected))
        np.testing.assert_allclose(data,expected,rtol=2e-5,atol=2e-6,equal_nan=True)
        valid=np.isfinite(expected);count+=int(valid.sum())
        if valid.any():maximum=max(maximum,float(np.max(np.abs(data[valid]-expected[valid]))))
    if count == 0:raise AssertionError('No finite output values')
    return dict(finite_values=count,max_absolute_error=maximum,rtol=2e-5,atol=2e-6,metadata_and_masks_match=True)


def probe(mode,work):
    """Test read AND write on the actual benchmark mount before a long run."""
    meta=dict(schema='terragpu-raster-cache-v1',dtype='<f4',nodata='NaN',height=64,width=64,count=1,
              tile_size=64,halo=0,crs_wkt=rasterio.crs.CRS.from_epsg(4326).to_wkt(),transform=[1,0,0,0,-1,64],bands=['probe'])
    xp=module(mode)
    with tempfile.TemporaryDirectory(dir=work,prefix='io-probe-') as tmp,io_mode(mode):
        path=Path(tmp)/'cache'
        with CacheWriter(path,meta,mode) as writer:
            writer.write_tile(0,0,xp.full((1,64,64),3.25,dtype=xp.float32))
        # Independent host read catches direct-write problems; device read catches read problems.
        np.testing.assert_equal(RasterCache(path).read_tile(0,0),3.25)
        result=RasterCache(path).read_tile(0,0,mode)
        result=xp.asnumpy(result) if mode != 'numpy' else result
        np.testing.assert_equal(result,3.25)
    return io_evidence(mode)


def run(output,work_root='data/gpu-io',data_root='data',source=None,modes=('numpy','cupy','kvikio-compat','kvikio-cufile'),
        tiles=(512,1024),sizes=(15,31),repeat=5,warmup=1,require_cufile=False,allow_dirty=False):
    if repeat<1 or warmup<0 or not sizes or any(s<1 or s%2==0 for s in sizes) or not tiles or any(t<1 for t in tiles):
        raise ValueError('Invalid experiment sizes/repetitions')
    if not modes or len(set(modes))!=len(modes) or any(m not in MODES for m in modes) or len(set(tiles))!=len(tiles):
        raise ValueError('Invalid modes or duplicate tile sizes')
    if len(set(sizes)) != len(sizes):raise ValueError('Duplicate focal sizes')
    if require_cufile and 'kvikio-cufile' not in modes:raise ValueError('--require-cufile requires the kvikio-cufile mode')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    dirty=bool(_git('status','--porcelain'))
    if dirty and not allow_dirty:raise RuntimeError('Commit source changes before benchmarking; --allow-dirty is for development only')
    work=Path(work_root);work.mkdir(parents=True,exist_ok=True)
    report=dict(schema_version=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),git_commit=_git('rev-parse','HEAD'),git_dirty=dirty,
                repeat=repeat,warmup=warmup,tiles=list(tiles),focal_sizes=list(sizes),records=[],preparation=[],probes={},skipped_modes={},
                hardware=dict(platform=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count()),
                packages={d.metadata['Name']:d.version for d in distributions()},
                threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
                gds_verified=False,notes=['Experimental uncompressed halo cache; not native GeoTIFF/NetCDF GPU decoding.',
                'Float64 moment accumulation and float32 cache output on both CPU and GPU.',
                'CPU metadata, initial ingestion and explicit GeoTIFF export remain on host.',
                'Cache pipeline read timings include H2D in cupy mode; write timings include D2H in cupy mode.',
                'KvikIO cuFile mode is requested, not verified direct storage. Inspect mount-specific GDS diagnostics.',
                'All measured outputs independently validated outside timing; filesystem caches uncontrolled.',
                'No fsync durability barrier, asynchronous overlap or measured peak device memory claim.',
                'Source packing is charged separately for each layout; do not label cache-only ratios as native-product speedups.'])
    output.parent.mkdir(parents=True,exist_ok=True);partial=output.with_suffix('.partial.json')
    if partial.exists():raise FileExistsError(partial)
    _write_json(partial,report)
    active=[]
    for mode in modes:
        print('Probing '+mode,flush=True)
        try:report['probes'][mode]=probe(mode,work)
        except Exception as error:
            if mode!='kvikio-cufile' or require_cufile:
                raise
            report['skipped_modes'][mode]=dict(error_type=type(error).__name__,reason='cuFile read/write probe failed on this mount; see system GDS diagnostics')
            print(f'cuFile probe unavailable ({type(error).__name__}: {error}); continuing with explicitly labelled baseline modes',flush=True)
        else:active.append(mode)
        _write_json(partial,report)
    if not active:raise RuntimeError('No available mode')
    if any(m!='numpy' for m in active):
        cp=module('cupy');props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        report['hardware']['gpu']=dict(name=props['name'].decode() if isinstance(props['name'],bytes) else props['name'],
            total_memory_bytes=props['totalGlobalMem'],cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion())
    with tempfile.TemporaryDirectory(dir=work,prefix='experiment-') as tmp:
        tmp=Path(tmp)
        if source is None:
            print('Preparing validated public WorldView NDVI/NDWI source',flush=True)
            start=time.perf_counter();item=fetch_worldview_sample(Path(data_root)/'worldview-example')
            report['download_verify_seconds']=time.perf_counter()-start
            source=tmp/'indices.tif';start=time.perf_counter();process_worldview(item,source)
            # Reuse the independent QA/georeferencing reference from the repository.
            from .worldview_validation import validate as validate_worldview
            report['native_preparation_seconds']=time.perf_counter()-start
            report['source_validation']=validate_worldview(item,source)
            report['source_kind']='Public WorldView-3 QA-masked NDVI/NDWI; CPU preparation included separately'
        else:
            report['native_preparation_seconds']=0.
            report['source_kind']='User supplied georeferenced GeoTIFF; original product preparation not measured'
        report['source_sha256']=sha256(source)
        for tile in tiles:
            print(f'Packing tile size {tile}',flush=True)
            cache_path=tmp/f'input-{tile}'
            prep=pack_raster(source,cache_path,tile_size=tile,halo=max(sizes)//2)
            report['preparation'].append(dict(tile_size=tile,**prep))
            cache=RasterCache(cache_path,verify=True)
            samples={m:[] for m in active};cold={};checks={}
            for iteration in range(1+warmup+repeat):
                # Rotate mode order so one backend is not always the first reader.
                order=active[iteration%len(active):]+active[:iteration%len(active)]
                for mode in order:
                    target=tmp/'output'
                    print(f'tile={tile} iteration={iteration} mode={mode}',flush=True)
                    timing=process(cache,target,sizes,mode)
                    check=validate(cache,target,sizes)
                    previous=checks.get(mode,{}).get('max_absolute_error',0.)
                    check['max_absolute_error']=max(check['max_absolute_error'],previous);checks[mode]=check
                    if iteration==0:cold[mode]=timing
                    elif iteration>warmup:samples[mode].append(timing)
                    # Measure one real standard-format export, separate from cache pipeline timings.
                    if iteration==0 and mode==active[0]:
                        exported=tmp/'export.tif';start=time.perf_counter();RasterCache(target).to_geotiff(exported)
                        prep['geotiff_export_seconds']=time.perf_counter()-start;exported.unlink()
                        report['preparation'][-1].update(geotiff_export_seconds=prep['geotiff_export_seconds'])
                    import shutil
                    shutil.rmtree(target)
            for mode in active:
                times=[s['total_seconds'] for s in samples[mode]]
                report['records'].append(dict(workload='worldview_spatial_features' if report['source_kind'].startswith('Public') else 'raster_spatial_features',
                    mode=mode,tile_size=tile,shape=[cache.meta['count'],cache.meta['height'],cache.meta['width']],
                    scope='uncompressed cache read, multi-scale computation, uncompressed cache write and metadata close; validation excluded',
                    samples_seconds=times,phase_samples=samples[mode],cold=cold[mode],correctness_passed=True,validation=checks[mode],
                    logical_input_bytes=prep['logical_bytes'],physical_input_bytes=prep['physical_bytes'],
                    input_bytes_per_second=prep['logical_bytes']/float(np.median(times)),**_summary(times)))
            _write_json(partial,report)
    report['status']='complete' if not report['skipped_modes'] else 'complete_available_modes'
    _write_json(output,report);partial.unlink()
    rows=[]
    for r in report['records']:
        base=next((v for v in report['records'] if v['mode']=='numpy' and v['tile_size']==r['tile_size']),None)
        rows.append(dict(mode=r['mode'],tile_size=r['tile_size'],median_seconds=r['median_seconds'],p95_seconds=r['p95_seconds'],
                         cpu_over_mode=base['median_seconds']/r['median_seconds'] if base else '',
                         read_median_seconds=float(np.median([v['read_seconds'] for v in r['phase_samples']])),
                         compute_median_seconds=float(np.median([v['compute_seconds'] for v in r['phase_samples']])),
                         write_median_seconds=float(np.median([v['write_seconds'] for v in r['phase_samples']])),
                         correctness_passed=True,gds_verified=False))
    with output.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--work-root',default='data/gpu-io');p.add_argument('--data-root',default='data')
    p.add_argument('--source',type=Path);p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    p.add_argument('--tiles',nargs='+',type=int,default=[512,1024]);p.add_argument('--sizes',nargs='+',type=int,default=[15,31])
    p.add_argument('--repeat',type=int,default=5);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--require-cufile',action='store_true');p.add_argument('--allow-dirty',action='store_true')
    run(**vars(p.parse_args()))


if __name__=='__main__':main()
