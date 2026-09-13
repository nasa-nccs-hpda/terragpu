"""Native GeoTIFF spatial pipelines, CPU worker sweeps and tile residency reuse.

Each timed trial includes input open/decode, optional product preparation/cache
packing, all requested analyses and compressed georeferenced output close.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import tempfile
import threading
import time

import numpy as np
import psutil
import rasterio
from rasterio.windows import Window

from .benchmark import _git
from .benchmark_gpu_io import features, reference
from .datasets import fetch_worldview_sample, sha256, _write_json
from .engine import array_module
from .gpu_io import RasterCache, pack_raster
from .paper_benchmark import _summary
from .worldview import process_worldview
from .worldview_validation import validate as validate_worldview


def available_cpus():
    count=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    allocated=os.environ.get('SLURM_CPUS_PER_TASK')
    if allocated:count=min(count,int(allocated))
    return max(1,count)


class MemoryMonitor:
    """Sample process RSS and device-wide CUDA usage; allocator reservation too.

    Device usage includes other processes; pool reservation is this process's
    default CuPy allocator, not all CUDA allocations. Sampling can miss peaks.
    """
    def __init__(self,xp=None,interval=.01):
        self.xp=xp;self.interval=interval;self.stop=threading.Event()
        self.process=psutil.Process();self.error=None
        self.rss_peak=0;self.device_peak=None;self.pool_peak=None
        self.device_id=xp.cuda.Device().id if xp is not None else None

    def sample(self):
        self.rss_peak=max(self.rss_peak,self.process.memory_info().rss)
        if self.xp is not None:
            free,total=self.xp.cuda.runtime.memGetInfo()
            self.device_peak=max(self.device_peak or 0,total-free)
            self.pool_peak=max(self.pool_peak or 0,self.xp.get_default_memory_pool().total_bytes())

    def loop(self):
        try:
            if self.xp is not None:self.xp.cuda.Device(self.device_id).use()
            while not self.stop.wait(self.interval):self.sample()
        except Exception as error:self.error=type(error).__name__

    def __enter__(self):
        self.sample();self.rss_start=self.rss_peak;self.device_start=self.device_peak
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.sample()

    def result(self):
        return dict(process_rss_start_bytes=self.rss_start,process_rss_sampled_peak_bytes=self.rss_peak,
                    device_used_start_bytes=self.device_start,device_used_sampled_peak_bytes=self.device_peak,
                    cupy_pool_reserved_sampled_peak_bytes=self.pool_peak,sampling_interval_seconds=self.interval,
                    sampling_error=self.error)


def read_tile(source,row,col,tile,halo):
    with rasterio.open(source) as src:
        top,left=max(0,row-halo),max(0,col-halo)
        bottom,right=min(src.height,row+tile+halo),min(src.width,col+tile+halo)
        values=src.read(window=Window(left,top,right-left,bottom-top),masked=True,out_dtype='float32').filled(np.nan)
        values=values*np.asarray(src.scales,dtype='float32')[:,None,None]+np.asarray(src.offsets,dtype='float32')[:,None,None]
        data=np.full((src.count,tile+2*halo,tile+2*halo),np.nan,dtype='float32')
        y,x=top-row+halo,left-col+halo
        data[:,y:y+values.shape[1],x:x+values.shape[2]]=values
        return data


def names(bands,sizes):
    return tuple(f'{b}_{stat}_{size}' for b in bands for size in sizes for stat in ('mean','variance'))


def bounded_map(function,items,workers):
    if workers==1:
        for item in items:yield function(item)
        return
    # SciPy filters release the GIL. Bound pending results to avoid image-sized RAM.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending=deque();items=iter(items)
        for _ in range(workers):
            item=next(items,None)
            if item is not None:pending.append(executor.submit(function,item))
        while pending:
            yield pending.popleft().result()
            item=next(items,None)
            if item is not None:pending.append(executor.submit(function,item))


def pipeline(source,folder,queries,*,backend='numpy',workers=1,tile=1024,strategy='reuse'):
    if backend not in ('numpy','cupy') or workers<1 or (backend=='cupy' and workers!=1):
        raise ValueError('GPU uses one controlling worker; CPU requires positive worker count')
    if strategy not in ('stream','reuse','packed'):raise ValueError('Unknown strategy')
    xp=array_module(backend);halo=max(max(q) for q in queries)//2
    with rasterio.open(source) as src:
        height,width,count=src.height,src.width,src.count
        if src.crs is None:raise ValueError('Georeferenced input required')
        bands=[b or f'band{i+1}' for i,b in enumerate(src.descriptions)]
        profile=dict(driver='GTiff',height=height,width=width,crs=src.crs,transform=src.transform,
                     dtype='float32',nodata=np.nan,tiled=True,blockxsize=256,blockysize=256,
                     compress='LZW',NUM_THREADS='1')
    cache=None;packing=None
    if strategy=='packed':
        packing=pack_raster(source,folder/'cache',tile_size=tile,halo=halo)
        cache=RasterCache(folder/'cache')
    origins=[(r,c) for r in range(0,height,tile) for c in range(0,width,tile)]
    outputs=[folder/f'query-{i}.tif' for i in range(len(queries))]
    def calculate(task):
        row,col,indices=task
        data=cache.read_tile(row,col,backend) if cache else xp.asarray(read_tile(source,row,col,tile,halo))
        h,w=min(tile,height-row),min(tile,width-col)
        answers=[]
        for index in indices:
            value=features(data,queries[index],xp)[:,halo:halo+h,halo:halo+w]
            answers.append((index,xp.asnumpy(value) if backend=='cupy' else np.ascontiguousarray(value)))
        return row,col,h,w,answers
    with ExitStack() as stack:
        writers=[]
        for index,query in enumerate(queries):
            writer=stack.enter_context(rasterio.open(outputs[index],'w',count=count*len(query)*2,**profile))
            writer.descriptions=names(bands,query);writers.append(writer)
        if strategy=='stream':
            # Read/transfer each tile again for each distinct analysis.
            tasks=((r,c,[i]) for i in range(len(queries)) for r,c in origins)
        else:
            # One read/transfer per tile across all distinct analyses.
            tasks=((r,c,range(len(queries))) for r,c in origins)
        for row,col,h,w,answers in bounded_map(calculate,tasks,workers):
            for index,array in answers:writers[index].write(array,window=Window(col,row,w,h))
    if backend=='cupy':xp.cuda.get_current_stream().synchronize()
    return outputs,packing


def validate_outputs(source,outputs,queries,tile):
    if not queries or len(outputs)!=len(queries):
        raise AssertionError('Expected exactly one output per query')
    if len({Path(p).resolve() for p in outputs})!=len(outputs):
        raise AssertionError('Duplicate output paths')
    halo=max(max(q) for q in queries)//2;maximum=0.;finite=0
    with rasterio.open(source) as src:
        bands=[b or f'band{i+1}' for i,b in enumerate(src.descriptions)]
        for output,query in zip(outputs,queries):
            with rasterio.open(output) as dst:
                if (dst.driver!='GTiff' or any(dtype!='float32' for dtype in dst.dtypes)
                        or dst.nodata is None or not np.isnan(dst.nodata)
                        or dst.compression is None or dst.compression.value!='LZW'
                        or not dst.profile.get('tiled')
                        or any(shape!=(256,256) for shape in dst.block_shapes)):
                    raise AssertionError('Output encoding/layout mismatch')
                if (dst.shape,dst.crs,dst.transform,dst.descriptions)!=(src.shape,src.crs,src.transform,names(bands,query)):
                    raise AssertionError('Output metadata mismatch')
                for row in range(0,src.height,tile):
                    for col in range(0,src.width,tile):
                        h,w=min(tile,src.height-row),min(tile,src.width-col)
                        expected=reference(read_tile(source,row,col,tile,halo),query)[:,halo:halo+h,halo:halo+w]
                        actual=dst.read(window=Window(col,row,w,h))
                        np.testing.assert_array_equal(np.isfinite(actual),np.isfinite(expected))
                        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6,equal_nan=True)
                        mask=np.isfinite(expected);finite+=int(mask.sum())
                        if mask.any():maximum=max(maximum,float(np.max(np.abs(actual[mask]-expected[mask]))))
    if finite==0:raise AssertionError('No finite output values')
    return dict(finite_values=finite,max_absolute_error=maximum,rtol=2e-5,atol=2e-6,metadata_and_masks_match=True)


def run(output,*,source=None,data_root='data',work_root='data/publication',backends=('numpy','cupy'),
        workers=(1,8),tiles=(1024,),query_counts=(1,3),strategies=('stream','reuse','packed'),
        sizes=(15,31),repeat=5,warmup=1,seed=731,storage_label='unspecified',allow_dirty=False):
    if repeat<1 or warmup<0:raise ValueError('Invalid repetition count')
    for values in (workers,tiles,query_counts,sizes):
        if not values or len(set(values))!=len(values) or any(type(v) is not int or v<1 for v in values):
            raise ValueError('Expected unique positive integer configurations')
    if any(s%2==0 for s in sizes):raise ValueError('Focal sizes must be odd')
    if not backends or len(set(backends))!=len(backends) or not set(backends)<={'numpy','cupy'}:
        raise ValueError('Invalid backends')
    if not strategies or len(set(strategies))!=len(strategies) or not set(strategies)<={'stream','reuse','packed'}:
        raise ValueError('Invalid strategies')
    if max(workers)>available_cpus():raise ValueError('CPU workers exceed allocation/affinity; request more CPUs or reduce --workers')
    output=Path(output);partial=output.with_suffix('.partial.json')
    if output.exists() or partial.exists():raise FileExistsError(output)
    dirty=bool(_git('status','--porcelain'))
    if dirty and not allow_dirty:raise RuntimeError('Commit source changes before benchmark')
    cp=array_module('cupy') if 'cupy' in backends else None
    work=Path(work_root);work.mkdir(parents=True,exist_ok=True)
    item=fetch_worldview_sample(Path(data_root)/'worldview-example') if source is None else None
    source=Path(source) if source is not None else None
    if source is not None:
        with rasterio.open(source) as src:source_shape=[src.count,src.height,src.width]
        inputs=[dict(name=source.name,sha256=sha256(source),bytes=source.stat().st_size)]
    else:
        metadata=json.loads(item.read_text());paths=[item]+[item.parent/metadata['assets'][k]['href'] for k in ('ms_analytic','cloud-mask-raster','ms-saturation-mask-raster')]
        inputs=[dict(name=p.name,sha256=sha256(p),bytes=p.stat().st_size) for p in paths]
        with rasterio.open(paths[1]) as src:source_shape=[2,src.height,src.width]
    report=dict(schema_version=1,status='running',timestamp_utc=datetime.now(timezone.utc).isoformat(),
                git_commit=_git('rev-parse','HEAD'),git_dirty=dirty,inputs=inputs,source_shape=source_shape,
                source_kind='WorldView native ARD to QA-masked indices to spatial features' if item else 'user GeoTIFF to spatial features',
                repeat=repeat,warmup=warmup,seed=seed,storage_label=storage_label,records=[],execution_order=[],
                hardware=dict(platform=platform.platform(),cpu_count=os.cpu_count(),available_cpus=available_cpus(),
                              cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None),
                packages={d.metadata['Name']:d.version for d in distributions()},
                threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','SLURM_CPUS_PER_TASK')},
                notes=['All times include decode, optional native product preparation/packing, analyses and compressed GeoTIFF close.',
                       'Downloads, hashing, validation, warmup and temporary-file removal are outside measured total.',
                       'CPU ThreadPoolExecutor workers parallelize tile reads and SciPy computation; one output writer; GDAL codec threads=1.',
                       'Distinct analyses scale window radii by query_index+1; reuse retains each tile, not the whole scene.',
                       'Packed strategy uses existing ordinary host/CuPy I/O; no GDS claim or GPU TIFF decoding.',
                       'Memory is sampled at 10 ms; RSS is process-wide, CUDA usage device-wide, pool reservation process-local.',
                       'Device-wide usage can include unrelated processes. Sampled peaks can miss transients.',
                       'Filesystem caches uncontrolled; no fsync durability barrier. Repeats within this invocation are not independent jobs.'])
    if cp is not None:
        props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        report['hardware']['gpu']=dict(name=props['name'].decode() if isinstance(props['name'],bytes) else props['name'],
            total_memory_bytes=props['totalGlobalMem'],cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion())
    cases=[dict(backend=b,workers=w,tile=t,query_count=q,strategy=s) for b in backends
           for w in (workers if b=='numpy' else (1,)) for t in tiles for q in query_counts for s in strategies]
    for case in cases:
        report['records'].append(dict(**case,queries=[[(s-1)*(i+1)+1 for s in sizes] for i in range(case['query_count'])],samples=[],warmup_samples=[]))
    output.parent.mkdir(parents=True,exist_ok=True);_write_json(partial,report)
    rng=np.random.default_rng(seed)
    for iteration in range(warmup+repeat):
        order=rng.permutation(len(cases)).tolist();report['execution_order'].append(order)
        for index in order:
            case=cases[index];record=report['records'][index]
            print(f'iteration={iteration} case={case}',flush=True)
            if case['backend']=='cupy':
                cp.cuda.get_current_stream().synchronize();cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            with tempfile.TemporaryDirectory(dir=work,prefix='publication-') as folder:
                folder=Path(folder)
                with rasterio.Env(GDAL_CACHEMAX=128*1024*1024),MemoryMonitor(cp if case['backend']=='cupy' else None) as monitor:
                    started=time.perf_counter()
                    prepared=source
                    prep_started=started
                    if item is not None:
                        prepared=folder/'indices.tif';process_worldview(item,prepared,backend='numpy')
                    native_seconds=time.perf_counter()-prep_started if item is not None else 0.
                    outputs,packing=pipeline(prepared,folder,record['queries'],backend=case['backend'],workers=case['workers'],tile=case['tile'],strategy=case['strategy'])
                    elapsed=time.perf_counter()-started
                check=validate_outputs(prepared,outputs,record['queries'],case['tile'])
                if item is not None:validate_worldview(item,prepared)
                if packing:RasterCache(folder/'cache',verify=True)
                sample=dict(total_seconds=elapsed,native_preparation_seconds=native_seconds,packing=packing,
                            memory=monitor.result(),validation=check,output_bytes=sum(p.stat().st_size for p in outputs))
                if monitor.error:raise RuntimeError('Memory sampling failed: '+monitor.error)
                record['warmup_samples' if iteration<warmup else 'samples'].append(sample)
            _write_json(partial,report)
    for record in report['records']:
        record.update(_summary([s['total_seconds'] for s in record['samples']]))
        record['correctness_passed']=True
    report['status']='complete';_write_json(output,report);partial.unlink()
    rows=[]
    for record in report['records']:
        cpus=[r for r in report['records'] if r['backend']=='numpy' and r['query_count']==record['query_count']]
        best=min((r['median_seconds'] for r in cpus),default=None)
        rows.append(dict(**{k:record[k] for k in cases[0]},median_seconds=record['median_seconds'],p95_seconds=record['p95_seconds'],
                         best_tested_cpu_over_case=best/record['median_seconds'] if best else '',
                         max_sampled_rss_bytes=max(s['memory']['process_rss_sampled_peak_bytes'] for s in record['samples']),
                         max_sampled_device_used_bytes=max(s['memory']['device_used_sampled_peak_bytes'] for s in record['samples']) if record['backend']=='cupy' else ''))
    with output.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--data-root',default='data');p.add_argument('--work-root',default='data/publication')
    p.add_argument('--backends',nargs='+',default=['numpy','cupy'])
    p.add_argument('--workers',nargs='+',type=int,default=sorted({1,min(8,available_cpus())}|{n for n in (2,4) if n<=available_cpus()}))
    p.add_argument('--tiles',nargs='+',type=int,default=[1024]);p.add_argument('--query-counts',nargs='+',type=int,default=[1,3])
    p.add_argument('--strategies',nargs='+',default=['stream','reuse','packed']);p.add_argument('--sizes',nargs='+',type=int,default=[15,31])
    p.add_argument('--repeat',type=int,default=5);p.add_argument('--warmup',type=int,default=1);p.add_argument('--seed',type=int,default=731)
    p.add_argument('--storage-label',default='unspecified')
    p.add_argument('--allow-dirty',action='store_true');run(**vars(p.parse_args()))


if __name__=='__main__':main()
