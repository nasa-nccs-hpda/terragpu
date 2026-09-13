"""Controlled scene-repetition scaling; synthetic spatial extent, not new imagery."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import rasterio
from rasterio.windows import Window

from .datasets import fetch_worldview_sample, sha256, _write_json
from .publication_benchmark import run, available_cpus
from .worldview import process_worldview


def repeat_scene(source, destination, scale, tile=512):
    """Repeat physical pixel values in a scale-by-scale mosaic with bounded RAM.

    The affine grid is extended for I/O testing only. Repeated content and seams
    are synthetic; no new geographic observations are represented.
    """
    if type(scale) is not int or scale<1 or tile<1:raise ValueError('Positive scale and tile required')
    source,destination=Path(source),Path(destination)
    if destination.exists():raise FileExistsError(destination)
    try:
        with rasterio.Env(GDAL_CACHEMAX=128*1024*1024),rasterio.open(source) as src:
            if src.crs is None:raise ValueError('Georeferenced source required')
            profile=dict(driver='GTiff',width=src.width*scale,height=src.height*scale,count=src.count,
                         crs=src.crs,transform=src.transform,dtype='float32',nodata=np.nan,
                         tiled=True,blockxsize=256,blockysize=256,compress='LZW',BIGTIFF='YES',NUM_THREADS='1')
            with rasterio.open(destination,'w',**profile) as dst:
                dst.descriptions=src.descriptions
                dst.update_tags(terragpu_synthetic_extent='true',repetition_scale=str(scale),
                                interpretation='Repeated source pixels; synthetic seams and extent; not independent imagery')
                for row in range(0,src.height,tile):
                    for col in range(0,src.width,tile):
                        h,w=min(tile,src.height-row),min(tile,src.width-col)
                        values=src.read(window=Window(col,row,w,h),masked=True,out_dtype='float32').filled(np.nan)
                        values=values*np.asarray(src.scales,dtype='float32')[:,None,None]+np.asarray(src.offsets,dtype='float32')[:,None,None]
                        for y in range(scale):
                            for x in range(scale):
                                dst.write(values,window=Window(x*src.width+col,y*src.height+row,w,h))
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def scaling(output,*,source=None,data_root='data',work_root='data/scaling',scales=(1,2,4),
            workers=(1,4,8),tiles=(1024,2048),query_counts=(1,3),backends=('numpy','cupy'),
            repeat=5,warmup=1,seed=731,storage_label='unspecified',allow_dirty=False):
    if not scales or any(type(s) is not int or s<1 for s in scales) or len(set(scales))!=len(scales):
        raise ValueError('Scales must be unique positive integers')
    if max(workers)>available_cpus():raise ValueError('CPU workers exceed allocation/affinity')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    work=Path(work_root);work.mkdir(parents=True,exist_ok=True)
    manifest=dict(status='running',design='Controlled repeated-scene scaling; synthetic extent and seams',
                  preparation_timed=False,scales=list(scales),runs=[],
                  notes=['WorldView QA/index preparation and mosaic construction are outside the timed spatial pipeline.',
                         'Each trial includes reading the physical generated GeoTIFF, spatial analyses and compressed output close.',
                         'Repeated content does not establish performance on independent larger scenes or cold storage.',
                         'Larger scenes increase total work; larger tiles increase per-tile GPU work. No GDS claim.'])
    _write_json(output/'scaling.json',manifest)
    rows=[]
    with tempfile.TemporaryDirectory(dir=work,prefix='scaling-') as directory:
        directory=Path(directory)
        manifest['source_kind']='WorldView QA-masked NDVI/NDWI' if source is None else 'User GeoTIFF physical values'
        if source is None:
            item=fetch_worldview_sample(Path(data_root)/'worldview-example')
            metadata=json.loads(item.read_text())
            paths=[item]+[item.parent/metadata['assets'][k]['href'] for k in ('ms_analytic','cloud-mask-raster','ms-saturation-mask-raster')]
            manifest['native_inputs']=[dict(name=p.name,sha256=sha256(p)) for p in paths]
            source=directory/'worldview-indices.tif';process_worldview(item,source,backend='numpy')
        source=Path(source)
        with rasterio.open(source) as src:count,height,width=src.count,src.height,src.width
        manifest['original_input']=dict(name=source.name,sha256=sha256(source),shape=[count,height,width])
        for scale in scales:
            # Conservative uncompressed input + all output queries + 1 GiB headroom.
            required=count*height*width*scale**2*4*(1+4*max(query_counts))+2**30
            free=shutil.disk_usage(work).free
            if free<required:raise RuntimeError(f'Scale {scale} needs approximately {required/2**30:.1f} GiB free scratch; available {free/2**30:.1f} GiB')
            print(f'Preparing synthetic scale={scale}, shape={height*scale}x{width*scale}, pixels={height*width*scale**2}',flush=True)
            generated=repeat_scene(source,directory/f'repeated-scene-{scale}.tif',scale)
            path=output/f'scale-{scale}.json'
            report=run(path,source=generated,work_root=directory,backends=backends,workers=workers,tiles=tiles,
                       query_counts=query_counts,strategies=['reuse'],repeat=repeat,warmup=warmup,seed=seed,
                       storage_label=storage_label,allow_dirty=allow_dirty)
            manifest['runs'].append(dict(scale=scale,pixel_multiplier=scale**2,report=path.name,
                                         input_bytes=generated.stat().st_size,decoded_input_bytes=count*height*width*scale**2*4))
            for record in report['records']:
                pixels=height*width*scale**2
                rows.append(dict(scale=scale,pixels=pixels,decoded_input_bytes=count*pixels*4,
                                 **{k:record[k] for k in ('backend','workers','tile','query_count','median_seconds')},
                                 pixels_per_second=pixels/record['median_seconds'],
                                 peak_rss_bytes=max(s['memory']['process_rss_sampled_peak_bytes'] for s in record['samples']),
                                 peak_device_used_bytes=max(s['memory']['device_used_sampled_peak_bytes'] for s in record['samples']) if record['backend']=='cupy' else None))
            with (output/'scaling.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            _write_json(output/'scaling.json',manifest)
            generated.unlink()
    manifest['status']='complete';_write_json(output/'scaling.json',manifest)
    return manifest


def plot_scaling(output,allow_dirty=False):
    """Select best tested CPU/GPU per size/query, retaining all raw configurations."""
    from .publication_figures import summarize
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);manifest=json.loads((output/'scaling.json').read_text())
    if manifest['status']!='complete':raise ValueError('Scaling sweep incomplete')
    reports=[]
    for entry in manifest['runs']:
        _,validated=summarize([output/entry['report']],allow_dirty=allow_dirty)
        reports.extend(validated)
    queries=sorted({r['query_count'] for report in reports for r in report['records']})
    fig,axes=plt.subplots(1,len(queries),figsize=(6*len(queries),4.5),squeeze=False)
    selected=[]
    for ax,q in zip(axes[0],queries):
        for backend in ('numpy','cupy'):
            points=[]
            for report in reports:
                cases=[r for r in report['records'] if r['query_count']==q and r['backend']==backend]
                if not cases:continue
                best=min(cases,key=lambda r:r['median_seconds'])
                pixels=report['source_shape'][1]*report['source_shape'][2]
                points.append((pixels/1e6,best['median_seconds']))
                selected.append(dict(backend=backend,query_count=q,pixels=pixels,
                                     workers=best['workers'],tile=best['tile'],median_seconds=best['median_seconds']))
            if points:
                points.sort();ax.plot(*zip(*points),marker='o',label=f'Best tested {backend}')
        ax.set(xlabel='Input pixels (millions)',ylabel='Read + analyses + output time (s)',
               title=f'{q} analyses; repeated-scene stress test',xscale='log',yscale='log')
        ax.grid(alpha=.2);ax.legend()
    fig.tight_layout()
    for extension in ('png','pdf'):fig.savefig(output/f'scaling.{extension}',dpi=180)
    plt.close(fig)
    _write_json(output/'best-tested.json',selected)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--source',type=Path)
    p.add_argument('--data-root',default='data');p.add_argument('--work-root',default='data/scaling')
    p.add_argument('--scales',nargs='+',type=int,default=[1,2,4])
    p.add_argument('--workers',nargs='+',type=int,default=sorted({1,min(8,available_cpus())}|{n for n in (4,) if n<=available_cpus()}))
    p.add_argument('--tiles',nargs='+',type=int,default=[1024,2048])
    p.add_argument('--query-counts',nargs='+',type=int,default=[1,3])
    p.add_argument('--backends',nargs='+',default=['numpy','cupy'])
    p.add_argument('--repeat',type=int,default=5);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--seed',type=int,default=731);p.add_argument('--storage-label',default='unspecified')
    p.add_argument('--allow-dirty',action='store_true')
    args=vars(p.parse_args());scaling(**args)
    plot_scaling(args['output'],allow_dirty=args['allow_dirty'])


if __name__=='__main__':main()
