"""Export plot-ready summary and raw timing CSVs from one completed run."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def export(directory, require_products=False, require_io=False):
    directory=Path(directory)
    if list(directory.glob('*.partial.json')):
        raise ValueError('Incomplete benchmark: partial JSON remains')
    suite=json.loads((directory/'suite.json').read_text())
    records=list(suite['records'])
    products_path=directory/'products.json'
    if require_products or products_path.exists():
        products=json.loads(products_path.read_text())
        if products['git_commit'] != suite['git_commit']:
            raise ValueError('Product and resident suites use different source revisions')
        if set(products['requested_backends']) != {r['backend'] for r in suite['records']}:
            raise ValueError('Product and resident suites use different backends')
        expected={(w,b) for w in ('hls_ndvi_io','pace_spectral_mean_io','viirs_spectral_mean_io')
                  for b in products['requested_backends']}
        if {(r['workload'],r['backend']) for r in products['records']} != expected:
            raise ValueError('Incomplete product/backend matrix')
        records.extend(products['records'])
    for path in sorted(directory.glob('ndvi-*.json')):
        records.append(json.loads(path.read_text()))
    io_paths=[directory/f'io-{b}.json' for b in ('numpy','cupy') if (directory/f'io-{b}.json').exists()]
    io_devices=set()
    for path in io_paths:
        report=json.loads(path.read_text())
        if report['device'] not in ('numpy','cupy') or path.stem != 'io-'+report['device']:
            raise ValueError('I/O filename does not match its device')
        if report['git_commit'] != suite['git_commit']:
            raise ValueError('I/O and resident suites use different source revisions')
        if report['device'] in io_devices:
            raise ValueError('Duplicate I/O device report')
        io_devices.add(report['device'])
        if set(report['samples_seconds']) != {'streaming','dask'}:
            raise ValueError('Incomplete I/O execution strategy comparison')
        for mode,values in report['samples_seconds'].items():
            records.append(dict(workload='synthetic_geotiff_ndvi_'+mode,backend=report['device'],
                                samples_seconds=values,correctness_passed=report['correctness_passed'],
                                shape=report['shape'],scope=report['scope'],tile_size=report['tile_size'],
                                input_array_sha256=report['input_array_sha256']))
    if require_io and io_devices != {r['backend'] for r in suite['records']}:
        raise ValueError('Incomplete I/O backend matrix')
    summaries=[];samples=[];seen={}
    for record in records:
        values=record['samples_seconds']
        key=(record['workload'],record['backend'])
        if key in seen or record['correctness_passed'] is not True:
            raise ValueError(f'Duplicate or unvalidated record: {key}')
        if not values or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0 for v in values):
            raise ValueError(f'Invalid timing samples: {key}')
        seen[key]=record
        summaries.append(dict(workload=record['workload'],backend=record['backend'],
                              median_seconds=float(np.median(values)),p95_seconds=float(np.percentile(values,95)),
                              repetitions=len(values),correctness_passed=record['correctness_passed'],
                              scope=record.get('scope',record.get('timing_scope','')),
                              valid_pixels_per_second=record.get('valid_pixels_per_second','')))
        samples.extend(dict(workload=record['workload'],backend=record['backend'],repetition=i,seconds=v) for i,v in enumerate(values))
    # Same-run CPU/GPU ratios; never pool different scenes or hardware silently.
    ratios=[]
    for cpu in summaries:
        if cpu['backend']!='numpy':continue
        for gpu in summaries:
            if gpu['workload']==cpu['workload'] and gpu['backend']=='cupy':
                a,b=seen[(cpu['workload'],'numpy')],seen[(gpu['workload'],'cupy')]
                for field in ('shape','scope','timing_scope','input_sha256','input_array_sha256','tile_size'):
                    if a.get(field)!=b.get(field):
                        raise ValueError(f'CPU/GPU comparison differs in {field}')
                ratios.append(dict(workload=cpu['workload'],cpu_over_gpu=cpu['median_seconds']/gpu['median_seconds']))
    if not summaries:
        raise ValueError('No benchmark records')
    for name,rows in [('summary.csv',summaries),('samples.csv',samples)]:
        with (directory/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (directory/'speedups.json').write_text(json.dumps(ratios,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory')
    parser.add_argument('--require-products',action='store_true')
    parser.add_argument('--require-io',action='store_true')
    export(**vars(parser.parse_args()))
