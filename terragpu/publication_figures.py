"""Generate native-pipeline figures; aggregate run medians, never pool repetitions."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

KEYS=('backend','workers','tile','query_count','strategy')


def case_key(record):return tuple(record[k] for k in KEYS)


def summarize(paths,allow_dirty=False):
    reports=[];hashes=set();matrix=None;identity=None
    for path in map(Path,paths):
        if path.with_suffix('.partial.json').exists():raise ValueError('Partial report remains')
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest in hashes:raise ValueError('Duplicate report')
        hashes.add(digest);report=json.loads(raw)
        if report.get('status')!='complete' or report.get('schema_version')!=1:
            raise ValueError('Incomplete or unsupported report')
        if report.get('git_dirty') is not False and not allow_dirty:raise ValueError('Dirty source report')
        signatures={case_key(r) for r in report['records']}
        if not signatures or len(signatures)!=len(report['records']):raise ValueError('Duplicate or missing cases')
        current={k:report.get(k) for k in ('git_commit','inputs','source_shape','source_kind','hardware','packages','threads','storage_label')}
        current['queries']={str(case_key(r)):r['queries'] for r in report['records']}
        if identity is not None and (identity!=current or matrix!=signatures):
            raise ValueError('Runs differ in source, hardware, environment, storage label or case matrix')
        identity=current;matrix=signatures
        for record in report['records']:
            if record.get('correctness_passed') is not True or len(record['samples'])!=report['repeat']:
                raise ValueError('Incomplete or unvalidated samples')
            values=[s['total_seconds'] for s in record['samples']]
            if not values or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in values):
                raise ValueError('Invalid timings')
            if not math.isclose(statistics.median(values),record['median_seconds'],rel_tol=1e-9):
                raise ValueError('Summary disagrees with raw timings')
            for sample in record['samples']:
                if sample['memory']['sampling_error'] is not None:raise ValueError('Memory sampling failed')
                if sample['validation']['metadata_and_masks_match'] is not True:raise ValueError('Validation failed')
        reports.append(report)
    if not reports:raise ValueError('No reports')
    if len(reports)>1 and identity['storage_label'] in (None,'unspecified'):
        raise ValueError('Give benchmark runs an explicit storage label before aggregation')
    rows=[]
    for key in sorted(matrix):
        runs=[next(r for r in report['records'] if case_key(r)==key) for report in reports]
        medians=[statistics.median(s['total_seconds'] for s in r['samples']) for r in runs]
        rss=[max(s['memory']['process_rss_sampled_peak_bytes'] for s in r['samples']) for r in runs]
        rows.append(dict(zip(KEYS,key),runs=len(runs),median_of_run_medians_seconds=statistics.median(medians),
                         min_run_median_seconds=min(medians),max_run_median_seconds=max(medians),
                         max_sampled_process_rss_bytes=max(rss),run_medians_seconds=medians))
    return rows,reports


def plot(paths,output,allow_dirty=False):
    rows,reports=summarize(paths,allow_dirty)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    query_counts=sorted({r['query_count'] for r in rows})
    for metric,ylabel,name in [('median_of_run_medians_seconds','Complete pipeline time (s)','pipeline-times'),
                              ('max_sampled_process_rss_bytes','Sampled process RSS (GiB)','process-memory')]:
        fig,axes=plt.subplots(len(query_counts),1,figsize=(max(9,len(rows)/len(query_counts)*.65),4.2*len(query_counts)),squeeze=False)
        for ax,q in zip(axes[:,0],query_counts):
            group=[r for r in rows if r['query_count']==q]
            labels=[f"{r['backend']} / {r['workers']}w\n{r['strategy']} / {r['tile']}px" for r in group]
            values=[r[metric]/(2**30 if name=='process-memory' else 1) for r in group]
            ax.bar(range(len(group)),values,color=['#286995' if r['backend']=='numpy' else '#bb6429' for r in group])
            if name=='pipeline-times':
                for i,r in enumerate(group):
                    ax.scatter([i]*len(r['run_medians_seconds']),r['run_medians_seconds'],color='black',s=15,zorder=3)
            ax.set_xticks(range(len(group)),labels,rotation=35,ha='right');ax.set_ylabel(ylabel)
            ax.set_title(f'{q} distinct analyses; {len(reports)} run(s)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.tight_layout()
        for extension in ('png','pdf'):fig.savefig(output/f'{name}.{extension}',dpi=180)
        plt.close(fig)
    with (output/'run-summary.csv').open('w',newline='') as stream:
        fields=[k for k in rows[0] if k!='run_medians_seconds']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    provenance=dict(git_commit=reports[0]['git_commit'], storage_label=reports[0].get('storage_label'),
                    input_manifests=reports[0]['inputs'],
                    report_sha256=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths],
                    aggregation='median of run medians; independent job provenance must be checked separately')
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (output/'figure-notes.txt').write_text(
        'Bars summarize whole-pipeline trials including compressed GeoTIFF close. Points are run medians.\n'
        'Within-run repetitions are not pooled across runs. These plots do not establish job independence.\n'
        'Memory is sampled process RSS, includes retained allocator/library memory, and can miss transient peaks.\n'
        'GPU device-wide memory and allocator reservation are in raw JSON; neither equals process peak VRAM.\n'
        'No GDS, disk-cache flush, fsync durability or larger-than-VRAM qualification is implied.\n')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('paths',nargs='+',type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--allow-dirty',action='store_true')
    plot(**vars(p.parse_args()))


if __name__=='__main__':main()
