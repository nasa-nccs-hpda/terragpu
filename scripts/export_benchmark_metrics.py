"""Export plot-ready summary and raw timing CSVs from one completed run."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def export(directory):
    directory=Path(directory)
    suite=json.loads((directory/'suite.json').read_text())
    records=list(suite['records'])
    for path in sorted(directory.glob('ndvi-*.json')):
        records.append(json.loads(path.read_text()))
    summaries=[];samples=[]
    for record in records:
        values=record['samples_seconds']
        summaries.append(dict(workload=record['workload'],backend=record['backend'],
                              median_seconds=float(np.median(values)),p95_seconds=float(np.percentile(values,95)),
                              repetitions=len(values),correctness_passed=record['correctness_passed']))
        samples.extend(dict(workload=record['workload'],backend=record['backend'],repetition=i,seconds=v) for i,v in enumerate(values))
    for name,rows in [('summary.csv',summaries),('samples.csv',samples)]:
        with (directory/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Same-run CPU/GPU ratios; never pool different scenes or hardware silently.
    ratios=[]
    for cpu in summaries:
        if cpu['backend']!='numpy':continue
        for gpu in summaries:
            if gpu['workload']==cpu['workload'] and gpu['backend']=='cupy':
                ratios.append(dict(workload=cpu['workload'],cpu_over_gpu=cpu['median_seconds']/gpu['median_seconds']))
    (directory/'speedups.json').write_text(json.dumps(ratios,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory')
    export(parser.parse_args().directory)
