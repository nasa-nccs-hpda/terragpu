import copy
import json
import pytest
from terragpu.publication_figures import summarize,plot


def report():
    return dict(schema_version=1,status='complete',git_dirty=False,git_commit='test',inputs=['test'],source_shape=[1,4,4],
        source_kind='fixture',hardware={},packages={},threads={},storage_label='nvme',repeat=2,
        records=[dict(backend='numpy',workers=1,tile=4,query_count=1,strategy='stream',queries=[[3]],
            correctness_passed=True,median_seconds=3.,samples=[dict(total_seconds=v,
                memory=dict(sampling_error=None,sampling_interval_seconds=.01,process_rss_start_bytes=512,
                            process_rss_sampled_peak_bytes=1024,device_used_start_bytes=None,
                            device_used_sampled_peak_bytes=None,cupy_pool_reserved_sampled_peak_bytes=None),
                validation=dict(metadata_and_masks_match=True,finite_values=16,max_absolute_error=0.,rtol=2e-5,atol=2e-6)) for v in [2.,4.]])])


def test_run_medians_not_pooled_and_figures(tmp_path):
    pytest.importorskip('matplotlib')
    first=report();second=copy.deepcopy(first);second['repeat']=3
    second['records'][0]['samples']=[dict(first['records'][0]['samples'][0],total_seconds=v) for v in [10.,20.,30.]]
    second['records'][0]['median_seconds']=20.
    paths=[tmp_path/'a.json',tmp_path/'b.json']
    for path,value in zip(paths,[first,second]):path.write_text(json.dumps(value))
    rows,_=summarize(paths)
    assert rows[0]['run_medians_seconds']==[3.,20.]
    assert rows[0]['median_of_run_medians_seconds']==11.5
    dest=plot(paths,tmp_path/'figures')
    assert (dest/'pipeline-times.png').stat().st_size>1000
    assert (dest/'process-memory.pdf').stat().st_size>1000
    with pytest.raises(FileExistsError):plot(paths,dest)
    with pytest.raises(ValueError,match='Duplicate'):summarize([paths[0],paths[0]])


@pytest.mark.parametrize('fault',['dirty','incomplete','median','validation','negative','matrix','storage','source'])
def test_reject_invalid_comparisons(tmp_path,fault):
    first=report();second=copy.deepcopy(first)
    if fault=='dirty':second['git_dirty']=True
    elif fault=='incomplete':second['status']='running'
    elif fault=='median':second['records'][0]['median_seconds']=99.
    elif fault=='validation':second['records'][0]['correctness_passed']=False
    elif fault=='negative':second['records'][0]['samples'][0]['total_seconds']=-1.
    elif fault=='matrix':second['records'][0]['workers']=2
    elif fault=='storage':second['storage_label']='panfs'
    elif fault=='source':second['inputs']=['different']
    paths=[tmp_path/'a.json',tmp_path/'b.json']
    for path,value in zip(paths,[first,second]):path.write_text(json.dumps(value))
    with pytest.raises(ValueError):summarize(paths)


@pytest.mark.parametrize('field,value',[
    ('process_rss_sampled_peak_bytes',float('nan')),
    ('process_rss_sampled_peak_bytes',-1),
    ('process_rss_sampled_peak_bytes',True),
    ('process_rss_sampled_peak_bytes',256),
    ('sampling_interval_seconds',0),
    ('device_used_sampled_peak_bytes',1024),
])
def test_reject_bad_memory_before_plot_output(tmp_path,field,value):
    data=report();data['records'][0]['samples'][0]['memory'][field]=value
    path=tmp_path/'input.json';path.write_text(json.dumps(data))
    with pytest.raises(ValueError):plot([path],tmp_path/'figures')
    assert not (tmp_path/'figures').exists()


def test_gpu_measurements_and_numerical_evidence():
    from terragpu.publication_figures import validate_sample
    sample=report()['records'][0]['samples'][0]
    with pytest.raises(ValueError,match='GPU memory'):validate_sample(sample,'cupy')
    sample['memory'].update(device_used_start_bytes=100,device_used_sampled_peak_bytes=200,
                            cupy_pool_reserved_sampled_peak_bytes=80)
    validate_sample(sample,'cupy')
    sample['memory']['device_used_start_bytes']=300
    with pytest.raises(ValueError,match='GPU memory'):validate_sample(sample,'cupy')
    sample['memory']['device_used_start_bytes']=100
    sample['validation']['finite_values']=0
    with pytest.raises(ValueError,match='numerical'):validate_sample(sample,'cupy')
