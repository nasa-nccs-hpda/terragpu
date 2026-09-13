import numpy as np
import pytest
import rasterio

pytest.importorskip('scipy')
pytest.importorskip('psutil')
from terragpu.publication_benchmark import pipeline,validate_outputs,run,MemoryMonitor,bounded_map,StageTimings
from terragpu.benchmark_gpu_io import reference
from test_gpu_io import scene


@pytest.mark.parametrize('profile',[False,True])
@pytest.mark.parametrize('strategy',['stream','reuse','packed'])
@pytest.mark.parametrize('backend,workers',[('numpy',1),('numpy',2),pytest.param('cupy',1,marks=pytest.mark.gpu)])
def test_native_pipeline_matches_whole_image_reference(tmp_path,strategy,backend,workers,profile):
    if backend=='cupy':pytest.importorskip('cupy')
    source=tmp_path/'input.tif';expected_input=scene(source)[None]
    queries=[[3,5],[5,7]]
    stages=StageTimings(profile)
    outputs,packing=pipeline(source,tmp_path,queries,backend=backend,workers=workers,tile=4,strategy=strategy,stages=stages)
    if profile:
        tiles=((expected_input.shape[1]+3)//4)*((expected_input.shape[2]+3)//4)
        assert stages.values['read_decode']['calls']==tiles*(len(queries) if strategy=='stream' else 1)
        assert stages.values['compute']['calls']==tiles*len(queries)
        assert stages.values['write']['calls']==tiles*len(queries)
        assert stages.values['output_close']['calls']==1
        assert all(v['seconds']>=0 for v in stages.values.values())
        if backend=='cupy':
            assert stages.values['host_to_device']['calls']==stages.values['read_decode']['calls']
            assert stages.values['device_to_host']['calls']==tiles*len(queries)
    else:assert not stages.values
    assert bool(packing)==(strategy=='packed')
    for path,query in zip(outputs,queries):
        with rasterio.open(path) as dst:
            np.testing.assert_allclose(dst.read(),reference(expected_input,query),rtol=2e-5,atol=2e-6,equal_nan=True)
            assert dst.compression.value=='LZW'
    assert validate_outputs(source,outputs,queries,4)['finite_values']>0


def test_complete_native_cpu_experiment(tmp_path,monkeypatch):
    monkeypatch.setattr('terragpu.publication_benchmark.available_cpus',lambda:2)
    source=tmp_path/'input.tif';scene(source)
    report=run(tmp_path/'report.json',source=source,work_root=tmp_path/'work',backends=['numpy'],
               workers=[1,2],tiles=[8],query_counts=[1,2],sizes=[3,5],repeat=1,warmup=0,allow_dirty=True,profile_stages=True)
    assert len(report['records'])==12 and report['status']=='complete'
    assert len(report['execution_order'])==1
    for record in report['records']:
        sample=record['samples'][0]
        assert record['correctness_passed']
        assert sample['stage_timings']['compute']['calls']>0
        assert sample['memory']['process_rss_sampled_peak_bytes']>0
        assert sample['memory']['device_used_sampled_peak_bytes'] is None
        assert sample['memory']['sampling_error'] is None
        assert sample['output_bytes']>0 and sample['total_seconds']>0
        assert len(record['queries'])==record['query_count']
    assert not list((tmp_path/'work').iterdir())
    assert not (tmp_path/'report.partial.json').exists()
    assert (tmp_path/'report.csv').exists()
    with pytest.raises(FileExistsError):
        run(tmp_path/'report.json',source=source,backends=['numpy'],workers=[1],allow_dirty=True)


def test_worker_error_propagates_and_monitor_stops():
    def failure(value):raise RuntimeError('tile failed')
    with MemoryMonitor() as monitor:
        with pytest.raises(RuntimeError,match='tile failed'):
            list(bounded_map(failure,range(10),2))
    assert not monitor.thread.is_alive()


def test_cpu_allocation_enforced(tmp_path,monkeypatch):
    monkeypatch.setattr('terragpu.publication_benchmark.available_cpus',lambda:1)
    with pytest.raises(ValueError,match='allocation'):
        run(tmp_path/'report.json',backends=['numpy'],workers=[2],allow_dirty=True)
    assert not (tmp_path/'report.json').exists()


@pytest.mark.parametrize('fault',['missing','extra','duplicate','compression','dtype','nodata'])
def test_output_contract_rejects_incomplete_or_reencoded_products(tmp_path,fault):
    source=tmp_path/'input.tif';scene(source);queries=[[3],[5]]
    outputs,_=pipeline(source,tmp_path,queries,tile=8)
    assert validate_outputs(source,outputs,queries,8)['finite_values']>0
    if fault=='missing':outputs=outputs[:1]
    elif fault=='extra':outputs=outputs+[source]
    elif fault=='duplicate':outputs=[outputs[0],outputs[0]]
    else:
        with rasterio.open(outputs[0]) as original:
            values=original.read();profile=original.profile;descriptions=original.descriptions
        if fault=='compression':profile['compress']='NONE'
        elif fault=='dtype':profile['dtype']='float64'
        elif fault=='nodata':profile['nodata']=None
        changed=tmp_path/'changed.tif'
        with rasterio.open(changed,'w',**profile) as dst:
            dst.write(values);dst.descriptions=descriptions
        outputs=[changed,outputs[1]]
    with pytest.raises(AssertionError):validate_outputs(source,outputs,queries,8)


def test_stage_timings_synchronize_boundaries(monkeypatch):
    from types import SimpleNamespace
    events=[]
    stream=SimpleNamespace(synchronize=lambda:events.append('sync'))
    xp=SimpleNamespace(cuda=SimpleNamespace(get_current_stream=lambda:stream))
    ticks=iter([2.,5.])
    monkeypatch.setattr('terragpu.publication_benchmark.time.perf_counter',lambda:next(ticks))
    stages=StageTimings(True)
    with stages.measure('compute',xp):events.append('compute')
    assert events==['sync','compute','sync']
    assert stages.values=={'compute':{'seconds':3.,'calls':1}}
    with StageTimings().measure('compute',xp):events.append('unprofiled')
    assert events[-2:]==['sync','unprofiled']
