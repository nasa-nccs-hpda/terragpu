import copy
import uuid
import pytest
from terragpu.run_provenance import execution_metadata,require_distinct_jobs,cpu_resources


def test_allowlisted_environment_and_unique_runs(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','101')
    monkeypatch.setenv('SLURM_STEP_ID','0')
    monkeypatch.delenv('SLURM_ARRAY_JOB_ID',raising=False)
    monkeypatch.setenv('EARTHDATA_TOKEN','fixture-secret-never-export')
    first=execution_metadata();second=execution_metadata()
    assert first['run_id']!=second['run_id']
    assert uuid.UUID(first['run_id'])
    assert first['scheduler']['job_id']=='101'
    assert first['scheduler']['array_job_id'] is None
    assert set(first['scheduler'])=={'job_id','step_id','array_job_id','array_task_id'}
    assert 'fixture-secret-never-export' not in str(first)


def report(job,step='0'):
    return dict(execution=dict(run_id=str(uuid.uuid4()),scheduler=dict(job_id=job,step_id=step)))


def test_distinct_allocations_not_steps():
    require_distinct_jobs([report('10'),report('11')])
    with pytest.raises(ValueError,match='allocation'):
        require_distinct_jobs([report('10','0'),report('10','1')])
    first=report('10');second=copy.deepcopy(first);second['execution']['scheduler']['job_id']='11'
    with pytest.raises(ValueError,match='Repeated'):require_distinct_jobs([first,second])


@pytest.mark.parametrize('bad',[{},report(None),report(''),report('abc'),report('0')])
def test_missing_provenance_cannot_establish_distinct_jobs(bad):
    with pytest.raises(ValueError):require_distinct_jobs([report('10'),bad])
    with pytest.raises(ValueError):require_distinct_jobs([report('10')])


def test_resource_evidence_does_not_equate_visible_and_allocated_cpus(monkeypatch):
    monkeypatch.setattr('terragpu.run_provenance.os.cpu_count',lambda:72)
    monkeypatch.setattr('terragpu.run_provenance.os.sched_getaffinity',lambda pid:{2,3,4,5},raising=False)
    monkeypatch.delenv('SLURM_CPUS_PER_TASK',raising=False)
    monkeypatch.setenv('SLURM_JOB_CPUS_PER_NODE','8(x2)')
    monkeypatch.setenv('SLURM_CPUS_ON_NODE','8')
    monkeypatch.setenv('EARTHDATA_TOKEN','do-not-record')
    resources=cpu_resources()
    assert resources['logical_cpu_count']==72
    assert resources['affinity']==[2,3,4,5]
    assert resources['scheduler_environment']['SLURM_CPUS_PER_TASK'] is None
    assert resources['scheduler_environment']['SLURM_JOB_CPUS_PER_NODE']=='8(x2)'
    assert resources['allocation_verified'] is False
    assert 'do-not-record' not in str(resources)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','4')
    assert cpu_resources()['scheduler_environment']['SLURM_CPUS_PER_TASK']=='4'
    assert cpu_resources()['allocation_verified'] is False


def test_resource_evidence_without_affinity_api(monkeypatch):
    monkeypatch.delattr('terragpu.run_provenance.os.sched_getaffinity',raising=False)
    monkeypatch.setattr('terragpu.run_provenance.os.cpu_count',lambda:None)
    resources=cpu_resources()
    assert resources['affinity'] is None
    assert resources['logical_cpu_count'] is None
