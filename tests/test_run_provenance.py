import copy
import uuid
import pytest
from terragpu.run_provenance import execution_metadata,require_distinct_jobs


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
