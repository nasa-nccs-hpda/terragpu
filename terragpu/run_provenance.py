"""Minimal allocation provenance; never serialize the full process environment."""
import os
import uuid


def cpu_resources():
    """Record visible CPUs and scheduler hints without inferring an allocation.

    A shell can inherit stale scheduler variables or affinity wider than its
    allocation. Keep per-node and per-task counts distinct, in their original
    form (job CPU lists can contain Slurm repetition notation).
    """
    affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    return dict(logical_cpu_count=os.cpu_count(),affinity=affinity,
                scheduler_environment={name:os.environ.get(name) for name in (
                    'SLURM_CPUS_PER_TASK','SLURM_CPUS_ON_NODE','SLURM_JOB_CPUS_PER_NODE',
                    'SLURM_NTASKS','SLURM_NTASKS_PER_NODE')},
                allocation_verified=False,
                interpretation='Affinity describes accessible CPUs, not allocated CPUs. Scheduler environment is recorded as evidence only; per-node counts are not per-task worker limits.')


def execution_metadata():
    return dict(run_id=str(uuid.uuid4()), cpu_resources=cpu_resources(), scheduler={
        field: os.environ.get(variable) for field, variable in (
            ('job_id', 'SLURM_JOB_ID'), ('step_id', 'SLURM_STEP_ID'),
            ('array_job_id', 'SLURM_ARRAY_JOB_ID'), ('array_task_id', 'SLURM_ARRAY_TASK_ID'))})


def require_distinct_jobs(reports):
    """Check distinct reported Slurm allocations, not statistical independence."""
    if len(reports)<2:raise ValueError('At least two reports are required for distinct-job checking')
    jobs=set();runs=set()
    for report in reports:
        execution=report.get('execution',{})
        try:run_id=str(uuid.UUID(execution['run_id']))
        except (KeyError,ValueError,TypeError,AttributeError) as error:
            raise ValueError('Missing or invalid run identifier') from error
        job=execution.get('scheduler',{}).get('job_id')
        if not isinstance(job,str) or not job.isascii() or not job.isdigit() or int(job)<1:
            raise ValueError('Missing or invalid Slurm job identifier')
        job=str(int(job))
        # Different srun steps in the same allocation are not separate jobs.
        if run_id in runs or job in jobs:raise ValueError('Repeated run or Slurm allocation')
        runs.add(run_id);jobs.add(job)
