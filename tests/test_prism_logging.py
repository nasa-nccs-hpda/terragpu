"""A redirected setup failure must still be visible in the Slurm console."""
import os
from pathlib import Path
import subprocess


def test_runner_reports_failure_and_keeps_exit_status(tmp_path):
    commands = tmp_path/'commands'
    commands.mkdir()
    python = commands/'python'
    python.write_text('#!/bin/sh\necho "synthetic environment failure" >&2\nexit 17\n')
    python.chmod(0o755)
    out = tmp_path/'result'
    script = Path(__file__).resolve().parents[1]/'scripts/run_prism_paper.sh'
    result = subprocess.run(['bash', str(script), str(out), 'cpu'], cwd=tmp_path,
                            env=dict(os.environ, PATH=f'{commands}:{os.environ["PATH"]}'),
                            capture_output=True, text=True)
    assert result.returncode == 17
    assert 'synthetic environment failure' in result.stdout + result.stderr
    assert 'FAILED at stage: Python environment' in result.stdout + result.stderr
    assert (out/'run-status.txt').read_text().strip() == '17'
    assert 'synthetic environment failure' in (out/'pip-freeze.txt').read_text()
