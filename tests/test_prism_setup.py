"""Exercise node architecture selection without installing an environment."""
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('machine,expected', [('x86_64','x86_64'), ('aarch64','aarch64'), ('arm64','aarch64')])
def test_setup_ignores_inherited_login_architecture(tmp_path, machine, expected):
    commands = tmp_path/'commands'
    commands.mkdir()
    uname = commands/'uname'
    uname.write_text(f'#!/bin/sh\necho {machine}\n')
    uname.chmod(0o755)
    for arch in ('x86_64','aarch64'):
        binary = tmp_path/f'uv-{arch}'/'uv'
        binary.parent.mkdir()
        binary.write_text(f'#!/bin/sh\necho {arch} "$@" >> "$CALL_LOG"\nif [ "$1" = venv ]; then mkdir -p "$5"; fi\n')
        binary.chmod(0o755)
    inherited = 'aarch64' if expected == 'x86_64' else 'x86_64'
    env = dict(os.environ, PATH=f'{commands}:{os.environ["PATH"]}',
               UV_BIN=str(tmp_path/f'uv-{inherited}'/'uv'),
               UV_PROJECT_ENVIRONMENT=f'.venv-{inherited}', CALL_LOG=str(tmp_path/'calls'))
    script = Path(__file__).resolve().parents[1]/'scripts/setup_prism_uv.sh'
    subprocess.run(['bash', str(script), 'cpu'], cwd=tmp_path, env=env, check=True, capture_output=True)
    lines = (tmp_path/'calls').read_text().splitlines()
    assert all(line.startswith(expected+' ') for line in lines)
    assert (tmp_path/f'.venv-prism-{expected}'/'requirements-resolved.txt').exists()
