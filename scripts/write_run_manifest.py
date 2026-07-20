#!/usr/bin/env python3
"""Write an auditable run manifest for RecSys26 reproductions.

The manifest is designed to make every run traceable:
- code: repo head + diff + optional patch hashes
- data: dataset audit json hash (source of truth)
- env: conda + key package versions + pip freeze hash
- outputs: prediction/metric file hashes + metric summary

This script is intentionally stdlib-only.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], cwd: Path | None = None, timeout_s: int = 30) -> dict[str, Any]:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
        return {
            'ok': p.returncode == 0,
            'returncode': p.returncode,
            'stdout': p.stdout.decode('utf-8', errors='replace'),
            'stderr': p.stderr.decode('utf-8', errors='replace'),
        }
    except Exception as e:  # pragma: no cover
        return {'ok': False, 'returncode': None, 'stdout': '', 'stderr': repr(e)}


def git_info(repo_dir: Path) -> dict[str, Any]:
    info: dict[str, Any] = {'repo_dir': str(repo_dir)}

    head = run(['git', 'rev-parse', 'HEAD'], cwd=repo_dir)
    if not head['ok']:
        info['git'] = {'ok': False, 'error': head['stderr'].strip()}
        return info

    info['git'] = {'ok': True, 'head': head['stdout'].strip()}

    status = run(['git', 'status', '--porcelain'], cwd=repo_dir)
    if status['ok']:
        lines = [ln for ln in status['stdout'].splitlines() if ln.strip()]
        info['git']['is_dirty'] = len(lines) > 0
        info['git']['status_porcelain'] = lines[:200]
    else:
        info['git']['is_dirty'] = None
        info['git']['status_porcelain'] = None

    diff_stat = run(['git', 'diff', '--stat'], cwd=repo_dir)
    if diff_stat['ok']:
        info['git']['diff_stat'] = diff_stat['stdout'].strip()
    else:
        info['git']['diff_stat'] = None

    diff_patch = run(['git', 'diff'], cwd=repo_dir, timeout_s=120)
    if diff_patch['ok']:
        patch_bytes = diff_patch['stdout'].encode('utf-8', errors='replace')
        info['git']['diff_patch_sha256'] = _sha256_bytes(patch_bytes)
        info['git']['diff_patch_bytes'] = len(patch_bytes)
    else:
        info['git']['diff_patch_sha256'] = None
        info['git']['diff_patch_bytes'] = None

    return info


def pip_freeze_sha256() -> dict[str, Any]:
    r = run([sys.executable, '-m', 'pip', 'freeze'], timeout_s=120)
    if not r['ok']:
        return {'ok': False, 'error': r['stderr'].strip()}
    b = r['stdout'].encode('utf-8', errors='replace')
    return {'ok': True, 'sha256': _sha256_bytes(b), 'bytes': len(b)}


def try_import_versions() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mod in ['torch', 'transformers', 'numpy']:
        try:
            m = __import__(mod)
            out[mod] = getattr(m, '__version__', None)
        except Exception:
            out[mod] = None
    return out


def maybe_json(path: Path) -> Any | None:
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def file_record(path_str: str) -> dict[str, Any] | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        return {'path': str(p), 'exists': False}
    rec: dict[str, Any] = {
        'path': str(p.resolve()),
        'exists': True,
        'bytes': p.stat().st_size,
        'sha256': sha256_file(p),
    }
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description='Write run_manifest.json')
    ap.add_argument('--out', required=True, help='Output JSON path')
    ap.add_argument('--method', required=True)
    ap.add_argument('--domain', required=True)
    ap.add_argument('--profile', default='')
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--repo_dir', required=True)
    ap.add_argument('--data_path', default='')
    ap.add_argument('--patch_file', action='append', default=[])
    ap.add_argument('--dataset_audit_json', default='')
    ap.add_argument('--train_cmd', default='')
    ap.add_argument('--export_cmd', default='')
    ap.add_argument('--eval_cmd', default='')
    ap.add_argument('--pred_file', default='')
    ap.add_argument('--metrics_file', default='')
    ap.add_argument('--pred_export_file', default='')
    ap.add_argument('--metrics_export_file', default='')
    ap.add_argument('--exit_code', type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    repo_dir = Path(args.repo_dir).resolve()

    manifest: dict[str, Any] = {
        'schema_version': 1,
        'created_utc': _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        'method': args.method,
        'domain': args.domain,
        'profile': args.profile,
        'run_dir': str(Path(args.run_dir).resolve()),
        'data_path': str(Path(args.data_path).resolve()) if args.data_path else '',
        'exit_code': int(args.exit_code),
        'host': socket.gethostname(),
        'user': os.getenv('USER', ''),
        'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES', ''),
        'python': {
            'executable': sys.executable,
            'version': sys.version,
        },
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
        },
        'conda': {
            'default_env': os.getenv('CONDA_DEFAULT_ENV', ''),
            'prefix': os.getenv('CONDA_PREFIX', ''),
        },
        'packages': try_import_versions(),
        'pip_freeze': pip_freeze_sha256(),
        'repo': git_info(repo_dir),
        'commands': {
            'train': args.train_cmd,
            'export': args.export_cmd,
            'eval': args.eval_cmd,
        },
        'files': {
            'pred_file': file_record(args.pred_file),
            'metrics_file': file_record(args.metrics_file),
            'pred_export_file': file_record(args.pred_export_file),
            'metrics_export_file': file_record(args.metrics_export_file),
        },
    }

    patches: list[dict[str, Any]] = []
    for pf in args.patch_file:
        p = Path(pf)
        if not p.exists():
            patches.append({'path': str(p), 'exists': False})
            continue
        patches.append(
            {
                'path': str(p.resolve()),
                'exists': True,
                'bytes': p.stat().st_size,
                'sha256': sha256_file(p),
            }
        )

    if patches:
        manifest['patches'] = patches
        diff_sha = manifest.get('repo', {}).get('git', {}).get('diff_patch_sha256')
        if diff_sha:
            manifest['patches'][0]['matches_git_diff_patch_sha256'] = patches[0].get('sha256') == diff_sha

    if args.dataset_audit_json:
        audit_p = Path(args.dataset_audit_json)
        manifest['dataset_audit'] = {
            'file': file_record(str(audit_p)),
            'content': maybe_json(audit_p) if audit_p.exists() else None,
        }

    if args.metrics_file:
        mp = Path(args.metrics_file)
        if mp.exists():
            manifest['metrics'] = maybe_json(mp)

    with out.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write('\n')

    print(f'[run_manifest] wrote: {out}')


if __name__ == '__main__':
    main()
