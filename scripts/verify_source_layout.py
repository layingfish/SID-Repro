#!/usr/bin/env python3
"""Validate the source-only RecSys26 workspace without requiring datasets/models."""

from __future__ import annotations

import argparse
import ast
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = "/data" + "/xqp_data/RecSys26"

REQUIRED_DIRS = (
    "baselines",
    "baselines/decoder",
    "baselines/ref01",
    "baselines/ref02",
    "baselines/ref03",
    "baselines/ref04",
    "baselines/ref05",
    "baselines/ref06",
    "baselines/ref07",
    "baselines/ref08",
    "baselines/reference",
    "configs",
    "experiments",
    "pipeline",
    "requirements",
    "scripts",
    "legacy/scripts",
    "third_party_clean",
    "patches",
    "reports",
)

REQUIRED_FILES = (
    "README.md",
    "RUNBOOK.md",
    "repos_manifest.tsv",
    "requirements/environment.yml",
    "requirements/requirements.txt",
    "scripts/run_data.sh",
    "scripts/run_rq1.sh",
    "scripts/run_rq2.sh",
    "scripts/run_rq3.sh",
    "scripts/run_rq4.sh",
    "pipeline/evaluation/unified_eval.py",
    "pipeline/decoder/train_with_manifest.py",
    "pipeline/export/export_with_manifest.py",
    "pipeline/data/export_sasrec_item_embed.py",
    "pipeline/tokenizer/build_letter_indices_setrec.sh",
    "scripts/rq1_native/repro_dispatch_paper.sh",
    "scripts/rq1_native/repro_setrec_t5.sh",
    "scripts/rq1_native/repro_llm_id.sh",
    "scripts/rq1_native/repro_seater.sh",
    "scripts/rq1_native/repro_eager.sh",
    "scripts/rq1_native/repro_diffgrm_paperalign.sh",
    "scripts/rq1_native/repro_letter_lc_rec.sh",
    "scripts/rq1_native/repro_sasrec.sh",
    "scripts/rq1_native/repro_onerec.sh",
    "legacy/scripts/scaling/run_experiment.py",
)

OPTIONAL_RESOURCES = {
    "datasets": ("DATA_ROOT", "raw/processed datasets and adapters"),
    "models": ("MODEL_ROOT", "pretrained model caches"),
    "envs": ("ENV_ROOT", "Conda environments"),
    "logs": ("RUN_ROOT", "training runs and checkpoints"),
}

CANONICAL_SOURCE_DIRS = (
    "baselines",
    "configs",
    "experiments",
    "pipeline",
    "requirements",
)


def _iter_text_files(paths: list[Path]):
    suffixes = {".py", ".sh", ".yaml", ".yml", ".json", ".gin", ".md", ".txt"}
    for path in paths:
        if path.is_file():
            yield path
            continue
        for candidate in path.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in suffixes:
                yield candidate


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--syntax", action="store_true", help="Parse canonical Python and Bash sources")
    parser.add_argument(
        "--strict-resources",
        action="store_true",
        help="Treat missing datasets/models/envs/logs as failures",
    )
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED_DIRS:
        if not (ROOT / rel).is_dir():
            failures.append(f"missing source directory: {rel}")
    for rel in REQUIRED_FILES:
        if not (ROOT / rel).is_file():
            failures.append(f"missing source file: {rel}")

    third_party = ROOT / "third_party"
    if not third_party.is_symlink() or third_party.resolve() != (ROOT / "third_party_clean").resolve():
        failures.append("third_party must be a symlink to third_party_clean")

    manifest_path = ROOT / "repos_manifest.tsv"
    if manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if len(rows) != 11:
            failures.append(f"repos_manifest.tsv must list 11 repositories, found {len(rows)}")
        for row in rows:
            name = row["name"]
            repo = ROOT / "third_party_clean" / name
            if not (repo / ".git").exists():
                failures.append(f"missing Git worktree: third_party_clean/{name}")
                continue
            head = _run(["git", "-C", str(repo), "rev-parse", "HEAD"])
            if head.returncode != 0:
                failures.append(f"cannot read Git HEAD for {name}: {head.stderr.strip()}")
            elif head.stdout.strip() != row["commit"]:
                failures.append(
                    f"commit mismatch for {name}: expected {row['commit']}, got {head.stdout.strip()}"
                )
            dirty = _run(["git", "-C", str(repo), "status", "--porcelain"])
            dirty_count = len([line for line in dirty.stdout.splitlines() if line.strip()])
            if dirty_count:
                warnings.append(f"{name}: {dirty_count} documented workspace changes")

    canonical_paths = [ROOT / rel for rel in CANONICAL_SOURCE_DIRS]
    canonical_paths.append(ROOT / "scripts")
    canonical_paths.extend((ROOT / "README.md", ROOT / "RUNBOOK.md"))
    hardcoded = []
    for path in _iter_text_files(canonical_paths):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            failures.append(f"cannot read {path.relative_to(ROOT)}: {exc}")
            continue
        if LEGACY_ROOT in text:
            hardcoded.append(str(path.relative_to(ROOT)))
    if hardcoded:
        failures.append("legacy root remains in canonical source: " + ", ".join(sorted(set(hardcoded))))

    if args.syntax:
        for path in _iter_text_files(canonical_paths):
            if path.suffix == ".py":
                try:
                    ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
                except SyntaxError as exc:
                    failures.append(f"Python syntax error in {path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
            elif path.suffix == ".sh":
                result = _run(["bash", "-n", str(path)])
                if result.returncode != 0:
                    failures.append(f"Bash syntax error in {path.relative_to(ROOT)}: {result.stderr.strip()}")

    for rel, (env_name, description) in OPTIONAL_RESOURCES.items():
        path = Path(os.environ.get(env_name, ROOT / rel))
        if path.exists():
            print(f"[RESOURCE] {rel}: present at {path}")
        else:
            message = f"{rel}: missing ({description}); expected at {path}"
            if args.strict_resources:
                failures.append(message)
            else:
                print(f"[OPTIONAL] {message}")

    for message in warnings:
        print(f"[WARN] {message}")
    for message in failures:
        print(f"[FAIL] {message}")

    if failures:
        print(f"[SUMMARY] source layout FAILED with {len(failures)} issue(s)")
        return 1

    print(f"[SUMMARY] source layout OK ({len(warnings)} workspace warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
