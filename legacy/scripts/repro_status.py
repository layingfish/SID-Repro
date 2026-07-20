#!/usr/bin/env python3
"""RecSys26 reproduction run status.

Reads `/data/xqp_data/RecSys26/logs/repro/**/<domain>/<run_ts>/` and prints a compact table.
Designed to work with system python (stdlib only).
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

RUN_TS_RE = re.compile(r"^\d{8}_\d{6}$")


def parse_mode(run_log: Path) -> str:
    if not run_log.exists():
        return "?"
    try:
        with run_log.open("r", errors="ignore") as f:
            for _ in range(80):
                line = f.readline()
                if not line:
                    break
                if line.startswith("[run] ") and " mode=" in line and "CUDA_VISIBLE_DEVICES" in line:
                    for tok in line.strip().split():
                        if tok.startswith("mode="):
                            return tok.split("=", 1)[1]
                    return "?"
    except Exception:
        return "?"
    return "?"


def read_exit_code(run_dir: Path) -> Optional[int]:
    p = run_dir / "exit_code.txt"
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except Exception:
        return -999


def run_state(run_dir: Path) -> str:
    code = read_exit_code(run_dir)
    if code is not None:
        return "DONE"
    run_log = run_dir / "run.log"
    if run_log.exists() and run_log.stat().st_size > 0:
        return "RUN"
    return "INIT"


def iter_runs(domain_dir: Path) -> Iterable[Path]:
    if not domain_dir.exists():
        return []
    runs = []
    for p in domain_dir.iterdir():
        if not p.is_dir():
            continue
        if RUN_TS_RE.match(p.name):
            runs.append(p)
    runs.sort(key=lambda p: p.name)
    return runs


def pick_run(
    run_root: Path, domain: str, prefer_mode: str
) -> Optional[Path]:
    domain_dir = run_root / domain
    runs = list(iter_runs(domain_dir))
    if not runs:
        return None

    def is_mode(run_dir: Path, mode: str) -> bool:
        return parse_mode(run_dir / "run.log") == mode

    # Prefer latest successful full run.
    if prefer_mode == "full":
        for run_dir in reversed(runs):
            if is_mode(run_dir, "full") and read_exit_code(run_dir) == 0:
                return run_dir
        for run_dir in reversed(runs):
            if is_mode(run_dir, "full"):
                return run_dir
        return runs[-1]

    # Prefer latest successful quick run.
    if prefer_mode == "quick":
        for run_dir in reversed(runs):
            if is_mode(run_dir, "quick") and read_exit_code(run_dir) == 0:
                return run_dir
        for run_dir in reversed(runs):
            if is_mode(run_dir, "quick"):
                return run_dir
        return runs[-1]

    # Any mode: prefer latest RUN, else latest successful, else latest.
    for run_dir in reversed(runs):
        if read_exit_code(run_dir) is None and run_state(run_dir) == "RUN":
            return run_dir
    for run_dir in reversed(runs):
        if read_exit_code(run_dir) == 0:
            return run_dir
    return runs[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/xqp_data/RecSys26/logs/repro")
    ap.add_argument(
        "--prefer",
        choices=["full", "quick", "any"],
        default="full",
        help="Which mode to prefer when selecting a run per method/domain.",
    )
    ap.add_argument(
        "--llm-id-variant",
        default="semid",
        choices=["all", "iid", "sid", "semid", "cid", "hid"],
    )
    ap.add_argument(
        "--domain",
        action="append",
        default=[],
        choices=["beauty", "toys", "sports", "steam"],
        help="Repeatable; defaults to all.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    domains = args.domain or ["beauty", "toys", "sports", "steam"]

    methods: list[Tuple[str, Path]] = [
        ("setrec", root / "setrec"),
    ]

    llm_variants = [args.llm_id_variant]
    if args.llm_id_variant == "all":
        llm_variants = ["iid", "sid", "semid", "cid", "hid"]
    for v in llm_variants:
        methods.append((f"llm_id/{v}", root / "llm_id" / v))

    methods += [
        ("eager", root / "eager"),
        ("letter", root / "letter"),
        ("etegrec", root / "etegrec"),
        ("seater", root / "seater"),
        ("tiger", root / "tiger"),
        ("rpg", root / "rpg"),
        ("diffgrm", root / "diffgrm"),
    ]

    rows = []
    for mlabel, mroot in methods:
        for d in domains:
            run_dir = pick_run(mroot, d, args.prefer)
            if run_dir is None:
                rows.append((mlabel, d, "-", "-", "-", "-"))
                continue
            mode = parse_mode(run_dir / "run.log")
            code = read_exit_code(run_dir)
            code_s = "?" if code is None else str(code)
            state = run_state(run_dir)
            rows.append((mlabel, d, run_dir.name, mode, code_s, state))

    header = ("method", "domain", "run", "mode", "exit", "state")
    colw = [max(len(r[i]) for r in rows + [header]) for i in range(len(header))]

    def fmt(row: Tuple[str, ...]) -> str:
        return "  ".join(row[i].ljust(colw[i]) for i in range(len(header)))

    print(fmt(header))
    for r in rows:
        print(fmt(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
