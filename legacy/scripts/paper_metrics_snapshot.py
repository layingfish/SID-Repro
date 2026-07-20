#!/usr/bin/env python3
"""Generate a compact markdown snapshot of paper-profile runs.

Reads latest runs under `/data/xqp_data/RecSys26/logs/repro_paper/**` and writes a
report to `/data/xqp_data/RecSys26/reports/paper_profile_metrics_<ts>.md`.

Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path


RUN_TS_RE = re.compile(r"^\d{8}_\d{6}$")


def tail_lines(path: Path, n: int) -> list[str]:
    try:
        with path.open("r", errors="ignore") as f:
            lines = f.readlines()
        return lines[-n:]
    except FileNotFoundError:
        return []


def resolve_latest(dir_path: Path) -> Path | None:
    if not dir_path.exists():
        return None

    latest = dir_path / "latest"
    if latest.exists():
        try:
            return latest.resolve()
        except Exception:
            return latest

    runs = [p for p in dir_path.iterdir() if p.is_dir() and RUN_TS_RE.match(p.name)]
    return sorted(runs)[-1] if runs else None


def read_exit_code(run_dir: Path) -> str:
    p = run_dir / "exit_code.txt"
    if not p.exists():
        return "?"
    try:
        return p.read_text().strip()
    except Exception:
        return "?"


def parse_setrec_all(run_log: Path) -> str | None:
    lines = tail_lines(run_log, 20000)
    if not lines:
        return None

    best_beta = None
    for line in reversed(lines):
        m = re.search(r"Best beta is ([0-9.]+)", line)
        if m:
            best_beta = m.group(1)
            break

    header_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if "======= All performance" in lines[i]:
            header_idx = i
            break
    if header_idx is None:
        return None

    def parse_test(line: str) -> tuple[str, str, str, str] | None:
        m = re.search(
            r"Recall: ([0-9.]+)-([0-9.]+).*?NDCG: ([0-9.]+)-([0-9.]+)",
            line,
        )
        if not m:
            return None
        return m.group(1), m.group(2), m.group(3), m.group(4)

    for j in range(header_idx + 1, min(header_idx + 20, len(lines))):
        line = lines[j]
        if "[Test]:" in line and "Recall:" in line and "NDCG:" in line:
            parsed = parse_test(line)
            if not parsed:
                continue
            r5, r10, n5, n10 = parsed
            suffix = f" (best_beta={best_beta})" if best_beta is not None else ""
            return f"All: R@5={r5} R@10={r10} N@5={n5} N@10={n10}{suffix}"

    # fallback: any test line after header
    for j in range(header_idx + 1, len(lines)):
        line = lines[j]
        if "[Test]:" in line and "Recall:" in line and "NDCG:" in line:
            parsed = parse_test(line)
            if not parsed:
                continue
            r5, r10, n5, n10 = parsed
            suffix = f" (best_beta={best_beta})" if best_beta is not None else ""
            return f"All: R@5={r5} R@10={r10} N@5={n5} N@10={n10}{suffix}"

    return None


def parse_ordereddict(line: str) -> dict[str, float] | None:
    m = re.search(r"OrderedDict\((\[.*\])\)", line)
    if not m:
        return None
    try:
        pairs = ast.literal_eval(m.group(1))
        return {k: float(v) for k, v in pairs}
    except Exception:
        return None


def parse_test_ordereddict(run_log: Path) -> str | None:
    lines = tail_lines(run_log, 12000)
    for line in reversed(lines):
        if "Test Results" in line and "OrderedDict" in line:
            d = parse_ordereddict(line)
            if not d:
                continue
            r5 = d.get("recall@5")
            r10 = d.get("recall@10")
            n5 = d.get("ndcg@5")
            n10 = d.get("ndcg@10")
            if None in (r5, r10, n5, n10):
                keys = [k for k in ("recall@5", "recall@10", "ndcg@5", "ndcg@10") if k in d]
                return "Test: " + " ".join(f"{k}={d[k]:.6g}" for k in keys)
            return f"Test: recall@5={r5:.6g} recall@10={r10:.6g} ndcg@5={n5:.6g} ndcg@10={n10:.6g}"
    return None


def parse_seater_test(run_log: Path) -> str | None:
    lines = tail_lines(run_log, 40000)
    for line in reversed(lines):
        if line.strip().startswith("test:"):
            def get(name: str) -> str | None:
                m = re.search(
                    rf"'{re.escape(name)}':\s*(?:np\.float64\()?(?P<num>[-0-9.eE]+)(?:\))?",
                    line,
                )
                return m.group("num") if m else None

            r20, r50 = get("recall@20"), get("recall@50")
            n20, n50 = get("ndcg@20"), get("ndcg@50")
            if None in (r20, r50, n20, n50):
                return line.strip()
            return f"Test: recall@20={r20} recall@50={r50} ndcg@20={n20} ndcg@50={n50}"
    return None


def parse_eager_last(run_log: Path) -> str | None:
    lines = tail_lines(run_log, 40000)
    for line in reversed(lines):
        if "[EAGER][SETRec]" in line:
            return line.strip()
    return None


def parse_etegrec_last(run_log: Path) -> str | None:
    lines = tail_lines(run_log, 40000)
    for line in reversed(lines):
        if "Test Results:" in line:
            return "Test: " + line.split("Test Results:", 1)[1].strip()
    return None


def parse_llm_id_last(train_log: Path) -> str | None:
    lines = tail_lines(train_log, 60000)
    last: dict[str, str] = {}
    for line in lines:
        m = re.search(r"test (hit|ncdg) @ (\d+) is ([0-9.]+)", line)
        if m:
            last[f"{m.group(1)}@{m.group(2)}"] = m.group(3)
    if not last:
        return None
    keys = [k for k in ("hit@5", "hit@10", "ncdg@5", "ncdg@10") if k in last]
    if not keys:
        return None
    return "Test(last): " + " ".join(f"{k}={last[k]}" for k in keys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/data/xqp_data/RecSys26"))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <root>/reports",
    )
    args = ap.parse_args()

    root: Path = args.root
    logs = root / "logs" / "repro_paper"
    out_dir = args.out_dir or (root / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    domains = ["beauty", "toys", "sports", "steam"]

    out: list[str] = []
    out.append(f"# Paper-profile metrics snapshot ({ts} UTC)")
    out.append("")
    out.append("Dataset: SETRec splits from `third_party/SETRec/data/`.")
    out.append("Logs: `/data/xqp_data/RecSys26/logs/repro_paper/**`.")
    out.append("")
    out.append("## Latest per method/domain")
    out.append("")

    def add_section(name: str) -> None:
        out.append(f"### {name}")

    def add_method(method: str, parser, *, sub: str | None = None) -> None:
        add_section(method if sub is None else f"{method}/{sub}")
        for d in domains:
            base = logs / method
            if sub is not None:
                base = base / sub
            run_dir = resolve_latest(base / d)
            if run_dir is None:
                out.append(f"- {d}: -")
                continue
            out.append(f"- {d}: exit={read_exit_code(run_dir)} run=`{run_dir.relative_to(root)}/run.log`")
            metric = parser(run_dir)
            if metric:
                out.append(f"  - {metric}")
        out.append("")

    add_method("diffgrm", lambda r: parse_test_ordereddict(r / "run.log"))
    add_method("eager", lambda r: parse_eager_last(r / "run.log"))
    add_method("etegrec", lambda r: parse_etegrec_last(r / "run.log"))

    def llm_parser(run_dir: Path) -> str | None:
        train_log = run_dir / "train.log"
        if not train_log.exists():
            return None
        return parse_llm_id_last(train_log)

    add_method("llm_id", llm_parser, sub="sid")
    add_method("rpg", lambda r: parse_test_ordereddict(r / "run.log"))
    add_method("seater", lambda r: parse_seater_test(r / "run.log"))
    add_method("setrec", lambda r: parse_setrec_all(r / "run.log"))

    def tiger_parser(run_dir: Path) -> str | None:
        run_log = run_dir / "run.log"
        lines = tail_lines(run_log, 40000)
        # Find the last metrics dict.
        last = None
        for line in reversed(lines):
            s = line.strip()
            if s.startswith("{") and s.endswith("}"):
                last = s
                break
        if not last:
            return None
        try:
            d = ast.literal_eval(last)
        except Exception:
            return None

        # Full-sequence metrics are under h@k_slice_:D (D=max slice depth).
        Ds = []
        for k in d.keys():
            m = re.match(r"h@\d+_slice_:(\d+)", str(k))
            if m:
                Ds.append(int(m.group(1)))
        if not Ds:
            return None
        D = max(Ds)

        r5 = d.get(f"h@5_slice_:{D}")
        r10 = d.get(f"h@10_slice_:{D}")
        n5 = d.get(f"ndcg@5_slice_:{D}")
        n10 = d.get(f"ndcg@10_slice_:{D}")
        if None in (r5, r10, n5, n10):
            return None
        return f"Test: recall@5={float(r5):.6g} recall@10={float(r10):.6g} ndcg@5={float(n5):.6g} ndcg@10={float(n10):.6g} (D={D})"

    def letter_parser(run_dir: Path) -> str | None:
        p = run_dir / "results.json"
        if not p.exists():
            return None
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        mean = obj.get("mean_results") or {}
        # Map hit@k -> recall@k (single positive per user).
        r5 = mean.get("hit@5")
        r10 = mean.get("hit@10")
        n5 = mean.get("ndcg@5")
        n10 = mean.get("ndcg@10")
        if None in (r5, r10, n5, n10):
            return None
        return f"Test: recall@5={float(r5):.6g} recall@10={float(r10):.6g} ndcg@5={float(n5):.6g} ndcg@10={float(n10):.6g}"

    add_method("tiger", tiger_parser)
    add_method("letter", letter_parser)

    report_path = out_dir / f"paper_profile_metrics_{ts}.md"
    report_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
