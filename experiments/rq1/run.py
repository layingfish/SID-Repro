from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
NATIVE_DIR = ROOT_DIR / "scripts" / "rq1_native"


def normalize_method(method: str) -> str:
    key = method.lower().replace("_", "-")
    aliases = {
        "howtoindex": "llm_id",
        "t5semid": "llm_id",
        "t5-semid": "llm_id",
        "llm-id": "llm_id",
        "letter-tiger": "letter",
        "letter-lc-rec": "letter-lc-rec",
        "rqkmeans": "onerec",
        "rq-kmeans": "onerec",
        "one-rec": "onerec",
    }
    return aliases.get(key, key)


def native_command(method: str, dataset: str, variant: str) -> list[str]:
    method = normalize_method(method)
    if method == "setrec":
        return ["bash", str(NATIVE_DIR / "repro_setrec_t5.sh"), dataset, "paper"]
    if method == "llm_id":
        return ["bash", str(NATIVE_DIR / "repro_llm_id.sh"), dataset, "paper", variant or "semid"]
    if method == "seater":
        return ["bash", str(NATIVE_DIR / "repro_seater.sh"), dataset, "paper"]
    if method == "eager":
        return ["bash", str(NATIVE_DIR / "repro_eager.sh"), dataset, "paper"]
    if method == "diffgrm":
        return ["bash", str(NATIVE_DIR / "repro_diffgrm_paperalign.sh"), dataset]
    if method == "onerec":
        return ["bash", str(NATIVE_DIR / "repro_onerec.sh"), dataset]
    if method == "sasrec":
        return ["bash", str(NATIVE_DIR / "repro_sasrec.sh"), dataset]
    if method == "letter-lc-rec":
        return ["bash", str(NATIVE_DIR / "repro_letter_lc_rec.sh"), dataset]
    if method in {"tiger", "rpg", "letter", "etegrec"}:
        dispatch_method = {
            "letter-lc-rec": "letter_lc_rec",
            "onerec": "onerec",
        }.get(method, method)
        return ["bash", str(NATIVE_DIR / "repro_dispatch_paper.sh"), dispatch_method, dataset, variant or ""]
    raise SystemExit(f"Unknown RQ1 method: {method}")


def run_native(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Run the paper RQ1 method-native pipeline.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", default="semid")
    parser.add_argument("--run_dir", default="", help="Optional override consumed by rq1_native scripts when supported.")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--print_only", action="store_true")
    args = parser.parse_args(argv)

    env = os.environ.copy()
    env.setdefault("RECSYS26_ROOT", str(ROOT_DIR))
    if args.run_dir:
        env["RECSYS26_RUN_DIR"] = str(Path(args.run_dir).resolve())
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus

    cmd = native_command(args.method, args.dataset, args.variant)
    if args.print_only:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True, env=env)


def run_module(command: str, argv: list[str]) -> None:
    commands = {
        "evaluate": "pipeline.evaluation.unified_eval",
    }
    module = importlib.import_module(commands[command])
    sys.argv = [f"run_rq1:{command}", *argv]
    module.main()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ1 launcher. Native pipelines are copied from the experiment repository under scripts/rq1_native."
    )
    parser.add_argument("command", choices=["native", "evaluate"])
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()

    if parsed.command == "native":
        run_native(parsed.args)
    else:
        run_module(parsed.command, parsed.args)


if __name__ == "__main__":
    main()
