from __future__ import annotations

import argparse
import importlib
import sys


COMMANDS = {
    "train-decoder": "pipeline.decoder.train_with_manifest",
    "export-decoder": "pipeline.export.export_with_manifest",
    "evaluate": "pipeline.evaluation.unified_eval",
    "length-diagnostics": "experiments.rq3.length_inference_diagnostics",
    "capacity-diagnostics": "experiments.rq3.length_capacity_teacher_diagnostics",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RQ3 scaling launcher for SID length and backbone-size experiments. "
            "Use one subcommand followed by the target module arguments."
        ),
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()

    module_name = COMMANDS[parsed.command]
    module = importlib.import_module(module_name)
    sys.argv = [f"run_rq3:{parsed.command}", *parsed.args]
    module.main()


if __name__ == "__main__":
    main()
