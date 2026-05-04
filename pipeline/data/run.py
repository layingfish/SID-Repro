from __future__ import annotations

import argparse
import importlib
import sys


COMMANDS = {
    "download": "pipeline.data.download_data",
    "split": "pipeline.data.make_setrec_splits",
    "build-text": "pipeline.data.build_setrec_text",
    "build-embeddings": "pipeline.data.build_setrec_t5_embeddings",
    "build-seater-tsv": "pipeline.data.build_seater_tsv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Data preparation launcher.")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()

    module = importlib.import_module(COMMANDS[parsed.command])
    sys.argv = [f"run_data:{parsed.command}", *parsed.args]
    module.main()


if __name__ == "__main__":
    main()
