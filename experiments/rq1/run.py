from __future__ import annotations

import argparse
import importlib
import sys


COMMANDS = {
    "generate-manifest": "pipeline.tokenizer.generate_manifest",
    "manifest-to-cached-ids": "pipeline.tokenizer.manifest_to_cached_ids",
    "export-rqvae-codes": "pipeline.tokenizer.export_rqvae_codes",
    "export-diffgrm-tokens": "pipeline.tokenizer.export_diffgrm_item_tokens",
    "residual-kmeans": "pipeline.tokenizer.residual_kmeans",
    "train-decoder": "pipeline.decoder.train_with_manifest",
    "export-decoder": "pipeline.export.export_with_manifest",
    "evaluate": "pipeline.evaluation.unified_eval",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RQ1 overall benchmark launcher. Use one subcommand followed by "
            "the target module arguments."
        ),
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()

    module_name = COMMANDS[parsed.command]
    module = importlib.import_module(module_name)
    sys.argv = [f"run_rq1:{parsed.command}", *parsed.args]
    module.main()


if __name__ == "__main__":
    main()
