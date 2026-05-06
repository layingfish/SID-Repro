from __future__ import annotations

import argparse
import importlib
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def normalize_method(method: str) -> str:
    key = method.lower().replace("_", "-")
    aliases = {
        "howtoindex": "t5-semid",
        "t5semid": "t5-semid",
        "t5-semid": "t5-semid",
        "llm-id": "t5-semid",
        "letter": "letter-tiger",
        "letter-tiger": "letter-tiger",
        "letter-lc-rec": "letter-lc-rec",
        "rqkmeans": "onerec",
        "rq-kmeans": "onerec",
        "one-rec": "onerec",
    }
    return aliases.get(key, key)


def build_native_script(args: argparse.Namespace) -> str:
    method = normalize_method(args.method)
    dataset = args.dataset
    run_dir = Path(args.run_dir or f"runs/rq1/{method}/{dataset}")
    variant = args.variant or "semid"
    gpus = args.gpus

    header = f"""
    set -euo pipefail
    export DATA_ROOT="${{DATA_ROOT:-data}}"
    export MODEL_ROOT="${{MODEL_ROOT:-models}}"
    export RUN_ROOT="${{RUN_ROOT:-runs}}"
    mkdir -p {q(run_dir)}
    """
    if gpus:
        header += f'\nexport CUDA_VISIBLE_DEVICES={q(gpus)}\n'

    if method == "setrec":
        script = f"""
        cd {q(ROOT_DIR / "baselines/reference/code")}
        torchrun --nproc_per_node="${{NPROC_PER_NODE:-4}}" finetune_t5.py \
          --base_model t5-small \
          --data_path ../data/{q(dataset)}/ \
          --output_dir {q(run_dir / "out")} \
          --cache_dir "${{MODEL_ROOT}}/hf/transformers" \
          --sem_encoder t5 \
          --n_query 5 --n_sem 4 --n_cf 1 --alpha 0.7 \
          --layers 512 256 128 \
          --batch_size 512 --micro_batch_size 128 \
          --num_epochs 30 \
          --learning_rate 0.001 \
          --cutoff_len 512 \
          --val_set_size 2000 \
          --warmup_steps 100 \
          --lr_scheduler cosine \
          --disable_early_stopping \
          --seed 42 \
          --wandb_project ""
        """
    elif method == "t5-semid":
        item_representation = "content_based"
        data_order = "random"
        extra = ""
        if variant == "iid":
            item_representation = "no_tokenization"
        elif variant == "sid":
            item_representation = "None"
            data_order = "remapped_sequential"
            extra = "--remapped_data_order original"
        elif variant == "cid":
            item_representation = "CF"
            data_order = "remapped_sequential"
            extra = "--remapped_data_order original --cluster_size 500 --cluster_number 20"
        elif variant == "hid":
            item_representation = "CF"
            data_order = "remapped_sequential"
            extra = "--remapped_data_order original --cluster_size 500 --cluster_number 20 --last_token_no_repetition"
        script = f"""
        cd {q(ROOT_DIR / "baselines/ref02")}
        python main.py \
          --task setrec_{q(dataset)} \
          --data_dir data/ \
          --seed 42 \
          --warmup_prop 0.05 \
          --lr 0.001 \
          --clip 1.0 \
          --model_type t5-small \
          --epochs 20 \
          --logging_step 1000 \
          --logging_dir {q(run_dir / "train.log")} \
          --model_dir {q(run_dir / "model.pt")} \
          --train_sequential_item_batch 64 \
          --train_sequential_yesno_batch 16 \
          --train_direct_yesno_batch 16 \
          --train_direct_candidate_batch 8 \
          --train_direct_straightforward_batch 16 \
          --item_representation {q(item_representation)} \
          --data_order {q(data_order)} \
          {extra}
        python test_only_llm_id.py \
          --ckpt_path {q(run_dir / "model.pt")} \
          --task setrec_{q(dataset)} \
          --data_dir data/ \
          --export_path {q(run_dir / "pred_topk.jsonl")} \
          --item_representation {q(item_representation)} \
          --data_order {q(data_order)} \
          --num_beams 20 \
          --topk 20 \
          {extra}
        """
    elif method == "rpg":
        script = f"""
        cd {q(ROOT_DIR / "baselines/ref04")}
        python main.py \
          --dataset=SETRec \
          --category={q(dataset)} \
          --epochs=150 \
          --train_batch_size=256 \
          --eval_batch_size=32 \
          --lr=0.001 \
          --temperature=0.03 \
          --n_codebook=64 \
          --codebook_size=256 \
          --num_beams=20 \
          --n_edges=500 \
          --propagation_steps=5 \
          --cache_dir={q(run_dir / "cache")} \
          --log_dir={q(run_dir)} \
          --ckpt_dir={q(run_dir / "ckpt")} \
          --run_id=rpg_setrec_{q(dataset)} \
          --num_proc=1 \
          --sent_emb_model=setrec_t5_tdcb \
          --sent_emb_dim=768 \
          --sent_emb_pca=0 \
          --sent_emb_batch_size=512
        """
    elif method == "diffgrm":
        script = f"""
        cd {q(ROOT_DIR / "baselines/ref06")}
        python main.py \
          --model=DIFF_GRM \
          --dataset=SETRec \
          --category={q(dataset)} \
          --epochs=200 \
          --train_batch_size=1024 \
          --eval_batch_size=32 \
          --n_digit=4 \
          --masking_strategy=guided \
          --guided_refresh_each_step=false \
          --guided_select=least \
          --guided_conf_metric=msp \
          --train_sliding=true \
          --min_hist_len=2 \
          --lr=0.003 \
          --label_smoothing=0.1 \
          --sent_emb_model=setrec_t5_tdcb \
          --sent_emb_dim=768 \
          --sent_emb_pca=256 \
          --normalize_after_pca=true \
          --force_regenerate_opq=true \
          --share_decoder_output_embedding=true \
          --cache_dir={q(run_dir / "cache")} \
          --log_dir={q(run_dir)} \
          --ckpt_dir={q(run_dir / "ckpt")} \
          --run_id=diffgrm_setrec_{q(dataset)} \
          --num_proc=1
        """
    elif method == "seater":
        script = f"""
        cd {q(ROOT_DIR / "baselines/ref01")}
        python main.py \
          --name SEATER_SETRec_{q(dataset)} \
          --workspace {q(run_dir / "workspace")} \
          --dataset_name SETRec_{q(dataset)} \
          --model SEATER \
          --vocab 8 \
          --gpu_id 0 \
          --epochs 100 \
          --batch_size 256 \
          --test_batch_size 1024 \
          --num_workers 4 \
          --no-tb \
          --no-train_tb
        """
    elif method == "eager":
        script = f"""
        cd {q(ROOT_DIR / "baselines/ref08")}
        python train_rec_setrec.py \
          --domain {q(dataset)} \
          --data_root "${{DATA_ROOT}}/setrec_data" \
          --output_dir {q(run_dir)} \
          --seed 2024 \
          --seq_len 20 --min_seq_len 5 \
          --train_sample_seg_cnt 10 \
          --parall 50 \
          --tree_num 2 \
          --k 128 \
          --d_model 96 \
          --enc_num_layers 1 \
          --dec_num_layers 2,2 \
          --n_head 4 \
          --init_way embkm,embkm \
          --feature_ratio 1.0 \
          --max_iters 100 \
          --train_batch_size 256 \
          --total_batch_num 60000 \
          --eval_step 300 \
          --eval_topk 10 \
          --eval_rerank_topk 10 \
          --merge_method confidence
        """
    elif method == "etegrec":
        script = f"""
        cd {q(ROOT_DIR / "baselines/ref07")}
        python main.py \
          --config config/setrec_{q(dataset)}.yaml \
          --log_dir={q(run_dir)} \
          --epochs=999 \
          --early_stop=50 \
          --batch_size=256 \
          --eval_batch_size=256 \
          --eval_step=1
        """
    elif method == "letter-tiger":
        script = f"""
        cd {q(ROOT_DIR / "baselines/ref03/LETTER-TIGER")}
        torchrun --nproc_per_node="${{NPROC_PER_NODE:-2}}" finetune.py \
          --base_model t5-small \
          --output_dir {q(run_dir / "out")} \
          --dataset SETRec_{q(dataset)} \
          --index_file .index.json \
          --per_device_batch_size 256 \
          --learning_rate 5e-4 \
          --epochs 200 \
          --temperature 1.0
        python test.py \
          --gpu_id 0 \
          --ckpt_path {q(run_dir / "out")} \
          --dataset SETRec_{q(dataset)} \
          --data_path ../data \
          --results_file {q(run_dir / "results.json")} \
          --test_batch_size 32 \
          --num_beams 20 \
          --test_prompt_ids 0 \
          --index_file .index.json
        """
    elif method == "letter-lc-rec":
        script = f"""
        cd {q(ROOT_DIR / "baselines/ref03/LETTER-LC-Rec")}
        bash run_train.sh
        bash run_test_ddp.sh
        """
    elif method == "tiger":
        script = f"""
        cd {q(ROOT_DIR / "baselines/decoder")}
        python train_rqvae.py configs/rqvae_amazon.gin
        python train_decoder.py configs/decoder_amazon.gin
        """
    elif method == "onerec":
        script = f"""
        cd {q(ROOT_DIR / "baselines/ref05")}
        python -m src.train --config-name train
        python -m src.inference --config-name inference
        """
    elif method == "sasrec":
        script = f"""
        cd {q(ROOT_DIR / "baselines/ref01")}
        python main.py \
          --dataset_name SETRec_{q(dataset)} \
          --model SASREC \
          --workspace {q(run_dir / "workspace")} \
          --gpu_id 0
        """
    else:
        raise SystemExit(f"Unknown RQ1 method: {args.method}")

    return textwrap.dedent(header + script).strip() + "\n"


def run_native(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Run a method-native RQ1 pipeline.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", default="")
    parser.add_argument("--run_dir", default="")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--print_only", action="store_true")
    args = parser.parse_args(argv)

    script = build_native_script(args)
    if args.print_only:
        print(script, end="")
        return
    subprocess.run(["bash", "-lc", script], check=True)


def run_module(command: str, argv: list[str]) -> None:
    commands = {
        "evaluate": "pipeline.evaluation.unified_eval",
    }
    module = importlib.import_module(commands[command])
    sys.argv = [f"run_rq1:{command}", *argv]
    module.main()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RQ1 overall benchmark launcher. RQ1 uses method-native training "
            "and decoding; only the final evaluator is shared."
        ),
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
