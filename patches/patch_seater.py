#!/usr/bin/env python3
"""Patch SEATER to export pred_topk.jsonl during evaluation.
Run this script ON the server: python3 patch_seater.py
"""
import os
import shutil

SEATER_DIR = '/data/xqp_data/RecSys26/third_party/SEATER'

# === Patch 1: trainer.py — add user ID tracking + pred export in _run_eval ===
trainer_path = os.path.join(SEATER_DIR, 'trainer.py')
with open(trainer_path) as f:
    code = f.read()

# Backup
shutil.copy2(trainer_path, trainer_path + '.bak_export')

# 1a. Add 'import json' if not present
if 'import json' not in code:
    code = code.replace('import torch\n', 'import torch\nimport json\n', 1)

# 1b. Add group_user_ids tracking in _run_eval
old_run_eval_init = """        group_pred_logits, group_next_items, group_pred_items = [], [], []

        user_item_record = dataloader.dataset.record"""
new_run_eval_init = """        group_pred_logits, group_next_items, group_pred_items = [], [], []
        group_user_ids = []  # for pred_topk.jsonl export

        user_item_record = dataloader.dataset.record"""
assert old_run_eval_init in code, "Cannot find _run_eval init block"
code = code.replace(old_run_eval_init, new_run_eval_init)

# 1c. Add user ID collection in the loop
old_extend = """            group_pred_items.extend(step_pred_item.cpu().numpy())
            group_next_items.extend(step_next_items)"""
new_extend = """            group_pred_items.extend(step_pred_item.cpu().numpy())
            group_next_items.extend(step_next_items)
            group_user_ids.extend(step_uid.numpy().tolist())"""
assert old_extend in code, "Cannot find group_pred_items.extend block"
code = code.replace(old_extend, new_extend)

# 1d. Add prediction export before return in _run_eval
old_return = """        return group_pred_items, group_next_items"""
new_return = """        # Export pred_topk.jsonl
        pred_output_path = os.path.join(self.cm.workspace, 'pred_topk.jsonl')
        with open(pred_output_path, 'w') as f:
            for uid, preds in zip(group_user_ids, group_pred_items):
                # SEATER item IDs = SETRec 0-based + 1; convert back
                items_raw = preds[:20].tolist() if hasattr(preds, 'tolist') else list(preds)[:20]
                seen = set()
                unique = []
                for x in items_raw:
                    item_0based = int(x) - 1
                    if item_0based >= 0 and item_0based not in seen:
                        seen.add(item_0based)
                        unique.append(item_0based)
                f.write(json.dumps({"user_id": int(uid), "predicted_items": unique}) + '\\n')
        print(f'[EXPORT] Saved {len(group_user_ids)} users pred_topk.jsonl -> {pred_output_path}')

        return group_pred_items, group_next_items"""
# Only replace the FIRST occurrence (in _run_eval, not in any other method)
idx = code.find(old_return)
assert idx != -1, "Cannot find 'return group_pred_items, group_next_items'"
code = code[:idx] + new_return + code[idx + len(old_return):]

with open(trainer_path, 'w') as f:
    f.write(code)
print(f'[OK] Patched {trainer_path}')


# === Patch 2: main.py — add --test_only and --ckpt_path flags ===
main_path = os.path.join(SEATER_DIR, 'main.py')
with open(main_path) as f:
    main_code = f.read()

shutil.copy2(main_path, main_path + '.bak_export')

# Add argparse flags
if '--test_only' not in main_code:
    old_random_seed = "parser.add_argument('--random_seed', type=int, help='random seed', default=2023)"
    new_random_seed = """parser.add_argument('--random_seed', type=int, help='random seed', default=2023)
parser.add_argument('--test_only', action='store_true', help='Skip training, run test only')
parser.add_argument('--ckpt_path', type=str, default=None, help='Checkpoint path for test-only mode')"""
    assert old_random_seed in main_code, "Cannot find random_seed arg"
    main_code = main_code.replace(old_random_seed, new_random_seed)

    # Replace train+test with conditional
    old_end = """trainer.train()
trainer.test()"""
    new_end = """if args.test_only:
    trainer.test(assigned_model_path=args.ckpt_path)
else:
    trainer.train()
    trainer.test()"""
    assert old_end in main_code, "Cannot find trainer.train()/test() block"
    main_code = main_code.replace(old_end, new_end)

    with open(main_path, 'w') as f:
        f.write(main_code)
    print(f'[OK] Patched {main_path}')
else:
    print(f'[SKIP] {main_path} already patched')

print('\n=== SEATER patch complete ===')
print('Run test-only: python main.py --dataset_name SETRec_beauty --model SEATER --vocab 8 --test_only --ckpt_path <path>')
