#!/usr/bin/env python3
"""Patch RPG to export pred_topk.jsonl during evaluation.
Also works for DiffGRM (same framework).
Usage: python3 patch_rpg_diffgrm.py [rpg|diffgrm]
"""
import os
import sys
import shutil

method = sys.argv[1] if len(sys.argv) > 1 else 'rpg'
if method == 'rpg':
    BASE_DIR = '/data/xqp_data/RecSys26/third_party/RPG_KDD2025'
elif method == 'diffgrm':
    BASE_DIR = '/data/xqp_data/RecSys26/third_party/DiffGRM'
else:
    raise ValueError(f'Unknown method: {method}')

# === Patch trainer.py — add pred export in evaluate() ===
trainer_path = os.path.join(BASE_DIR, 'genrec', 'trainer.py')
with open(trainer_path) as f:
    code = f.read()

shutil.copy2(trainer_path, trainer_path + '.bak_export')

if 'pred_topk.jsonl' in code:
    print(f'[SKIP] {trainer_path} already patched')
    sys.exit(0)

# Add import json at top
if 'import json' not in code:
    code = code.replace('import os\n', 'import os\nimport json\n', 1)

# The evaluate() method collects batch results but only stores metric values.
# We need to also collect decoded item predictions per user.
# Strategy: Add a new method `evaluate_and_export` that wraps evaluate.
# Or: modify evaluate to optionally save predictions.

# Simpler: Add a separate export method that runs after evaluate.
# The model.generate() returns token-level predictions (batch, topk, seq_len).
# We need to decode these back to item IDs using tokenizer.item2tokens reverse map.

# Add export_predictions method to Trainer class
export_method = '''
    def export_predictions(self, dataloader, output_path, split='test'):
        """Export per-user top-K item predictions to pred_topk.jsonl."""
        self.model.eval()

        # Build reverse token-to-item mapping
        tokens2item_id = {}
        for item, tokens in self.tokenizer.item2tokens.items():
            item_id = self.tokenizer.item2id[item]
            tokens2item_id[tokens] = item_id

        all_user_preds = []
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Export preds - {split}",
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                if self.config['use_ddp']:
                    preds = self.model.module.generate(batch, n_return_sequences=20)
                else:
                    preds = self.model.generate(batch, n_return_sequences=20)

                if isinstance(preds, tuple):
                    preds = preds[0]

                preds_cpu = preds.detach().cpu()
                labels_cpu = batch['labels'].detach().cpu()

                for i in range(preds_cpu.shape[0]):
                    # Decode each predicted token sequence to item ID
                    decoded_items = []
                    seen = set()
                    for j in range(preds_cpu.shape[1]):
                        pred_tokens = preds_cpu[i, j].tolist()
                        # Remove eos token
                        eos = self.tokenizer.eos_token
                        if eos in pred_tokens:
                            pred_tokens = pred_tokens[:pred_tokens.index(eos)]
                        pred_key = tuple(pred_tokens)
                        item_id = tokens2item_id.get(pred_key, None)
                        if item_id is not None:
                            # Convert from 1-based to SETRec 0-based
                            item_0based = item_id - 1
                            if item_0based >= 0 and item_0based not in seen:
                                seen.add(item_0based)
                                decoded_items.append(item_0based)

                    # Get user's label token sequence to identify user
                    label_tokens = labels_cpu[i].tolist()
                    if eos in label_tokens:
                        label_tokens = label_tokens[:label_tokens.index(eos)]
                    gt_item_id = tokens2item_id.get(tuple(label_tokens), None)

                    all_user_preds.append({
                        'gt_item_id_1based': gt_item_id,
                        'predicted_items': decoded_items[:20],
                    })

        # Save
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            for entry in all_user_preds:
                f.write(json.dumps(entry) + '\\n')
        self.log(f'[EXPORT] Saved {len(all_user_preds)} predictions to {output_path}')
        return all_user_preds
'''

# Insert before the 'def end(self):' method
old_end = '    def end(self):'
assert old_end in code, "Cannot find 'def end(self):' in trainer.py"
code = code.replace(old_end, export_method + '\n' + old_end)

with open(trainer_path, 'w') as f:
    f.write(code)
print(f'[OK] Patched {trainer_path}')


# === Patch pipeline.py — call export_predictions after test ===
pipeline_path = os.path.join(BASE_DIR, 'genrec', 'pipeline.py')
with open(pipeline_path) as f:
    pcode = f.read()

shutil.copy2(pipeline_path, pipeline_path + '.bak_export')

if 'export_predictions' not in pcode:
    # Add import
    if 'import json' not in pcode:
        pcode = pcode.replace('import os\n', 'import os\nimport json\n', 1)

    old_test_results = "        test_results = self.trainer.evaluate(test_dataloader)"
    new_test_results = """        test_results = self.trainer.evaluate(test_dataloader)

        # Export per-user predictions
        if self.accelerator.is_main_process:
            export_path = os.path.join(self.project_dir, 'pred_topk.jsonl')
            self.trainer.export_predictions(test_dataloader, export_path)"""

    if old_test_results in pcode:
        pcode = pcode.replace(old_test_results, new_test_results)
        with open(pipeline_path, 'w') as f:
            f.write(pcode)
        print(f'[OK] Patched {pipeline_path}')
    else:
        print(f'[WARN] Could not find test_results line in pipeline.py')
else:
    print(f'[SKIP] {pipeline_path} already patched')

print(f'\n=== {method.upper()} patch complete ===')
