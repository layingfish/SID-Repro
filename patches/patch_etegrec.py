#!/usr/bin/env python3
"""Patch ETEGRec to export pred_topk.jsonl during test.
Run ON server: python3 patch_etegrec.py
"""
import os
import shutil

ETEGREC_DIR = '/data/xqp_data/RecSys26/third_party/ETEGRec'
trainer_path = os.path.join(ETEGREC_DIR, 'trainer.py')

with open(trainer_path) as f:
    code = f.read()

shutil.copy2(trainer_path, trainer_path + '.bak_export')

if 'pred_topk.jsonl' in code:
    print(f'[SKIP] {trainer_path} already patched')
    exit(0)

# The _test_epoch method generates predictions at the token level.
# It has code2item mapping: code2item[str(c)] -> list of item_ids (1-based).
# Model generates top-10 (n_return_sequences=10).
# We need to decode predicted token sequences to item IDs.
#
# Strategy: After the eval loop, decode all predictions and save.
# The challenge: predictions are token-level, computed batch by batch.
# We need to collect them and decode after the loop.

# Add prediction collection and export at the end of _test_epoch

old_return = """        for m in metrics:
            metrics[m] = round(metrics[m] / total, 6)

        return metrics"""

new_return = """        for m in metrics:
            metrics[m] = round(metrics[m] / total, 6)

        # Export pred_topk.jsonl
        self._export_pred_topk(code)

        return metrics"""

assert old_return in code, "Cannot find metrics return block in _test_epoch"
code = code.replace(old_return, new_return)

# Now add the _export_pred_topk method.
# It needs to re-run inference to collect per-user predictions.
# Actually, it's cleaner to collect predictions DURING the existing loop.
# Let me modify the approach: collect predictions in the loop instead.

# Revert the above change and do it differently:
# Collect predictions during the loop
code = code.replace(new_return, old_return)  # revert

# Instead, modify the inner loop to collect predictions
old_loop_metrics = """            for m in metrics.keys():
                metrics[m] += _metrics[m]"""

new_loop_metrics = """            for m in metrics.keys():
                metrics[m] += _metrics[m]

            # Collect predictions for export
            preds_cpu = preds.detach().cpu().tolist()
            for pred_seq_batch in preds_cpu:
                _all_pred_seqs.append(pred_seq_batch)"""

assert old_loop_metrics in code, "Cannot find inner metrics accumulation"
code = code.replace(old_loop_metrics, new_loop_metrics)

# Add _all_pred_seqs initialization before the loop
old_iter_data = """        iter_data = tqdm(
            test_data,
            total=len(test_data),
            ncols=100,
            desc=set_color(f"Evaluate   ", "pink"),
            disable=(not verbose) or (not self.accelerator.is_main_process),
        )"""

new_iter_data = """        _all_pred_seqs = []  # collect for pred_topk export

        iter_data = tqdm(
            test_data,
            total=len(test_data),
            ncols=100,
            desc=set_color(f"Evaluate   ", "pink"),
            disable=(not verbose) or (not self.accelerator.is_main_process),
        )"""

assert old_iter_data in code, "Cannot find iter_data tqdm block"
code = code.replace(old_iter_data, new_iter_data)

# Add export at the end of _test_epoch (before return metrics)
old_final_return = """        for m in metrics:
            metrics[m] = round(metrics[m] / total, 6)

        return metrics"""

new_final_return = """        for m in metrics:
            metrics[m] = round(metrics[m] / total, 6)

        # Export pred_topk.jsonl
        try:
            import json as _json
            # Build code2item: token_sequence_str -> list of 1-based item IDs
            _code2item = defaultdict(list)
            for i, c in enumerate(code[1:]):
                _code2item[str(c)].append(i + 1)

            # Get test user IDs from test_data
            _test_user_ids = []
            for batch_idx, data in enumerate(test_data):
                targets = data.get("targets", data.get("labels", None))
                if targets is not None:
                    _test_user_ids.extend(range(batch_idx * targets.shape[0],
                                                batch_idx * targets.shape[0] + targets.shape[0]))

            _output_path = os.path.join(self.save_path, 'pred_topk.jsonl')
            _n_exported = 0
            with open(_output_path, 'w') as _f:
                for _idx, _pred_seqs in enumerate(_all_pred_seqs):
                    # _pred_seqs is a list of token sequences (top-K predictions)
                    # Each sequence is a list of code_length tokens
                    # Decode to item IDs
                    _decoded = []
                    _seen = set()
                    if isinstance(_pred_seqs[0], list):
                        _candidates = _pred_seqs
                    else:
                        _candidates = [_pred_seqs]
                    for _seq in _candidates:
                        _key = str(_seq)
                        _items = _code2item.get(_key, [])
                        for _item_1based in _items:
                            _item_0based = _item_1based - 1
                            if _item_0based >= 0 and _item_0based not in _seen:
                                _seen.add(_item_0based)
                                _decoded.append(_item_0based)
                    # We don't have explicit user IDs; use sequential index
                    _f.write(_json.dumps({"user_idx": _idx, "predicted_items": _decoded[:20]}) + '\\n')
                    _n_exported += 1
            print(f'[EXPORT] Saved {_n_exported} users to {_output_path}')
        except Exception as e:
            print(f'[EXPORT] Failed: {e}')

        return metrics"""

assert old_final_return in code, "Cannot find final return in _test_epoch"
code = code.replace(old_final_return, new_final_return)

with open(trainer_path, 'w') as f:
    f.write(code)
print(f'[OK] Patched {trainer_path}')
print('\n=== ETEGRec patch complete ===')
