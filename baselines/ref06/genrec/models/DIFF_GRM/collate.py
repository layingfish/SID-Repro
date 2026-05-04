

import torch
from typing import Dict, List, Any


def stack_to_tensor(seq, dtype=None):

    if torch.is_tensor(seq[0]):
        out = torch.stack(seq, dim=0)
        return out.to(dtype) if dtype is not None else out
    return torch.tensor(seq, dtype=(dtype if dtype is not None else torch.long))


def collate_fn_train(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:


    return {
        'history_sid': stack_to_tensor([b['history_sid'] for b in batch]),
        'history_mask': stack_to_tensor([b['history_mask'] for b in batch], dtype=torch.bool),
        'decoder_input_ids': stack_to_tensor([b['decoder_input_ids'] for b in batch]),
        'decoder_labels': stack_to_tensor([b['decoder_labels'] for b in batch]),
    }


def collate_fn_val(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:


    return {
        'history_sid': stack_to_tensor([b['history_sid'] for b in batch]),
        'history_mask': stack_to_tensor([b['history_mask'] for b in batch], dtype=torch.bool),
        'labels': stack_to_tensor([b['labels'] for b in batch]),
    }


def collate_fn_test(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:


    return collate_fn_val(batch)
