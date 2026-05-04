

import torch
from typing import Dict, List, Any


def stack_to_tensor(seq):

    if torch.is_tensor(seq[0]):
        return torch.stack(seq, dim=0)
    return torch.tensor(seq, dtype=torch.long)


def collate_fn_train(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:


    return {
        'history_sid': stack_to_tensor([b['history_sid'] for b in batch]),
        'decoder_input_ids': stack_to_tensor([b['decoder_input_ids'] for b in batch]),
        'decoder_labels': stack_to_tensor([b['decoder_labels'] for b in batch]),
    }


def collate_fn_val(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:


    return {
        'history_sid': stack_to_tensor([b['history_sid'] for b in batch]),
        'labels': stack_to_tensor([b['labels'] for b in batch]),
    }


def collate_fn_test(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:


    return collate_fn_val(batch)
