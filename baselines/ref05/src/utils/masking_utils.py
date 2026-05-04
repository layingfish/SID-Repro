from typing import Optional

import torch

def create_last_k_mask(
    sequence_length: int, last_item_index: torch.Tensor, last_k: Optional[int] = None
) -> torch.tensor:


    if last_k is None:
        start_index = torch.zeros_like(last_item_index)
    else:
        if last_k < 1:
            raise ValueError("last_k must be None or greater than or equal to 1")
        start_index = torch.clamp(
            last_item_index - last_k + 1, min=0
        )

    indices = (
        torch.arange(sequence_length, device=last_item_index.device)
        .unsqueeze(0)
        .expand(last_item_index.size(0), -1)
    )

    mask = (indices >= start_index.unsqueeze(1)) & (
        indices <= last_item_index.unsqueeze(1)
    )
    return mask
