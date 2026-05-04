import torch
from typing import Tuple


@torch.no_grad()
def _topk_2d_sum(s: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


    B, X, Y = s.shape
    flat = s.reshape(B, -1)
    top_k = min(k, X * Y)
    vals, idx = torch.topk(flat, k=top_k, dim=-1)
    ix = idx // Y
    iy = idx % Y
    return vals, ix, iy


@torch.no_grad()
def combine_remaining_topk(
    per_digit_logp: torch.Tensor,
    topK_final: int,
    per_digit_topL: int,
) -> Tuple[torch.Tensor, torch.Tensor]:


    B, r, V = per_digit_logp.shape
    assert r >= 1
    L = min(per_digit_topL, V)


    vals = []
    ids = []
    for i in range(r):
        v, idx = torch.topk(per_digit_logp[:, i], k=L, dim=-1)
        vals.append(v)
        ids.append(idx)


    cur_vals = vals[0]
    cur_ids = ids[0].unsqueeze(-1)
    for i in range(1, r):
        s = cur_vals.unsqueeze(2) + vals[i].unsqueeze(1)
        v, ix, iy = _topk_2d_sum(s, k=min(topK_final, L * L))


        prev = torch.gather(cur_ids, dim=1, index=ix.unsqueeze(-1).expand(-1, -1, cur_ids.shape[-1]))
        tok_i = torch.gather(ids[i], dim=1, index=iy)
        cur_ids = torch.cat([prev, tok_i.unsqueeze(-1)], dim=-1)
        cur_vals = v

    return cur_vals, cur_ids
