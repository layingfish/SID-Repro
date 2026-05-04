import torch
import torch.nn.functional as F
from typing import Tuple

from .comb_topk import combine_remaining_topk


@torch.no_grad()
def decode_ablate_confidence(
    model,
    encoder_hidden: torch.Tensor,
    tokenizer,
    steps: int,
    n_return_sequences: int,
) -> torch.Tensor:


    device = encoder_hidden.device
    B = encoder_hidden.size(0)
    n_digit = model.n_digit
    VOC = model.codebook_size


    beam_cfg = model.config.get('vectorized_beam_search', {}) or {}
    split = model.config.get('current_split', 'val')

    def _as_int(d, k, default):
        try:
            return int(d.get(k, default))
        except Exception:
            return int(default)

    base = beam_cfg.get(split, beam_cfg)
    BEAM_ACT = _as_int(base, 'beam_act', 128)
    TOP_K_FINAL_CFG = _as_int(beam_cfg, 'top_k_final', n_return_sequences)
    TOP_K_FINAL = min(TOP_K_FINAL_CFG, n_return_sequences)

    neg_key = 'neg_inf_fp16' if encoder_hidden.dtype == torch.float16 else 'neg_inf_fp32'
    NEG_INF = float(beam_cfg.get(neg_key, -10000.0 if 'fp16' in neg_key else -1.0e9))


    ab_cfg = model.config.get('ablate_decode', {}) or {}
    if 'beam' in ab_cfg and isinstance(ab_cfg['beam'], dict) and split in ab_cfg['beam']:
        BEAM_ACT = int(ab_cfg['beam'][split].get('beam_act', BEAM_ACT))
    per_digit_topL = int(ab_cfg.get('per_digit_topk') or BEAM_ACT)


    MASK_ID = -1


    mask_positions = torch.ones(B, n_digit, device=device)
    batch0 = {
        'decoder_input_ids': torch.zeros(B, n_digit, device=device, dtype=torch.long),
        'encoder_hidden': encoder_hidden,
        'mask_positions': mask_positions
    }
    out0 = model.forward_decoder_only(batch0, digit=None, use_cache=False)
    logp0 = F.log_softmax(out0.logits, dim=-1)


    if steps <= 1:
        comb_vals, comb_tok = combine_remaining_topk(
            per_digit_logp=logp0,
            topK_final=BEAM_ACT,
            per_digit_topL=per_digit_topL,
        )
        seqs = _post_select(comb_tok, comb_vals, tokenizer, TOP_K_FINAL)
        return seqs


    beam_ids = torch.full((B, BEAM_ACT, n_digit), MASK_ID, dtype=torch.long, device=device)
    beam_lp = torch.full((B, BEAM_ACT), NEG_INF, dtype=logp0.dtype, device=device)


    flat0 = logp0.view(B, -1)
    best0, idx0 = torch.topk(flat0, k=BEAM_ACT, dim=-1)
    d0 = idx0 // VOC
    t0 = idx0 % VOC
    batch_idx = torch.arange(B, device=device).unsqueeze(1)
    beam_idx = torch.arange(BEAM_ACT, device=device).unsqueeze(0)
    beam_ids[batch_idx, beam_idx, d0] = t0
    beam_lp[:, :] = best0

    if steps == 2:
        return _final_combine(model, encoder_hidden, tokenizer, beam_ids, beam_lp,
                              TOP_K_FINAL, BEAM_ACT, per_digit_topL)


    mask_pos = (beam_ids == MASK_ID).float()
    dec_in = torch.clamp(beam_ids, min=0).view(-1, n_digit)
    mp_flat = mask_pos.view(-1, n_digit)
    batch1 = {
        'decoder_input_ids': dec_in,
        'encoder_hidden': encoder_hidden.unsqueeze(1).repeat(1, BEAM_ACT, 1, 1).view(-1, encoder_hidden.size(1), encoder_hidden.size(2)),
        'mask_positions': mp_flat
    }
    out1 = model.forward_decoder_only(batch1, digit=None, use_cache=False)
    lp1 = F.log_softmax(out1.logits, dim=-1).view(B, BEAM_ACT, n_digit, VOC)

    masked = lp1 + (1.0 - mask_pos.unsqueeze(-1)) * NEG_INF
    cand = beam_lp.unsqueeze(-1).unsqueeze(-1) + masked
    flat = cand.view(B, -1)
    best1, idx1 = torch.topk(flat, k=BEAM_ACT, dim=-1)
    parent = idx1 // (n_digit * VOC)
    remain = idx1 % (n_digit * VOC)
    d1 = remain // VOC
    t1 = remain % VOC

    new_ids = beam_ids[batch_idx, parent].clone()
    new_ids.scatter_(2, d1.unsqueeze(-1), t1.unsqueeze(-1))
    beam_ids = new_ids
    beam_lp = best1

    return _final_combine(model, encoder_hidden, tokenizer, beam_ids, beam_lp,
                          TOP_K_FINAL, BEAM_ACT, per_digit_topL)


@torch.no_grad()
def _final_combine(
    model,
    encoder_hidden,
    tokenizer,
    beam_ids,
    beam_lp,
    top_k_final: int,
    beam_act: int,
    per_digit_topL: int,
) -> torch.Tensor:


    device = encoder_hidden.device
    B, K, D = beam_ids.shape
    MASK_ID = -1

    mask_pos = (beam_ids == MASK_ID).float()
    dec_in = torch.clamp(beam_ids, min=0).view(-1, D)
    mp_flat = mask_pos.view(-1, D)
    batch = {
        'decoder_input_ids': dec_in,
        'encoder_hidden': encoder_hidden.unsqueeze(1).repeat(1, K, 1, 1).view(-1, encoder_hidden.size(1), encoder_hidden.size(2)),
        'mask_positions': mp_flat
    }
    out = model.forward_decoder_only(batch, digit=None, use_cache=False)
    lp = F.log_softmax(out.logits, dim=-1).view(B, K, D, -1)

    r_mask = mask_pos.bool()
    bb = B * K
    V = lp.size(-1)


    per_bb = torch.stack([lp[:, :, d, :].reshape(bb, V) for d in range(D)], dim=1)


    mask_flat = r_mask.view(bb, D)
    r_per = mask_flat.sum(dim=1)
    assert float(r_per.min().item()) == float(r_per.max().item()), "remaining-column count must be equal across beams"
    r = int(r_per[0].item())

    order = torch.argsort(mask_flat.float(), dim=1, descending=True)
    gather_index = order.unsqueeze(-1).expand(-1, -1, V)
    per_bb = per_bb.gather(dim=1, index=gather_index)[:, :r, :]


    comb_vals, comb_tok = combine_remaining_topk(per_bb, topK_final=beam_act, per_digit_topL=per_digit_topL)
    parent_lp = beam_lp.reshape(bb).unsqueeze(1)
    total_lp = (parent_lp + comb_vals).reshape(B, K * beam_act)
    best, besti = torch.topk(total_lp, k=beam_act, dim=-1)


    final = []
    for b in range(B):
        for kidx in range(beam_act):
            parent = (besti[b, kidx] // beam_act).item()
            sid = beam_ids[b, parent].clone()
            maskb = r_mask[b, parent].clone()
            combos = comb_tok.view(B, K, beam_act, -1)[b, parent]
            chosen = combos[besti[b, kidx] % beam_act]
            it = 0
            for d in range(D):
                if maskb[d]:
                    sid[d] = chosen[it]
                    it += 1
            final.append(sid)
    final = torch.stack(final, dim=0).reshape(B, beam_act, D)

    return _post_select(final, best, tokenizer, top_k_final)


def _post_select(seqs: torch.Tensor, scores: torch.Tensor, tokenizer, top_k_final: int) -> torch.Tensor:


    B, K, D = seqs.shape
    out = []
    for b in range(B):
        cand = seqs[b]
        lp = scores[b]
        order = torch.argsort(lp, descending=True)
        uniq = []
        for idx in order:
            s = cand[idx]
            legal = tokenizer.codebooks_to_item_id(s.tolist()) is not None
            if not legal:
                continue
            if not any(torch.equal(s, u) for u in uniq):
                uniq.append(s)
            if len(uniq) >= top_k_final:
                break
        if not uniq:
            uniq = [cand[order[0]]]
        while len(uniq) < top_k_final:
            uniq.append(uniq[-1])
        out.append(torch.stack(uniq))
    return torch.stack(out, dim=0)
