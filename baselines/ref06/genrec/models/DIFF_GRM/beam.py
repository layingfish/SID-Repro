

import torch
import torch.nn.functional as F


def _beam_step_select(mode,
                      logp_matrix,
                      cur_beam_logp,
                      beam_ids,
                      n_digit, VOC, beam_act,
                      rand_cfg):


    B = logp_matrix.size(0)

    if mode == "confidence":

        cand_lp  = cur_beam_logp.unsqueeze(-1) + logp_matrix
        flat_lp  = cand_lp.view(B, -1)
        best_lp, flat_idx = torch.topk(flat_lp, k=beam_act)
    else:

        temperature = rand_cfg.get("temperature", 1.0)
        logits = (cur_beam_logp.unsqueeze(-1) + logp_matrix) / temperature


        top_k = rand_cfg.get("top_k")
        if top_k is not None:
            kth_vals, _ = logits.topk(top_k, dim=-1)
            min_valid   = kth_vals[..., -1:].detach()
            logits      = torch.where(logits < min_valid, logits.new_full((), -1e9), logits)


        top_p = rand_cfg.get("top_p")
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)


            sorted_indices_to_remove = cumsum_probs > top_p

            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False


            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        probs = torch.softmax(logits, dim=-1)
        flat_prob = probs.view(B, -1)


        original_state = torch.get_rng_state()
        try:

            seed = rand_cfg.get("seed")
            if seed is not None:
                torch.manual_seed(seed)

            flat_idx = torch.multinomial(flat_prob, beam_act, replacement=False)
            idx_rows = torch.arange(B, device=flat_idx.device).unsqueeze(1)
            best_lp  = logits.view(B, -1)[idx_rows, flat_idx]
        finally:

            torch.set_rng_state(original_state)


    parent   = flat_idx // (n_digit * VOC)
    remain   = flat_idx %  (n_digit * VOC)
    d_pos    = remain // VOC
    tok      = remain %  VOC

    batch_idx = torch.arange(B, device=beam_ids.device).unsqueeze(1)
    next_ids  = beam_ids[batch_idx, parent].clone()
    next_ids.scatter_(2, d_pos.unsqueeze(-1), tok.unsqueeze(-1))
    return best_lp, next_ids


def expand_cross_kv_for_beams(initial_kv_cache, beam_size):


    if initial_kv_cache is None:
        return None

    expanded = []
    for layer_cache in initial_kv_cache:
        if layer_cache is None:
            expanded.append(None)
            continue

        self_kv, cross_kv = layer_cache
        if cross_kv is not None:
            k, v = cross_kv
            k = k.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, *k.shape[1:])
            v = v.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, *v.shape[1:])
            cross_kv = (k, v)

        expanded.append((None, cross_kv))
    return expanded


def iterative_mask_decode(model, encoder_hidden, n_return_sequences=1, tokenizer=None, mode="confidence", rand_cfg=None):


    device = encoder_hidden.device
    batch_size = encoder_hidden.size(0)
    n_digit = model.n_digit
    codebook_size = model.codebook_size


    if hasattr(model, 'config') and 'vectorized_beam_search' in model.config:
        beam_config = model.config['vectorized_beam_search']


        split = model.config.get("current_split", "val")


        if split in beam_config:
            BEAM_ACT = int(beam_config[split]["beam_act"])
            BEAM_MAX = int(beam_config[split]["beam_max"])
        elif isinstance(beam_config.get("beam_act"), dict):
            BEAM_ACT = int(beam_config["beam_act"].get(split,
                                                       beam_config["beam_act"]["val"]))
            BEAM_MAX = int(beam_config["beam_max"].get(split,
                                                       beam_config["beam_max"]["val"]))
        else:
            BEAM_ACT = int(beam_config["beam_act"])
            BEAM_MAX = int(beam_config["beam_max"])

        TOP_K_FINAL = min(int(beam_config['top_k_final']), n_return_sequences)

        NEG_INF_FP32 = float(beam_config['neg_inf_fp32'])
        NEG_INF_FP16 = float(beam_config['neg_inf_fp16'])

        assert BEAM_ACT <= BEAM_MAX, "beam_act should not exceed beam_max"
    else:

        raise ValueError("Missing 'vectorized_beam_search' configuration in model.config")


    if mode == "random":

        rb_cfg = model.config.get("random_beam", {})
        BEAM_ACT = int(rb_cfg.get("beam_act", BEAM_ACT))
        BEAM_MAX = int(rb_cfg.get("beam_max", BEAM_MAX))

        assert BEAM_ACT <= BEAM_MAX, "random_beam.beam_act should not exceed random_beam.beam_max"


    decode_order = None
    if mode == "random":

        original_state = torch.get_rng_state()
        try:
            seed = model.config.get("random_beam", {}).get("seed")
            if seed is not None:
                torch.manual_seed(seed)
            decode_order = torch.randperm(n_digit).tolist()
            if batch_size == 1:
                print(f"[RANDOM_BEAM] 🎲 Decode order: {decode_order}")
        finally:

            torch.set_rng_state(original_state)


    MASK_ID = tokenizer.mask_token if tokenizer is not None else -1
    VOC = codebook_size


    if batch_size == 1:
        print(f"[VECTORIZED_BEAM] 🚀 Using optimized beam search:")
        print(f"[VECTORIZED_BEAM] BEAM_ACT: {BEAM_ACT}, BEAM_MAX: {BEAM_MAX}, TOP_K_FINAL: {TOP_K_FINAL}")


    with torch.no_grad():

        mask_positions = torch.ones(batch_size, n_digit, device=device)


        batch_dict = {
            'decoder_input_ids': torch.zeros(batch_size, n_digit, device=device, dtype=torch.long),
            'encoder_hidden': encoder_hidden,
            'mask_positions': mask_positions
        }


        outputs = model.forward_decoder_only(batch_dict, digit=None, use_cache=True)
        all_logits = outputs.logits
        initial_kv_cache = outputs.past_key_values


        all_log_probs = F.log_softmax(all_logits, dim=-1)

        if mode == "random":

            first_col = decode_order[0]
            probs_col = all_log_probs[:, first_col, :]
            top_k_probs, top_k_idx = torch.topk(probs_col, k=BEAM_ACT, dim=-1)


            first_col_tensor = torch.full((batch_size, BEAM_ACT), first_col, device=device, dtype=torch.long)
            first_token = top_k_idx
        else:


            flattened_log_probs = all_log_probs.view(batch_size, -1)


            top_k_probs, top_k_indices = torch.topk(flattened_log_probs, k=BEAM_ACT)


            first_col_tensor = top_k_indices // VOC
            first_token = top_k_indices % VOC


        beam_ids = torch.full((batch_size, BEAM_MAX, n_digit), MASK_ID,
                             dtype=torch.long, device=device)


        NEG_INF = NEG_INF_FP16 if top_k_probs.dtype == torch.float16 else NEG_INF_FP32
        beam_logp = torch.full((batch_size, BEAM_MAX), NEG_INF,
                              dtype=top_k_probs.dtype, device=device)


        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        beam_indices = torch.arange(BEAM_ACT, device=device).unsqueeze(0)

        beam_ids[batch_indices, beam_indices, first_col_tensor] = first_token
        beam_logp[:, :BEAM_ACT] = top_k_probs


        encoder_hidden_expanded = encoder_hidden.unsqueeze(1).repeat(1, BEAM_MAX, 1, 1)
        encoder_hidden_expanded = encoder_hidden_expanded.view(-1, encoder_hidden.size(1), encoder_hidden.size(2))


        kv_cache_for_act = expand_cross_kv_for_beams(initial_kv_cache, BEAM_ACT)
        kv_cache_final = expand_cross_kv_for_beams(initial_kv_cache, BEAM_ACT)


    if mode == "random":

        for step, cur_col in enumerate(decode_order[1:], 1):
            with torch.no_grad():

                active_beam_ids = beam_ids[:, :BEAM_ACT, :]
                active_beam_logp = beam_logp[:, :BEAM_ACT]


                mask_positions = (active_beam_ids == MASK_ID).float()


                decoder_input = torch.clamp(active_beam_ids, min=0).view(-1, n_digit)
                mask_pos_flat = mask_positions.view(-1, n_digit)


                expanded_kv_cache = kv_cache_for_act


                batch_dict = {
                    'decoder_input_ids': decoder_input,
                    'encoder_hidden': encoder_hidden_expanded[:batch_size * BEAM_ACT],
                    'mask_positions': mask_pos_flat
                }


                outputs = model.forward_decoder_only(batch_dict, digit=None,
                                                   past_key_values=expanded_kv_cache, use_cache=True)
                all_logits = outputs.logits


                all_logits = all_logits.view(batch_size, BEAM_ACT, n_digit, codebook_size)


                all_log_probs = F.log_softmax(all_logits, dim=-1)


                mask_expanded = mask_positions.unsqueeze(-1)
                masked_log_probs = all_log_probs + (1 - mask_expanded) * NEG_INF


                logits = masked_log_probs[:, :, cur_col, :]

                joint_lp = logits + active_beam_logp.unsqueeze(-1)
                flat_lp  = joint_lp.view(batch_size, -1)
                best_lp, flat_idx = torch.topk(flat_lp, k=BEAM_ACT)


                parent_beam_ids = flat_idx // VOC
                token_ids = flat_idx % VOC


                batch_range = torch.arange(batch_size, device=device).unsqueeze(1)
                new_beam_ids = active_beam_ids[batch_range, parent_beam_ids]
                new_beam_ids.scatter_(2, torch.full((batch_size, BEAM_ACT), cur_col, device=device, dtype=torch.long).unsqueeze(-1), token_ids.unsqueeze(-1))


                beam_ids[:, :BEAM_ACT, :] = new_beam_ids
                beam_logp[:, :BEAM_ACT] = best_lp


                if BEAM_ACT < BEAM_MAX:
                    beam_ids[:, BEAM_ACT:, :] = MASK_ID
                    beam_logp[:, BEAM_ACT:] = NEG_INF
    else:

        for step in range(1, n_digit - 1):
            with torch.no_grad():

                active_beam_ids = beam_ids[:, :BEAM_ACT, :]
                active_beam_logp = beam_logp[:, :BEAM_ACT]


                mask_positions = (active_beam_ids == MASK_ID).float()


                decoder_input = torch.clamp(active_beam_ids, min=0).view(-1, n_digit)
                mask_pos_flat = mask_positions.view(-1, n_digit)


                expanded_kv_cache = kv_cache_for_act


                batch_dict = {
                    'decoder_input_ids': decoder_input,
                    'encoder_hidden': encoder_hidden_expanded[:batch_size * BEAM_ACT],
                    'mask_positions': mask_pos_flat
                }


                outputs = model.forward_decoder_only(batch_dict, digit=None,
                                                   past_key_values=expanded_kv_cache, use_cache=True)
                all_logits = outputs.logits


                all_logits = all_logits.view(batch_size, BEAM_ACT, n_digit, codebook_size)


                all_log_probs = F.log_softmax(all_logits, dim=-1)


                mask_expanded = mask_positions.unsqueeze(-1)
                masked_log_probs = all_log_probs + (1 - mask_expanded) * NEG_INF


                flattened_log_probs = masked_log_probs.view(batch_size, BEAM_ACT, -1)


                best_logprobs, new_beam_ids = _beam_step_select(
                    mode=mode,
                    logp_matrix=flattened_log_probs,
                    cur_beam_logp=active_beam_logp,
                    beam_ids=active_beam_ids,
                    n_digit=n_digit, VOC=VOC, beam_act=BEAM_ACT,
                    rand_cfg=rand_cfg or {}
                )


                beam_ids[:, :BEAM_ACT, :] = new_beam_ids
                beam_logp[:, :BEAM_ACT] = best_logprobs


                if BEAM_ACT < BEAM_MAX:
                    beam_ids[:, BEAM_ACT:, :] = MASK_ID
                    beam_logp[:, BEAM_ACT:] = NEG_INF


    with torch.no_grad():
        if mode == "random":

            active_beam_ids = beam_ids[:, :BEAM_ACT, :]
            final_beam_logp = beam_logp[:, :BEAM_ACT]
        else:


            active_beam_ids = beam_ids[:, :BEAM_ACT, :]
            active_beam_logp = beam_logp[:, :BEAM_ACT]


            mask_positions = (active_beam_ids == MASK_ID).float()


            decoder_input = torch.clamp(active_beam_ids, min=0).view(-1, n_digit)
            mask_pos_flat = mask_positions.view(-1, n_digit)


            final_expanded_kv_cache = kv_cache_final

            batch_dict = {
                'decoder_input_ids': decoder_input,
                'encoder_hidden': encoder_hidden_expanded[:batch_size * BEAM_ACT],
                'mask_positions': mask_pos_flat
            }


            outputs = model.forward_decoder_only(batch_dict, digit=None,
                                               past_key_values=final_expanded_kv_cache, use_cache=True)
            all_logits = outputs.logits


            all_logits = all_logits.view(batch_size, BEAM_ACT, n_digit, codebook_size)
            all_log_probs = F.log_softmax(all_logits, dim=-1)


            last_mask_pos = torch.argmax(mask_positions.float(), dim=-1)


            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, BEAM_ACT)
            beam_idx = torch.arange(BEAM_ACT, device=device).unsqueeze(0).expand(batch_size, -1)

            final_logits = all_log_probs[batch_idx, beam_idx, last_mask_pos]
            best_token_logprobs, best_tokens = torch.max(final_logits, dim=-1)


            active_beam_ids.scatter_(2, last_mask_pos.unsqueeze(-1), best_tokens.unsqueeze(-1))
            final_beam_logp = active_beam_logp + best_token_logprobs


        dedup_strategy = "simple"
        if hasattr(model, 'config') and 'dedup_strategy' in model.config:
            dedup_strategy = model.config['dedup_strategy']

        if dedup_strategy == "none":

            top_logprobs, top_indices = torch.topk(final_beam_logp, k=min(TOP_K_FINAL, BEAM_ACT), dim=-1)
            batch_range = torch.arange(batch_size, device=device).unsqueeze(1)
            final_sequences = active_beam_ids[batch_range, top_indices]

            if batch_size == 1:
                print(f"[VECTORIZED_BEAM] ✅ Generated {final_sequences.shape[1]} sequences (no deduplication)")

        elif dedup_strategy == "simple":


            assert tokenizer is not None, "tokenizer is required for legality check"

            final_sequences = []
            for b in range(batch_size):
                batch_sequences = active_beam_ids[b]
                batch_logprobs = final_beam_logp[b]


                sorted_indices = torch.argsort(batch_logprobs, descending=True)
                unique_sequences = []

                for idx in sorted_indices:
                    seq = batch_sequences[idx]

                    is_legal = tokenizer.codebooks_to_item_id(seq.tolist()) is not None
                    if not is_legal:
                        continue

                    is_duplicate = any(torch.equal(seq, existing) for existing in unique_sequences)
                    if not is_duplicate:
                        unique_sequences.append(seq)
                        if len(unique_sequences) >= TOP_K_FINAL:
                            break


                while len(unique_sequences) < TOP_K_FINAL:
                    if unique_sequences:

                        unique_sequences.append(unique_sequences[-1])
                    else:

                        for idx in range(BEAM_ACT):
                            seq = batch_sequences[idx]
                            if tokenizer.codebooks_to_item_id(seq.tolist()) is not None:
                                unique_sequences.append(seq)
                                break

                        if not unique_sequences:
                            unique_sequences.append(batch_sequences[0])

                batch_final = torch.stack(unique_sequences[:TOP_K_FINAL])
                final_sequences.append(batch_final)

            final_sequences = torch.stack(final_sequences)
            if batch_size == 1:
                print(f"[VECTORIZED_BEAM] ✅ Generated {final_sequences.shape[1]} unique sequences (simple deduplication + legality check)")

        else:


            assert tokenizer is not None, "tokenizer is required for legality check"

            final_sequences = []
            for b in range(batch_size):
                batch_sequences = active_beam_ids[b]
                batch_logprobs = final_beam_logp[b]


                seq_to_logprob = {}
                for i in range(BEAM_ACT):
                    seq_tuple = tuple(batch_sequences[i].cpu().tolist())

                    is_legal = tokenizer.codebooks_to_item_id(list(seq_tuple)) is not None
                    if not is_legal:
                        continue

                    if seq_tuple in seq_to_logprob:

                        seq_to_logprob[seq_tuple] = torch.logaddexp(
                            seq_to_logprob[seq_tuple],
                            batch_logprobs[i]
                        )
                    else:
                        seq_to_logprob[seq_tuple] = batch_logprobs[i]


                sorted_items = sorted(seq_to_logprob.items(),
                                    key=lambda x: x[1].item(), reverse=True)


                unique_sequences = []
                for seq_tuple, _ in sorted_items[:TOP_K_FINAL]:
                    seq_tensor = torch.tensor(seq_tuple, device=device, dtype=torch.long)
                    unique_sequences.append(seq_tensor)


                while len(unique_sequences) < TOP_K_FINAL:
                    if unique_sequences:

                        unique_sequences.append(unique_sequences[-1])
                    else:

                        for idx in range(BEAM_ACT):
                            seq = batch_sequences[idx]
                            if tokenizer.codebooks_to_item_id(seq.tolist()) is not None:
                                unique_sequences.append(seq)
                                break

                        if not unique_sequences:
                            unique_sequences.append(batch_sequences[0])

                batch_final = torch.stack(unique_sequences[:TOP_K_FINAL])
                final_sequences.append(batch_final)

            final_sequences = torch.stack(final_sequences)
            if batch_size == 1:
                print(f"[VECTORIZED_BEAM] ✅ Generated {final_sequences.shape[1]} unique sequences (probability-weighted deduplication + legality check)")


    if tokenizer is not None:

        total_seqs = final_sequences.numel() // n_digit
        legal_final = sum(tokenizer.codebooks_to_item_id(seq.tolist()) is not None
                          for seq in final_sequences.view(-1, n_digit))
        final_legal_ratio = legal_final / total_seqs


        unique_seqs = len({tuple(seq.tolist()) for seq in final_sequences.view(-1, n_digit)})
        duplicate_ratio = 1 - unique_seqs / total_seqs


        return final_sequences, final_legal_ratio, duplicate_ratio


    return final_sequences


def fast_beam_search_for_eval(model, encoder_hidden, beam_size=10, max_len=4, tokenizer=None, mode="confidence", rand_cfg=None):


    result = iterative_mask_decode(
        model=model,
        encoder_hidden=encoder_hidden,
        n_return_sequences=beam_size,
        tokenizer=tokenizer,
        mode=mode,
        rand_cfg=rand_cfg or {}
    )


    if isinstance(result, tuple):
        return result[0]
    else:
        return result
