

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from genrec.model import AbstractModel
from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer
from .ablate_decode import decode_ablate_confidence


def make_norm(norm_type: str, dim: int, eps: float):
    if (norm_type or "layernorm").lower() == "rmsnorm":
        return nn.RMSNorm(dim, eps=eps)
    return nn.LayerNorm(dim, eps=eps)


class MultiHeadAttention(nn.Module):

    def __init__(self, emb_dim, n_head, attn_drop=0.1, resid_drop=0.1):
        super().__init__()
        assert emb_dim % n_head == 0
        self.n_head = n_head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // n_head


        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.resid_dropout = nn.Dropout(resid_drop)


        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x, attention_mask=None, key_value=None, past_key_value=None, use_cache=False, is_decoder_self_attn=False):
        B, T, C = x.size()

        if key_value is not None:

            q = self.qkv(x)[:, :, :self.emb_dim]
            k, v = key_value.chunk(2, dim=-1)
            T_kv = k.size(1)
        else:

            q, k, v = self.qkv(x).chunk(3, dim=-1)
            T_kv = T


        if past_key_value is not None and use_cache and is_decoder_self_attn:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
            T_kv = k.size(1)


        k_for_cache = k
        v_for_cache = v


        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)


        scale = 1.0 / (self.head_dim ** 0.5)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale


        if attention_mask is not None:

            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            att = att.masked_fill(attention_mask == 0, float('-inf'))

            all_inf = torch.isinf(att).all(dim=-1, keepdim=True)
            if all_inf.any():
                att = att.masked_fill(all_inf, 0.0)

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)


        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)


        y = self.resid_dropout(self.proj(y))


        present_key_value = (k_for_cache, v_for_cache) if use_cache else None

        return y, present_key_value


class FeedForward(nn.Module):

    def __init__(self, emb_dim, n_inner, resid_drop=0.1, act='gelu'):
        super().__init__()
        self.c_fc = nn.Linear(emb_dim, n_inner)
        self.c_proj = nn.Linear(n_inner, emb_dim)
        self.dropout = nn.Dropout(resid_drop)
        self.act = F.gelu if act == 'gelu' else F.relu

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):

    def __init__(self, emb_dim, n_head, n_inner, attn_drop=0.1, resid_drop=0.1,
                 act='gelu', norm_type='layernorm', norm_eps=1e-6):
        super().__init__()
        self.ln_1 = make_norm(norm_type, emb_dim, norm_eps)
        self.attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_2 = make_norm(norm_type, emb_dim, norm_eps)
        self.mlp = FeedForward(emb_dim, n_inner, resid_drop, act)

    def forward(self, x, attention_mask=None):

        attn_output, _ = self.attn(self.ln_1(x), attention_mask=attention_mask, is_decoder_self_attn=False)
        x = x + attn_output


        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderBlock(nn.Module):

    def __init__(self, emb_dim, n_head, n_inner, attn_drop=0.1, resid_drop=0.1,
                 act='gelu', norm_type='layernorm', norm_eps=1e-6):
        super().__init__()
        self.ln_1 = make_norm(norm_type, emb_dim, norm_eps)
        self.self_attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_2 = make_norm(norm_type, emb_dim, norm_eps)
        self.cross_attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_3 = make_norm(norm_type, emb_dim, norm_eps)
        self.mlp = FeedForward(emb_dim, n_inner, resid_drop, act)

    def forward(self, x, encoder_hidden=None, attention_mask=None,
                past_key_value=None, use_cache=False, cross_key_value=None):


        self_past_kv = None
        cross_past_kv = None
        if past_key_value is not None:
            if len(past_key_value) >= 1:
                self_past_kv = past_key_value[0]
            if len(past_key_value) >= 2:
                cross_past_kv = past_key_value[1]

        attn_output, present_key_value = self.self_attn(
            self.ln_1(x),
            attention_mask=None,
            past_key_value=self_past_kv,
            use_cache=use_cache,
            is_decoder_self_attn=True
        )
        x = x + attn_output


        if encoder_hidden is not None:
            if cross_key_value is not None:

                encoder_kv = cross_key_value
            else:

                encoder_kv = torch.cat([encoder_hidden, encoder_hidden], dim=-1)

            cross_attn_output, cross_present = self.cross_attn(
                self.ln_2(x),
                key_value=encoder_kv,
                past_key_value=cross_past_kv,
                use_cache=use_cache
            )
            x = x + cross_attn_output

            if use_cache:
                present_key_value = (present_key_value, cross_present)


        x = x + self.mlp(self.ln_3(x))

        return_dict = {}
        return_dict['hidden_states'] = x
        if use_cache:
            return_dict['present_key_value'] = present_key_value

        return return_dict


class ModelOutput:

    def __init__(self):
        self.loss = None
        self.logits = None
        self.hidden_states = None
        self.past_key_values = None


class DIFF_GRM(AbstractModel):

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super().__init__(config, dataset, tokenizer)

        self.config = config
        self.tokenizer = tokenizer
        self.n_digit = config['n_digit']
        self.codebook_size = config['codebook_size']
        self.vocab_size = tokenizer.vocab_size


        self.n_embd = config['n_embd']
        self.n_head = config['n_head']
        self.n_inner = config['n_inner']
        self.dropout = config['dropout']


        self.encoder_n_layer = config['encoder_n_layer']
        self.decoder_n_layer = config['decoder_n_layer']


        self.norm_type = (config.get('norm_type', 'layernorm') or 'layernorm').lower()
        self.norm_eps  = float(config.get('norm_eps', 1e-6 if self.norm_type=='rmsnorm' else 1e-5))


        self.masking_strategy = config.get('masking_strategy', 'random')

        if self.masking_strategy == 'sequential':

            seq_cfg = config.get('sequential_steps', 'auto')
            self.seq_steps = self.n_digit if seq_cfg in (None, 'auto') else int(seq_cfg)
            assert 1 <= self.seq_steps <= self.n_digit, \
                f"sequential_steps must be 1~{self.n_digit}, got {self.seq_steps}"


            self.sequential_paths = config.get('sequential_paths', 1)
            assert self.sequential_paths >= 1, \
                f"sequential_paths must be >= 1, got {self.sequential_paths}"

            self.augment_factor = self.seq_steps * self.sequential_paths
            print(f"[MODEL] ▶ use SEQUENTIAL views: steps={self.seq_steps}, "
                  f"paths={self.sequential_paths}, augment_factor={self.augment_factor}")

            self.mask_probs = None
        elif self.masking_strategy == 'guided':

            guided_cfg = config.get('guided_steps', 'auto')
            self.guided_steps = self.n_digit if guided_cfg in (None, 'auto') else int(guided_cfg)

            self.guided_steps = min(self.guided_steps, self.n_digit, 4)
            self.guided_conf_metric = config.get('guided_conf_metric', 'msp')
            assert self.guided_conf_metric in ('msp', 'entropy'), \
                f"guided_conf_metric must be one of ['msp','entropy'], got {self.guided_conf_metric}"

            self.guided_select = config.get('guided_select', 'most')
            assert self.guided_select in ('most', 'least'), \
                f"guided_select must be one of ['most','least'], got {self.guided_select}"
            self.augment_factor = self.guided_steps
            print(f"[MODEL] ▶ GUIDED: steps={self.guided_steps}, metric={self.guided_conf_metric}, "
                  f"select={self.guided_select}, augment_factor={self.augment_factor}")
            self.mask_probs = None
        else:


            self.mask_prob_random = bool(config.get('mask_prob_random', False))
            if self.mask_prob_random:
                low = float(config.get('mask_prob_random_min', 0.0))
                high = float(config.get('mask_prob_random_max', 1.0))
                if not (0.0 <= low <= high <= 1.0):
                    raise ValueError(
                        f"mask_prob_random_min/max must satisfy 0.0 <= min <= max <= 1.0, got min={low}, max={high}"
                    )
                sampled_prob = float(np.random.uniform(low, high))

                self.augment_factor = 1
                self.mask_probs = [sampled_prob]
                self.sampled_mask_prob = sampled_prob
                print(
                    f"[MODEL] Using RANDOMLY-SAMPLED masking prob: {sampled_prob:.4f} (range [{low}, {high}]); disable multi-view (augment_factor=1)"
                )
            elif 'mask_probs' in config and config['mask_probs'] is not None:

                mask_probs_raw = config['mask_probs']

                if isinstance(mask_probs_raw, str):

                    self.mask_probs = [float(p.strip()) for p in mask_probs_raw.split(',')]
                elif isinstance(mask_probs_raw, (list, tuple)):

                    self.mask_probs = [float(p) for p in mask_probs_raw]
                elif isinstance(mask_probs_raw, (int, float)):

                    self.mask_probs = [float(mask_probs_raw)]
                else:

                    try:
                        mask_probs_str = str(mask_probs_raw)
                        self.mask_probs = [float(p.strip()) for p in mask_probs_str.split(',')]
                    except (ValueError, AttributeError):
                        raise ValueError(f"Cannot parse mask_probs: {mask_probs_raw} (type: {type(mask_probs_raw)}). "
                                       "Expected string like '1.0,0.75,0.5,0.25' or list like [1.0, 0.75, 0.5, 0.25]")

                self.augment_factor = len(self.mask_probs)
                print(f"[MODEL] Using multi-probability masking: {self.mask_probs}")
            else:

                mask_prob = config.get('mask_prob', 0.5)
                self.augment_factor = config.get('augment_factor', 4)
                self.mask_probs = [float(mask_prob)] * self.augment_factor
                print(f"[MODEL] Using single-probability masking: {mask_prob} x {self.augment_factor}")


        if self.masking_strategy == 'random' and self.mask_probs is not None:
            for i, prob in enumerate(self.mask_probs):
                if not (0.0 <= prob <= 1.0):
                    raise ValueError(f"mask_probs[{i}] = {prob} is not in valid range [0.0, 1.0]")


        self.embedding = nn.Embedding(self.vocab_size, self.n_embd)


        self.item_mlp = nn.Sequential(
            nn.Linear(self.n_digit * self.n_embd, self.n_embd),
            nn.ReLU(),
            nn.Linear(self.n_embd, self.n_embd)
        )


        self.mask_emb_table = nn.Embedding(self.n_digit, self.n_embd)


        self.max_history_len = config.get('max_history_len', 50)
        self.pos_emb_enc = nn.Embedding(self.max_history_len, self.n_embd)


        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                self.n_embd, self.n_head, self.n_inner,
                config['attn_pdrop'], config['resid_pdrop'],
                act='gelu',
                norm_type=self.norm_type, norm_eps=self.norm_eps
            )
            for _ in range(self.encoder_n_layer)
        ])


        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                self.n_embd, self.n_head, self.n_inner,
                config['attn_pdrop'], config['resid_pdrop'],
                act='gelu',
                norm_type=self.norm_type, norm_eps=self.norm_eps
            )
            for _ in range(self.decoder_n_layer)
        ])


        self.ln_f = make_norm(self.norm_type, self.n_embd, self.norm_eps)


        share_out = self.config.get('share_decoder_output_embedding', True)
        if share_out:

            self.output_adapter = nn.Identity()
            print(f"[DIFF_GRM] Using shared embedding dot-product output layer")
        else:

            self.output_adapter = nn.Linear(self.n_embd, self.n_embd, bias=False)
            print(f"[DIFF_GRM] Using independent Linear output adapter")


        self.drop = nn.Dropout(self.dropout)


        self.apply(self._init_weights)


        ab_cfg = self.config.get('ablate_decode', {}) or {}
        if bool(ab_cfg.get('enabled', False)):
            modes = list(self.config.get('beam_search_modes', []) or [])
            to_add = ['confidence_s1', 'confidence_s2', 'confidence_s3']
            if 'confidence' in modes:
                base = modes.index('confidence')
                for i, m in enumerate(to_add, 1):
                    if m not in modes:
                        modes.insert(base + i, m)
            else:
                for m in reversed(to_add):
                    if m not in modes:
                        modes.insert(0, m)
            self.config['beam_search_modes'] = modes

    def resample_mask_prob_if_needed(self):


        if self.masking_strategy == 'random' and getattr(self, 'mask_prob_random', False):
            low = float(getattr(self, 'mask_prob_random_min', 0.0)) if hasattr(self, 'mask_prob_random_min') else float(self.config.get('mask_prob_random_min', 0.0))
            high = float(getattr(self, 'mask_prob_random_max', 1.0)) if hasattr(self, 'mask_prob_random_max') else float(self.config.get('mask_prob_random_max', 1.0))
            if not (0.0 <= low <= high <= 1.0):
                raise ValueError(
                    f"mask_prob_random_min/max must satisfy 0.0 <= min <= max <= 1.0, got min={low}, max={high}"
                )
            sampled_prob = float(np.random.uniform(low, high))
            self.mask_probs = [sampled_prob]
            self.sampled_mask_prob = sampled_prob
            print(f"[MODEL] [Epoch-Resample] RANDOM masking prob resampled to {sampled_prob:.4f} (range [{low}, {high}]); augment_factor=1")

    def set_masking_mode(self, strategy: str, **kw):


        self.masking_strategy = strategy

        if strategy == 'sequential':

            seq_cfg = kw.get('sequential_steps', self.config.get('sequential_steps', 'auto'))
            self.seq_steps = self.n_digit if seq_cfg in (None, 'auto') else int(seq_cfg)
            self.sequential_paths = int(kw.get('sequential_paths', self.config.get('sequential_paths', 1)))
            self.augment_factor = self.seq_steps * self.sequential_paths
            self.mask_probs = None
            print(f"[SCHEDULE] → SEQUENTIAL: steps={self.seq_steps}, paths={self.sequential_paths}, augment_factor={self.augment_factor}")

        elif strategy == 'guided':
            guided_cfg = kw.get('guided_steps', self.config.get('guided_steps', 'auto'))
            self.guided_steps = self.n_digit if guided_cfg in (None, 'auto') else int(guided_cfg)
            self.guided_steps = min(self.guided_steps, self.n_digit, 4)
            self.guided_conf_metric = kw.get('guided_conf_metric', self.config.get('guided_conf_metric', 'msp'))
            self.guided_select = kw.get('guided_select', self.config.get('guided_select', 'least'))

            self.config['guided_refresh_each_step'] = bool(kw.get(
                'guided_refresh_each_step',
                self.config.get('guided_refresh_each_step', False)
            ))
            self.augment_factor = self.guided_steps
            self.mask_probs = None
            print(f"[SCHEDULE] → GUIDED({self.guided_select}): steps={self.guided_steps}, metric={self.guided_conf_metric}, refresh={self.config['guided_refresh_each_step']}, augment_factor={self.augment_factor}")

        elif strategy == 'random':

            self.mask_prob_random = bool(kw.get('mask_prob_random', self.config.get('mask_prob_random', False)))
            if self.mask_prob_random:
                self.mask_probs = [float(np.random.uniform(
                    float(kw.get('mask_prob_random_min', self.config.get('mask_prob_random_min', 0.0))),
                    float(kw.get('mask_prob_random_max', self.config.get('mask_prob_random_max', 1.0)))
                ))]
                self.augment_factor = 1
            else:
                if 'mask_probs' in kw and kw['mask_probs'] is not None:
                    self.mask_probs = [float(p) for p in (kw['mask_probs'] if isinstance(kw['mask_probs'], (list, tuple)) else str(kw['mask_probs']).split(','))]
                    self.augment_factor = len(self.mask_probs)
                else:
                    mp = float(kw.get('mask_prob', self.config.get('mask_prob', 0.5)))
                    af = int(kw.get('augment_factor', self.config.get('augment_factor', 4)))
                    self.mask_probs = [mp] * af
                    self.augment_factor = af
            print(f"[SCHEDULE] → RANDOM: mask_probs={self.mask_probs}, augment_factor={self.augment_factor}")
        else:
            raise ValueError(f"Unknown masking strategy: {strategy}")

    def _compute_digit_logits(self, hidden_last, digit):


        if digit is None:
            raise ValueError("digit参数不能为None，必须指定要计算的codebook位置")

        if digit >= self.n_digit:
            raise ValueError(f"digit={digit} 超出范围，应该在 [0, {self.n_digit-1}]")


        start = self.tokenizer.sid_offset + digit * self.codebook_size
        end = start + self.codebook_size

        E_sub = self.embedding.weight[start:end]


        h = self.output_adapter(hidden_last)


        logits = torch.matmul(h, E_sub.t())

        return logits

    @property
    def n_parameters(self) -> str:


        n_params = sum(p.numel() for p in self.parameters())
        return f"{n_params:,}"

    def forward(self, batch: dict, return_loss=True) -> ModelOutput:


        device = next(self.parameters()).device


        if hasattr(self, '_debug_printed'):
            pass
        else:
            print(f"[DIFF_GRM] Using RPG_ED-style encoder: MLP compression + fixed 50-length sequence")
            print(f"[DIFF_GRM] vocab_size: {self.vocab_size}, codebook_size: {self.codebook_size}")
            print(f"[DIFF_GRM] masking_strategy: {self.masking_strategy}")
            if self.masking_strategy == 'random' and self.mask_probs is not None:
                print(f"[DIFF_GRM] mask_probs: {self.mask_probs}")
            self._debug_printed = True


        history_sid = batch['history_sid'].to(device)
        B, seq_len, n_digit = history_sid.shape


        valid_hist = ((history_sid == -1) | ((history_sid >= 0) & (history_sid < self.codebook_size))).all()
        assert bool(valid_hist), \
            f"history_sid 应为 codebook id(0..{self.codebook_size-1}) 或 -1(PAD)，但发现越界值"


        history_tokens = torch.zeros(B, seq_len, n_digit, dtype=torch.long, device=device)
        for d in range(n_digit):

            codebook_ids = history_sid[:, :, d]
            token_ids = torch.where(
                codebook_ids == -1,
                torch.zeros_like(codebook_ids),
                codebook_ids + self.tokenizer.sid_offset + d * self.codebook_size
            )

            token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
            history_tokens[:, :, d] = token_ids


        tok_emb = self.embedding(history_tokens)
        B, S, _, d = tok_emb.shape


        item_emb = tok_emb.reshape(B, S, self.n_digit * d)
        item_emb = self.item_mlp(item_emb)


        pos_ids = torch.arange(S, device=item_emb.device)
        pos_emb = self.pos_emb_enc(pos_ids)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)


        encoder_hidden = item_emb + pos_emb
        encoder_hidden = self.drop(encoder_hidden)


        if 'history_mask' in batch:
            history_mask = batch['history_mask'].to(device)

            attention_mask = history_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(-1, -1, seq_len, -1)
        else:
            attention_mask = None


        encoder_hidden = encoder_hidden
        for block in self.encoder_blocks:
            encoder_hidden = block(encoder_hidden, attention_mask=attention_mask)

        encoder_hidden = self.ln_f(encoder_hidden)


        if 'history_mask' in batch:
            history_mask = batch['history_mask'].to(device)
            encoder_hidden = encoder_hidden * history_mask.unsqueeze(-1).float()

        if not return_loss:

            output = ModelOutput()
            output.hidden_states = encoder_hidden
            return output


        decoder_input_ids = batch['decoder_input_ids'].to(device)
        decoder_labels = batch['decoder_labels'].to(device)


        decoder_input_ids = torch.clamp(decoder_input_ids, 0, self.codebook_size - 1)
        decoder_labels = torch.clamp(decoder_labels, 0, self.codebook_size - 1)


        all_masked_input_ids = []
        all_labels = []
        all_mask_positions = []
        all_encoder_hidden = []

        if self.masking_strategy == 'sequential':

            for p in range(self.sequential_paths):

                orders = torch.argsort(torch.rand(B, self.n_digit, device=device), dim=1)


                full_mask = torch.ones(B, self.n_digit, dtype=torch.bool, device=device)
                inp0 = decoder_input_ids.new_zeros(B, self.n_digit)
                all_masked_input_ids.append(inp0)
                all_labels.append(decoder_labels)
                all_mask_positions.append(full_mask.float())
                all_encoder_hidden.append(encoder_hidden)


                for reveal in range(1, self.seq_steps):
                    mask_pos = torch.ones_like(full_mask)


                    reveal_idx = orders[:, :reveal]
                    mask_pos.scatter_(1, reveal_idx, 0)

                    inp = decoder_input_ids.clone()
                    inp[mask_pos] = 0

                    all_masked_input_ids.append(inp)
                    all_labels.append(decoder_labels)
                    all_mask_positions.append(mask_pos.float())
                    all_encoder_hidden.append(encoder_hidden)
        elif self.masking_strategy == 'guided':
            B = decoder_labels.size(0)
            device = decoder_labels.device

            def score_with_mask(cur_mask: torch.Tensor):

                cur_inp = decoder_input_ids.new_zeros(B, self.n_digit)
                cur_inp[~cur_mask] = decoder_labels[~cur_mask]

                _was_training = self.training
                self.eval()
                with torch.no_grad():
                    if B == 1:
                        print(f"[GUIDED] scoring: self.training={self.training}")
                    logits = self.forward_decoder_only(
                        {
                            'decoder_input_ids': cur_inp,
                            'encoder_hidden': encoder_hidden,
                            'mask_positions': cur_mask.float()
                        },
                        return_loss=False, digit=None, use_cache=False
                    ).logits
                if _was_training:
                    self.train()


                probs = F.softmax(logits, dim=-1)
                if self.guided_conf_metric == 'entropy':
                    ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                    conf = -ent
                else:
                    conf = probs.max(dim=-1).values

                return conf

            refresh = str(self.config.get('guided_refresh_each_step', False)).lower() in ('1','true','yes','y')
            all_masked_input_ids, all_labels, all_mask_positions, all_encoder_hidden = [], [], [], []

            if not refresh:

                full_mask = torch.ones(B, self.n_digit, dtype=torch.bool, device=device)
                conf = score_with_mask(full_mask)
                if self.guided_select == 'most':
                    order = torch.argsort(conf, 1, True)
                else:
                    order = torch.argsort(conf, 1, False)

                for t in range(1, self.guided_steps + 1):
                    cur_mask = torch.zeros(B, self.n_digit, dtype=torch.bool, device=device)
                    cols = order[:, :t]
                    cur_mask.scatter_(1, cols, True)

                    cur_inp = decoder_input_ids.new_zeros(B, self.n_digit)
                    cur_inp[~cur_mask] = decoder_labels[~cur_mask]

                    all_masked_input_ids.append(cur_inp)
                    all_labels.append(decoder_labels)
                    all_mask_positions.append(cur_mask.float())
                    all_encoder_hidden.append(encoder_hidden)
            else:

                cur_mask = torch.zeros(B, self.n_digit, dtype=torch.bool, device=device)
                for t in range(1, self.guided_steps + 1):
                    conf = score_with_mask(cur_mask)


                    if self.guided_select == 'most':
                        conf = conf.masked_fill(cur_mask, -1e9)
                        cols = torch.argmax(conf, dim=1, keepdim=True)
                    else:
                        conf = conf.masked_fill(cur_mask,  1e9)
                        cols = torch.argmin(conf, dim=1, keepdim=True)

                    cur_mask.scatter_(1, cols, True)

                    cur_inp = decoder_input_ids.new_zeros(B, self.n_digit)
                    cur_inp[~cur_mask] = decoder_labels[~cur_mask]

                    all_masked_input_ids.append(cur_inp)
                    all_labels.append(decoder_labels)
                    all_mask_positions.append(cur_mask.float())
                    all_encoder_hidden.append(encoder_hidden)

        else:


            batch_mask_prob = None
            if getattr(self, 'mask_prob_random', False):
                low = float(self.config.get('mask_prob_random_min', 0.0))
                high = float(self.config.get('mask_prob_random_max', 1.0))

                batch_mask_prob = float(torch.empty(1).uniform_(low, high).item())
            for view_idx, mask_prob in enumerate(self.mask_probs):
                if batch_mask_prob is not None:
                    mask_prob = batch_mask_prob

                mask_positions = torch.rand(B, self.n_digit, device=device) < mask_prob


                no_mask_samples = ~mask_positions.any(dim=1)
                if no_mask_samples.any():

                    mask_positions[no_mask_samples, 0] = True


                masked_input_ids = decoder_input_ids.clone()
                masked_input_ids[mask_positions] = 0


                all_masked_input_ids.append(masked_input_ids)
                all_labels.append(decoder_labels)
                all_mask_positions.append(mask_positions.float())
                all_encoder_hidden.append(encoder_hidden)


        decoder_input_ids = torch.cat(all_masked_input_ids, dim=0)
        decoder_labels = torch.cat(all_labels, dim=0)
        mask_positions = torch.cat(all_mask_positions, dim=0)
        encoder_hidden = torch.cat(all_encoder_hidden, dim=0)


        B_expanded = B * self.augment_factor


        assert decoder_input_ids.shape[0] == B_expanded, f"decoder_input_ids shape mismatch: {decoder_input_ids.shape[0]} vs {B_expanded}"
        assert decoder_labels.shape[0] == B_expanded, f"decoder_labels shape mismatch: {decoder_labels.shape[0]} vs {B_expanded}"
        assert mask_positions.shape[0] == B_expanded, f"mask_positions shape mismatch: {mask_positions.shape[0]} vs {B_expanded}"
        assert encoder_hidden.shape[0] == B_expanded, f"encoder_hidden shape mismatch: {encoder_hidden.shape[0]} vs {B_expanded}"


        if self.masking_strategy == 'guided':
            m = mask_positions.view(B, self.augment_factor, self.n_digit).sum(-1)
            assert torch.all(m[:, 1:] >= m[:, :-1]), "guided views should increase masked count monotonically"


        encoder_kv_list = []
        for blk in self.decoder_blocks:

            kv_proj = blk.cross_attn.qkv(encoder_hidden)

            k = kv_proj[..., self.n_embd:2*self.n_embd]
            v = kv_proj[..., 2*self.n_embd:]

            layer_kv = torch.cat([k, v], dim=-1)
            encoder_kv_list.append(layer_kv)


        decoder_emb = torch.zeros(B_expanded, self.n_digit, self.n_embd, device=device)

        for d in range(self.n_digit):

            codebook_ids = decoder_input_ids[:, d]


            token_ids = codebook_ids + self.tokenizer.sid_offset + d * self.codebook_size
            token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)


            token_emb = self.embedding(token_ids)


            mask_emb = self.mask_emb_table.weight[d]
            mask_emb = mask_emb.unsqueeze(0).expand(B_expanded, -1)


            is_masked = mask_positions[:, d].unsqueeze(-1)
            decoder_emb[:, d, :] = torch.where(is_masked.bool(), mask_emb, token_emb)


        decoder_emb = self.drop(decoder_emb)


        decoder_hidden = decoder_emb
        for i, block in enumerate(self.decoder_blocks):
            block_output = block(
                decoder_hidden,
                encoder_hidden=encoder_hidden,
                past_key_value=None,
                use_cache=False,
                cross_key_value=encoder_kv_list[i]
            )
            decoder_hidden = block_output['hidden_states']

        decoder_hidden = self.ln_f(decoder_hidden)


        if self.masking_strategy == 'random' and getattr(self, 'mask_prob_random', False):


            per_sample_loss = torch.zeros(B_expanded, device=device)
            for d in range(self.n_digit):
                logits_d = self._compute_digit_logits(decoder_hidden[:, d, :], digit=d)
                labels_d = decoder_labels[:, d]
                mask_d = mask_positions[:, d].float()
                loss_d = F.cross_entropy(
                    logits_d, labels_d, reduction='none',
                    label_smoothing=self.config.get('label_smoothing', 0.1)
                )
                per_sample_loss += loss_d * mask_d

            t_actual = mask_positions.float().mean(dim=1)
            t_actual = torch.clamp(t_actual, min=1e-6)
            total_loss = (per_sample_loss / t_actual).mean()
        else:

            total_loss = 0.0
            total_weight = 0.0
            for d in range(self.n_digit):
                logits_d = self._compute_digit_logits(decoder_hidden[:, d, :], digit=d)
                labels_d = decoder_labels[:, d]
                mask_d = mask_positions[:, d].float()
                loss_d = F.cross_entropy(
                    logits_d, labels_d, reduction='none',
                    label_smoothing=self.config.get('label_smoothing', 0.1)
                )
                total_loss += (loss_d * mask_d).sum()
                total_weight += mask_d.sum()
            if total_weight > 0:
                total_loss = total_loss / total_weight
            else:
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        output = ModelOutput()
        output.loss = total_loss
        output.hidden_states = decoder_hidden
        output.logits = None

        return output

    def forward_decoder_only(self, batch: dict, return_loss=False, digit=None,
                            past_key_values=None, use_cache=False) -> ModelOutput:


        device = next(self.parameters()).device

        decoder_input_ids = batch['decoder_input_ids'].to(device)
        encoder_hidden = batch['encoder_hidden'].to(device)
        B, n_digit = decoder_input_ids.shape


        if 'mask_positions' in batch:
            mask_positions = batch['mask_positions'].to(device)
        else:
            mask_positions = torch.zeros(B, n_digit, device=device)


        encoder_kv_list = None

        if past_key_values is None and use_cache:

            encoder_kv_list = []
            for blk in self.decoder_blocks:
                with torch.no_grad():
                    kv_proj = blk.cross_attn.qkv(encoder_hidden)
                    k = kv_proj[..., self.n_embd:2*self.n_embd]
                    v = kv_proj[..., 2*self.n_embd:]
                    layer_kv = torch.cat([k, v], dim=-1)
                encoder_kv_list.append(layer_kv)
        elif past_key_values is not None:

            encoder_kv_list = []
            for layer_cache in past_key_values:
                if layer_cache is not None and len(layer_cache) >= 2:
                    _, cross_kv = layer_cache
                    if cross_kv is not None:
                        cross_key, cross_value = cross_kv
                        layer_kv = torch.cat([cross_key, cross_value], dim=-1)
                        encoder_kv_list.append(layer_kv)
                    else:
                        encoder_kv_list.append(None)
                else:
                    encoder_kv_list.append(None)


        decoder_emb = torch.zeros(B, n_digit, self.n_embd, device=device)

        for d in range(n_digit):

            token_ids = decoder_input_ids[:, d] + self.tokenizer.sid_offset + d * self.codebook_size
            token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
            token_emb = self.embedding(token_ids)


            mask_emb = self.mask_emb_table.weight[d]
            mask_emb = mask_emb.unsqueeze(0).expand(B, -1)


            is_masked = mask_positions[:, d].unsqueeze(-1)
            decoder_emb[:, d, :] = torch.where(is_masked.bool(), mask_emb, token_emb)


        decoder_emb = self.drop(decoder_emb)


        decoder_hidden = decoder_emb
        present_key_values = []

        for i, block in enumerate(self.decoder_blocks):

            layer_past = past_key_values[i] if past_key_values is not None else None


            current_cross_kv = encoder_kv_list[i] if encoder_kv_list is not None else None

            block_output = block(
                decoder_hidden,
                encoder_hidden=encoder_hidden,
                past_key_value=layer_past,
                use_cache=use_cache,
                cross_key_value=current_cross_kv
            )
            decoder_hidden = block_output['hidden_states']


            if use_cache:
                layer_present = block_output.get('present_key_value')
                if layer_present is not None and len(layer_present) >= 2:
                    self_present, cross_present = layer_present

                    if cross_present is not None:

                        layer_kv = encoder_kv_list[i] if encoder_kv_list is not None else None
                        if layer_kv is not None:
                            k, v = layer_kv.chunk(2, dim=-1)
                            cross_present = (k, v)
                    present_key_values.append((self_present, cross_present))
                else:
                    present_key_values.append(layer_present)


        if not use_cache:
            present_key_values = None

        decoder_hidden = self.ln_f(decoder_hidden)


        if digit is not None:
            logits = self._compute_digit_logits(decoder_hidden[:, digit, :], digit=digit)
        else:

            logits = []
            for d in range(n_digit):
                logits_d = self._compute_digit_logits(decoder_hidden[:, d, :], digit=d)
                logits.append(logits_d)
            logits = torch.stack(logits, dim=1)

        output = ModelOutput()
        output.hidden_states = decoder_hidden
        output.logits = logits
        output.past_key_values = present_key_values

        return output

    def generate(self, batch, n_return_sequences=1, mode="confidence"):


        from .beam import fast_beam_search_for_eval


        was_training = self.training
        self.eval()

        try:

            with torch.no_grad():
                encoder_outputs = self.forward(batch, return_loss=False)
                encoder_hidden = encoder_outputs.hidden_states


                if mode in ("confidence", "random"):
                    generated_sequences = fast_beam_search_for_eval(
                        model=self,
                        encoder_hidden=encoder_hidden,
                        beam_size=n_return_sequences,
                        max_len=self.n_digit,
                        tokenizer=self.tokenizer,
                        mode=mode,
                        rand_cfg=self.config.get("random_beam", {})
                    )
                    return generated_sequences


                if mode.startswith("confidence_s") and bool(self.config.get('ablate_decode', {}).get('enabled', False)):
                    try:
                        steps = int(mode.split("confidence_s")[-1])
                    except Exception:
                        steps = int(self.config.get('ablate_decode', {}).get('steps_default', 3))
                    if steps >= 4:

                        generated_sequences = fast_beam_search_for_eval(
                            model=self,
                            encoder_hidden=encoder_hidden,
                            beam_size=n_return_sequences,
                            max_len=self.n_digit,
                            tokenizer=self.tokenizer,
                            mode="confidence",
                            rand_cfg=self.config.get("random_beam", {})
                        )
                    else:
                        generated_sequences = decode_ablate_confidence(
                            model=self,
                            encoder_hidden=encoder_hidden,
                            tokenizer=self.tokenizer,
                            steps=steps,
                            n_return_sequences=n_return_sequences,
                        )
                    return generated_sequences


                generated_sequences = fast_beam_search_for_eval(
                    model=self,
                    encoder_hidden=encoder_hidden,
                    beam_size=n_return_sequences,
                    max_len=self.n_digit,
                    tokenizer=self.tokenizer,
                    mode="confidence",
                    rand_cfg=self.config.get("random_beam", {})
                )
                return generated_sequences

        finally:

            if was_training:
                self.train()

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):

            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.ones_(module.weight)
