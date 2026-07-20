import torch
from torch import nn
from transformers import T5Config, T5EncoderModel, T5ForConditionalGeneration

from . import data as data_utils


def _apply_special_token_ids(model):
    model.config.pad_token_id = data_utils.PAD
    model.config.eos_token_id = data_utils.EOS
    model.config.decoder_start_token_id = data_utils.BOS
    model.config.vocab_size = data_utils.VOCAB_SIZE

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = data_utils.PAD
        model.generation_config.eos_token_id = data_utils.EOS
        model.generation_config.decoder_start_token_id = data_utils.BOS


class HierarchicalMLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class HierarchicalT5EncoderModel(nn.Module):
    """T5 encoder backbone with per-hierarchy semantic-ID heads."""

    def __init__(
        self,
        pretrained: str | None = "t5-small",
        d_model: int = 512,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        d_kv: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        if pretrained:
            self.encoder = T5EncoderModel.from_pretrained(pretrained)
            self.encoder.resize_token_embeddings(data_utils.VOCAB_SIZE)
            self.encoder.config.vocab_size = data_utils.VOCAB_SIZE
            self.encoder.config.pad_token_id = data_utils.PAD
            with torch.no_grad():
                self.encoder.shared.weight.normal_(
                    mean=0.0, std=self.encoder.config.d_model ** -0.5
                )
        else:
            config = T5Config(
                vocab_size=data_utils.VOCAB_SIZE,
                d_model=d_model,
                d_ff=d_ff,
                num_layers=num_layers,
                num_heads=num_heads,
                d_kv=d_kv,
                dropout_rate=dropout,
                feed_forward_proj="gated-gelu",
                layer_norm_epsilon=1e-6,
                pad_token_id=data_utils.PAD,
                use_cache=False,
                is_encoder_decoder=False,
            )
            self.encoder = T5EncoderModel(config)

        hidden_dim = int(self.encoder.config.d_model)
        self.num_levels = int(data_utils.NUM_LEVELS)
        self.codebook_width = int(data_utils.CODEBOOK_WIDTH)
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)

        self.code_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.codebook_width, hidden_dim)
                for _ in range(self.num_levels)
            ]
        )
        self.heads = nn.ModuleList(
            [
                HierarchicalMLPHead(
                    input_dim=hidden_dim * (level + 1),
                    hidden_dim=hidden_dim,
                    output_dim=self.codebook_width,
                    dropout=dropout,
                )
                for level in range(self.num_levels)
            ]
        )

    def pool_last_item(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        lengths = attention_mask.long().sum(dim=1)
        contexts = []
        item_span = self.num_levels + 1
        for idx in range(hidden_states.size(0)):
            end = int(lengths[idx].item())
            if end <= 0:
                contexts.append(hidden_states[idx, 0].new_zeros(self.hidden_dim))
                continue
            # Each history item is encoded as [SEP, code_0, ..., code_{L-1}].
            # The historical best runs used last-item pooling over the full item span.
            start = max(0, end - item_span)
            contexts.append(hidden_states[idx, start:end].mean(dim=0))
        return torch.stack(contexts, dim=0)

    def encode_context(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self.pool_last_item(outputs.last_hidden_state, attention_mask)

    def _build_head_input(self, context: torch.Tensor, prefix_codes: torch.Tensor | None):
        if prefix_codes is None or prefix_codes.numel() == 0:
            return context

        parts = [context]
        for level in range(prefix_codes.size(1)):
            parts.append(self.code_embeddings[level](prefix_codes[:, level]))
        return torch.cat(parts, dim=-1)

    def predict_level_logits(
        self,
        context: torch.Tensor,
        level: int,
        prefix_codes: torch.Tensor | None = None,
    ):
        head_input = self._build_head_input(context, prefix_codes)
        return self.heads[level](head_input)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_codes: torch.Tensor | None = None,
    ):
        context = self.encode_context(input_ids=input_ids, attention_mask=attention_mask)
        logits = []

        for level in range(self.num_levels):
            prefix_codes = None if level == 0 else target_codes[:, :level]
            logits.append(
                self.predict_level_logits(
                    context=context,
                    level=level,
                    prefix_codes=prefix_codes,
                )
            )

        return context, logits


def create_model(
    pretrained: str | None = "t5-small",
    d_model: int = 512,
    d_ff: int = 2048,
    num_layers: int = 6,
    num_decoder_layers: int | None = None,
    num_heads: int = 8,
    d_kv: int = 64,
    dropout: float = 0.1,
    architecture: str = "seq2seq",
):
    if architecture == "hierarchical":
        return HierarchicalT5EncoderModel(
            pretrained=pretrained,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
            d_kv=d_kv,
            dropout=dropout,
        )

    if pretrained:
        model = T5ForConditionalGeneration.from_pretrained(pretrained)
        model.resize_token_embeddings(data_utils.VOCAB_SIZE)
        _apply_special_token_ids(model)

        with torch.no_grad():
            model.shared.weight.normal_(mean=0.0, std=model.config.d_model ** -0.5)

        return model

    config = T5Config(
        vocab_size=data_utils.VOCAB_SIZE,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_decoder_layers=num_decoder_layers or num_layers,
        num_heads=num_heads,
        d_kv=d_kv,
        dropout_rate=dropout,
        feed_forward_proj="gated-gelu",
        layer_norm_epsilon=1e-6,
        pad_token_id=data_utils.PAD,
        eos_token_id=data_utils.EOS,
        decoder_start_token_id=data_utils.BOS,
        use_cache=False,
        is_encoder_decoder=True,
    )
    model = T5ForConditionalGeneration(config)
    _apply_special_token_ids(model)
    return model


def load_checkpoint_model(ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    state_dict = ckpt["model_state_dict"]

    architecture = args.get("architecture")
    if architecture is None:
        if any(key.startswith("code_embeddings.") for key in state_dict):
            architecture = "hierarchical"
        else:
            architecture = "seq2seq"

    num_levels = args.get("num_levels")
    codebook_width = args.get("codebook_width")
    if architecture == "hierarchical" and (num_levels is None or codebook_width is None):
        head_levels = [
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("code_embeddings.") and key.endswith(".weight")
        ]
        if head_levels:
            num_levels = max(head_levels) + 1
            first_weight = state_dict[f"code_embeddings.{min(head_levels)}.weight"]
            codebook_width = int(first_weight.shape[0])

    data_utils.set_runtime_config(
        num_levels=num_levels,
        codebook_width=codebook_width,
    )

    model = create_model(
        pretrained=args.get("pretrained", "t5-small"),
        d_model=args.get("d_model", 512),
        d_ff=args.get("d_ff", 2048),
        num_layers=args.get("num_layers", 6),
        num_decoder_layers=args.get("num_decoder_layers"),
        num_heads=args.get("num_heads", 8),
        d_kv=args.get("d_kv", 64),
        dropout=args.get("dropout", 0.1),
        architecture=architecture,
    )
    model.load_state_dict(state_dict, strict=True)
    return model, ckpt
