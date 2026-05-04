import warnings
from time import sleep
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from transformers.cache_utils import DynamicCache

from src.utils import pylogger, rich_utils
from functools import partial
from tokenizers.processors import TemplateProcessing
from src.data.loading.components.interfaces import TokenizerConfig

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def print_warnings_for_missing_configs(cfg: DictConfig) -> None:
    _DEFAULT_CONFIGS = [
        "data_loading",
        "model",
        "loss",
        "optim",
        "eval",
    ]
    has_warnings = False
    for config in _DEFAULT_CONFIGS:
        if not cfg.get(config):
            log.warning(
                f"Config {config} was not found in the config tree. Make sure this is expected."
            )
            has_warnings = True
    if has_warnings:
        sleep(3)


def extras(cfg: DictConfig) -> None:


    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return


    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")


    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    if cfg.extras.get("print_config_warnings"):
        print_warnings_for_missing_configs(cfg)


    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def delete_module(module: torch.nn.Module, module_name: str) -> None:


    if hasattr(module, module_name):
        delattr(module, module_name)

    for name, submodule in module.named_children():
        delete_module(submodule, module_name)


def find_module_shape(
    module: torch.nn.Module, module_name: str
) -> Optional[torch.Size]:


    if hasattr(module, module_name):
        return getattr(module, module_name).weight.shape

    for name, submodule in module.named_children():
        shape = find_module_shape(submodule, module_name)
        if shape:
            return shape
    return None


def reset_parameters(module: torch.nn.Module) -> None:


    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    else:
        for layer in module.children():
            reset_parameters(layer)


def get_var_if_not_none(value: Optional[Any], default_value: Any) -> Any:


    return value if value is not None else default_value


def get_class_name_str(class_definition: Any) -> str:


    return ".".join([class_definition.__module__, class_definition.__name__])


def has_class_object_inside_list(obj_list: list, class_type: Any) -> bool:


    return any(isinstance(obj, class_type) for obj in obj_list)


def convert_legacy_kv_cache_to_dynamic(
    kv_cache: Union[DynamicCache, Tuple[torch.Tensor]]
) -> DynamicCache:


    if isinstance(kv_cache, DynamicCache):
        return kv_cache

    return DynamicCache.from_legacy_cache(kv_cache)


def get_parent_module_and_attr(
    model: torch.nn.Module, module_name: str
) -> Tuple[torch.nn.Module, str]:


    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def lightning_precision_to_dtype(precision: str) -> torch.dtype:


    precision_map = {
        "32": torch.float32,
        "32-true": torch.float32,
        "64": torch.float64,
        "16": torch.float16,
        "16-mixed": torch.float16,
        "bf16": torch.bfloat16,
        "half": torch.float16,
    }

    if precision in precision_map:
        return precision_map[precision]
    else:
        raise ValueError(
            f"Unsupported precision type: '{precision}'. "
            "Supported precision types are: '32', '32-true', '64', '16', '16-mixed', 'bf16', 'half'."
        )

def load_tokenize(config: TokenizerConfig) -> Any:

    tokenizer = config.tokenizer
    if hasattr(config, "special_tokens"):
        tokenizer.add_special_tokens(config.special_tokens)
    if config.postprocess_eos_token:
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single="$A " + tokenizer.eos_token,
            special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id)],
        )
    tokenize = partial(
        tokenizer.encode_plus,
        max_length=config.max_length,
        padding=config.padding,
        truncation=config.truncation,
        add_special_tokens=config.add_special_tokens,
        return_tensors="pt",
    )
    return tokenize

def sample_gumbel(shape: Tuple, device: torch.device, eps=1e-20) -> torch.Tensor:

    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:

    y = logits + sample_gumbel(logits.shape, logits.device)
    sample = F.softmax(y / temperature, dim=-1)
    return sample
