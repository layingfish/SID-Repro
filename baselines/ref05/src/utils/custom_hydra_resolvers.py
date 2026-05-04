import ast
import operator as op
from typing import Optional

from omegaconf import DictConfig, ListConfig, OmegaConf


def remove_chars_from_string(s: str, chars: str) -> str:


    return s.translate(str.maketrans("", "", chars))


def conditional_expression(
    condition_expression, value_if_true, value_if_false, **kwargs
):


    try:

        result = eval(condition_expression, {}, kwargs)
        return value_if_true if result else value_if_false
    except Exception as e:
        raise ValueError(
            f"Error evaluating condition: {condition_expression}. Error: {e}"
        )


def extract_fields_from_list_of_dicts(
    list_of_dicts: ListConfig,
    key: str,
    default: str = None,
    filter_key: str = None,
    filter_value: str = None,
) -> ListConfig:


    if filter_key and filter_value:
        filtered_dicts = [
            d for d in list_of_dicts if d.get(filter_key) == eval(filter_value)
        ]
    else:
        filtered_dicts = list_of_dicts

    return ListConfig([d.get(key, default) for d in filtered_dicts])


def create_map_from_list_of_dicts(
    list_of_dicts: ListConfig, key: str, value: Optional[str] = None
) -> DictConfig:


    if value is None:
        return DictConfig({d[key]: d for d in list_of_dicts if key in d})

    return DictConfig(
        {d[key]: d[value] for d in list_of_dicts if key in d and value in d}
    )


def math_eval(expression: str) -> float:


    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.BitXor: op.xor,
        ast.USub: op.neg,
    }

    def eval_(node):

        match node:
            case ast.Constant(value) if isinstance(value, int):
                return value
            case ast.BinOp(left, op, right):
                return operators[type(op)](eval_(left), eval_(right))
            case ast.UnaryOp(op, operand):
                return operators[type(op)](eval_(operand))
            case _:
                raise TypeError(node)

    return eval_(ast.parse(expression, mode="eval").body)


def remove_item_from_list(input_list: ListConfig, item_to_remove: str) -> ListConfig:


    return ListConfig([item for item in input_list if item != item_to_remove])


OmegaConf.register_new_resolver("remove_chars_from_string", remove_chars_from_string)
OmegaConf.register_new_resolver("conditional_expression", conditional_expression)
OmegaConf.register_new_resolver(
    "extract_fields_from_list_of_dicts", extract_fields_from_list_of_dicts
)
OmegaConf.register_new_resolver(
    "create_map_from_list_of_dicts", create_map_from_list_of_dicts
)
OmegaConf.register_new_resolver("math_eval", math_eval)
OmegaConf.register_new_resolver("remove_item_from_list", remove_item_from_list)
