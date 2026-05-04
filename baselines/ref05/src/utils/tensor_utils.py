from typing import Optional, Tuple, Union

import torch

from src.utils.file_utils import open_local_or_remote


def locations_to_index_tuple(locations: torch.Tensor, num_dims: int = 2) -> Tuple:


    return tuple(locations[:, i] for i in range(num_dims))


def extract_locations(
    data: torch.tensor, locations: torch.tensor, num_dims: int = 2
) -> torch.tensor:


    index_tuple = locations_to_index_tuple(locations=locations, num_dims=num_dims)


    extracted_values = data[index_tuple]

    return extracted_values


def merge_list_of_keyed_tensors_to_single_tensor(
    data: list[dict[str, torch.Tensor]],
    index_key: str,
    value_key: str,
) -> torch.Tensor:


    batch_size = len(data)
    dimensions = torch.tensor(data[0][value_key]).size()
    output_tensor = torch.zeros((batch_size, *dimensions))
    for row in data:
        index = row[index_key]
        value = row[value_key]
        if index < batch_size:
            output_tensor[index] = torch.tensor(value)
        else:
            raise IndexError(
                f"Index {index} out of bounds for batch size {batch_size}."
            )
    return output_tensor


def deduplicate_rows_in_tensor(
    file_path: Optional[str] = None, return_tensor: bool = False
) -> Union[None, torch.Tensor]:


    if not file_path.endswith(".pt"):
        return None
    data = torch.load(open_local_or_remote(file_path, mode="rb"))
    assert len(data.size()) == 2, "Input data must be a 2D PyTorch tensor."


    unique_rows, inverse_indices, counts = torch.unique(
        data, dim=0, return_inverse=True, return_counts=True
    )

    output_indices = torch.zeros_like(inverse_indices)


    duplicate_indices = torch.where(counts > 1)[0]

    for i in range(len(duplicate_indices)):

        num_of_collisions = counts[duplicate_indices[i]]


        indices_to_change = torch.where(inverse_indices == duplicate_indices[i])[0]


        range_to_add = torch.arange(1, num_of_collisions + 1)


        output_indices = output_indices.scatter(0, indices_to_change, range_to_add)


    result = torch.cat((data, output_indices.unsqueeze(1)), dim=1).long()
    if return_tensor:
        return result
    else:

        torch.save(result, file_path)
        return None


def transpose_tensor_from_file(
    file_path: Optional[str] = None,
    return_tensor: bool = False,
    dim1: int = -2,
    dim2: int = -1,
) -> Union[None, torch.Tensor]:


    if not file_path.endswith(".pt"):
        return None
    data = torch.load(open_local_or_remote(file_path, mode="rb"))


    result = data.transpose(dim1, dim2)
    if return_tensor:
        return result
    else:

        torch.save(result, file_path)
        return None
