from typing import List, Optional

import torch
from torch import nn


class MLP(nn.Module):


    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim_list: Optional[List[int]] = None,
        activation: nn.Module = nn.ReLU,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:


        super().__init__()

        if hidden_dim_list is None:
            hidden_dim_list = []
        hidden_dim_list.append(output_dim)
        layers = [nn.Linear(input_dim, hidden_dim_list[0], bias=bias)]
        for i in range(1, len(hidden_dim_list)):
            layers.append(activation())
            layers.append(
                nn.Linear(hidden_dim_list[i - 1], hidden_dim_list[i], bias=bias)
            )
            layers.append(nn.Dropout(dropout))
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
