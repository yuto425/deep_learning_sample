import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleAttention(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self._query_linear = nn.Linear(depth, depth)
        self._key_linear = nn.Linear(depth, depth)
        self._value_linear = nn.Linear(depth, depth)
        self._out = nn.Linear(depth, depth)
        self._depth = depth

    def forward(self,
                input: torch.Tensor,
                memory: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        query = self._query_linear(input)
        key = self._key_linear(memory)
        value = self._value_linear(memory)
        weights = torch.matmul(
            query, key.transpose(1, 2)) / math.sqrt(self._depth)
        weights = weights.masked_fill(mask == 0, -1e10)
        normalized_weights = F.softmax(weights, dim=-1)
        attention_output = torch.matmul(normalized_weights, value)
        output = self._out(attention_output)
        return output
