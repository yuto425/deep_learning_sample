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


class MultiHeadedAttention(nn.Module):
    def __init__(self,
                 depth: int,
                 head_num: int,
                 dropout_rate: float = 0.1) -> None:
        super().__init__()
        self._query_linear = nn.Linear(depth, depth)
        self._key_linear = nn.Linear(depth, depth)
        self._value_linear = nn.Linear(depth, depth)
        self._out = nn.Linear(depth, depth)
        self._depth = depth
        self._head_num = head_num
        assert self._depth % self._head_num == 0
        self._head_size = int(self._depth / self._head_num)
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self,
                input: torch.Tensor,
                memory: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        mixed_query = self._query_linear(input)
        mixed_key = self._key_linear(memory)
        mixed_value = self._value_linear(memory)
        query = self._split_head(mixed_query)
        key = self._split_head(mixed_key)
        value = self._split_head(mixed_value)
        weights = torch.matmul(
            query, key.transpose(-1, -2)) / math.sqrt(self._head_size)
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e10)
        normalized_weights = F.softmax(weights, dim=-1)
        normalized_weights = self._dropout(normalized_weights)
        attention_output = torch.matmul(normalized_weights, value)
        return self._combine_head(attention_output)

    def _split_head(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self._head_num, self._head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def _combine_head(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (self._depth,)
        return x.view(*new_shape)
