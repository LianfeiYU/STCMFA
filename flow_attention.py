import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os
import torch


class Flow_Attention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(Flow_Attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.ln = nn.LayerNorm(16)

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def forward(self, queries, keys, values):
        initial = values
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # print("q:", queries.shape) #torch.Size([3, 4, 170, 16])
        # print("k:", keys.shape)
        # print("v:", values.shape)
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        # incoming and outgoing
        normalizer_row = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + 1e-6, keys.sum(dim=2) + 1e-6))
        normalizer_col = 1.0 / (torch.einsum("nhsd,nhd->nhs", keys + 1e-6, queries.sum(dim=2) + 1e-6))
        # reweighting
        normalizer_row_refine = (
            torch.einsum("nhld,nhd->nhl", queries + 1e-6, (keys * normalizer_col[:, :, :, None]).sum(dim=2) + 1e-6))
        normalizer_col_refine = (
            torch.einsum("nhsd,nhd->nhs", keys + 1e-6, (queries * normalizer_row[:, :, :, None]).sum(dim=2) + 1e-6))
        # competition and allocation
        normalizer_row_refine = torch.sigmoid(
            normalizer_row_refine * (float(queries.shape[2]) / float(keys.shape[2])))
        normalizer_col_refine = torch.softmax(normalizer_col_refine, dim=-1) * keys.shape[2]  # B h L vis
        # multiply
        kv = keys.transpose(-2, -1) @ (values * normalizer_col_refine[:, :, :, None])
        x = (((queries @ kv) * normalizer_row[:, :, :, None]) * normalizer_row_refine[:, :, :, None]).transpose(1, 2).contiguous()
        # print("x:", x.shape) #torch.Size([3, 170, 4, 16])
        x = initial + x  # 输入输出相加
        x = self.ln(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        # self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        # self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        # self.value_projection = nn.Linear(d_model, d_values * n_heads)

        self.query_projection = nn.GRU(input_size=d_model, hidden_size=d_keys * n_heads, num_layers=1, bidirectional=False)
        self.key_projection = nn.GRU(input_size=d_model, hidden_size=d_keys * n_heads, num_layers=1, bidirectional=False)
        self.value_projection = nn.GRU(input_size=d_model, hidden_size=d_values * n_heads, num_layers=1, bidirectional=False)

        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # queries = self.query_projection(queries).view(B, L, H, -1)
        # keys = self.key_projection(keys).view(B, S, H, -1)
        # values = self.value_projection(values).view(B, S, H, -1)

        # print("111",queries.size())
        queries, _ = self.query_projection(queries)
        # print("222", queries.size())
        queries = queries.view(B, L, H, -1)
        # print("333",queries.size())
        keys, _ = self.key_projection(keys)
        # print("444", keys.size())
        keys = keys.view(B, S, H, -1)
        values, _ = self.value_projection(values)
        # print("555", values.size())
        values = values.view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values
        )
        out = out.reshape(B, L, -1)

        return self.out_projection(out)
