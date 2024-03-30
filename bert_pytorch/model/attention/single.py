import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        # 单头注意力scores: batch * seq_len * seq_len
        # 多头注意力scores: batch * head * seq_len * seq_len
        # mask: torch.ByteTensor([batch, 1, seq_len, seq_len)
        if mask is not None:
            # mask为0的地方用-1e9填充
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)
        # return: batch * head * seq_len * d_model/head
        return torch.matmul(p_attn, value), p_attn
