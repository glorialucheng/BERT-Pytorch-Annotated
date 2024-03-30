import torch.nn as nn
import torch
import math


# 采用固定的position embedding，也就是不可学习的
class PositionalEmbedding(nn.Module):
    # d_model表示输入的token的维度，max_len表示最大的sequence length
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # 偶数位置
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位置
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
