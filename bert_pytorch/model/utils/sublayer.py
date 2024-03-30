import torch.nn as nn
from .layer_norm import LayerNorm


# 残差模块，返回 x + sublayer(x)
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        # 在Transformer的实现中，应该是先add再norm即：
        # self.norm(x + self.dropout(sublayer(x)))
        # 但是在哈佛大学的Transformer实现中也是先norm再add： https://nlp.seas.harvard.edu/2018/04/03/attention.html

        return x + self.dropout(sublayer(self.norm(x)))
