
import torch
from torch.nn import Linear, Module, Dropout, Flatten
import torch.nn.functional as F


class LinearDimensionalityReduction(Module):
    """Encoder for the baseline"""

    def __init__(self, input_size, output_size, dropout=0.) -> None:
        super(LinearDimensionalityReduction, self).__init__()
        self.linear = Linear(input_size, output_size)
        self.dropout = Dropout(dropout)

    def forward(self, pretrained_encoding):
        return self.dropout(self.linear(pretrained_encoding))

class EncodeForAttention(Module):
    """Encoder for the attention"""

    def __init__(self, in_size, out_size) -> None:
        super(EncodeForAttention, self).__init__()
        self.lin = Linear(in_size, out_size)

    def forward(self, X):
        # X is [batch_size, channels, h, w]:
        X = torch.flatten(X, start_dim=2)  # [batch_size, channels, h*w]
        X = torch.permute(X, (0, 2, 1))  # [batch_size, h*w, channels]
        return self.lin(X)