
import torch
from torch.nn import Linear, Module, Dropout


class LinearDimensionalityReduction(Module):

    def __init__(self, input_size, output_size, dropout=0.) -> None:
        super(LinearDimensionalityReduction, self).__init__()
        self.linear = Linear(input_size, output_size)
        self.dropout = Dropout(dropout)

    def forward(self, pretrained_encoding):
        return self.dropout(self.linear(pretrained_encoding))
