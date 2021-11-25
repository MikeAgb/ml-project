
import torch
from torch.nn import Linear, Module, Dropout, Flatten
import torch.nn.functional as F


class LinearDimensionalityReduction(Module):

    def __init__(self, input_size, output_size, dropout=0.) -> None:
        super(LinearDimensionalityReduction, self).__init__()
        self.linear = Linear(input_size, output_size)
        self.dropout = Dropout(dropout)

    def forward(self, pretrained_encoding):
        return self.dropout(self.linear(pretrained_encoding))

class EncodeFromCNNLayer(Module):
    def __init__(self, output_size=256) -> None:
        super(EncodeFromCNNLayer, self).__init__()
        self.flatten = Flatten()
        self.lin_1 = Linear(7 * 7 * 512, 2048)
        self.lin_2 = Linear(2048, 1024)
        self.lin_3 = Linear(1024, output_size)

    def forward(self, X):
        X = self.flatten(X)
        X = F.relu(self.lin_1(X))
        X = F.relu(self.lin_2(X))
        return F.relu(self.lin_3(X))
