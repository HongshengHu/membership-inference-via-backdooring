import torch
from torch import nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP, self).__init__()
        self.layer_input1 = nn.Linear(dim_in, 256)
        self.relu = nn.ReLU()
        self.layer_input2 = nn.Linear(256,128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(128, dim_out)

    def forward(self, x):
        x = self.layer_input1(x)
        x = self.relu(x)
        x = self.layer_input2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_hidden(x)
        return x

