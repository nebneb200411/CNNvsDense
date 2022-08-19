import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNetwork(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.dense1 = nn.Linear(in_ch, 200)
        self.dense2 = nn.Linear(200, 10)
    
    def forward(self, x):
        x = self.dense1(x.view(x.shape[0], -1))
        x = F.relu(x)
        x = self.dense2(x)
        return x
        