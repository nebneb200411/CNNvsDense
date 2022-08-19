import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_ch=1, f=32):
        super().__init__()
        # input_image (B, 1, 28, 28)
        models = []
        
        models += [
            nn.Conv2d(in_ch, f, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(f),
            nn.ReLU(True),
        ]

        # out: (B, 32, 16, 16)

        blocks_num = 3
        for i in range(blocks_num):
            mult = int(2**i)
            models += [
                nn.Conv2d(f * mult, f * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(f * mult * 2),
                nn.ReLU(True),
            ]
        
        #Out: (B, 256, 2, 2)

        # In(B, 2 * 2* 256)
        self.dense = nn.Linear(f * mult * 2 * 2 * 2, 10)
        # Out(B, 10)

        self.models = nn.Sequential(*models)

    def forward(self, x):
        # input_image (B, 1, 28, 28)
        x = self.models(x)
        # Out (B, 256, 2, 2)
        x = self.dense(x.view(x.shape[0], -1))
        # Out (B, 10)
        return x
