from torch import nn


class Identity(nn.Module):
    def forward(self, inputs):
        return inputs