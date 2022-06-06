import torch
from torch import nn
import torch.nn.functional as F


class mIoULoss(nn.Module):
    def __init__(self, n_classes=23):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N, h, w = target.size()
        target_oneHot = torch.zeros(N, h, w, self.classes+1, device=target.device).scatter_(3, target.view(N, h, w, 1), 1)[:, :, :, :self.classes]
        target_oneHot = target_oneHot[target.view(N, h, w, 1).repeat(1, 1, 1, self.classes) < 23]
        target_oneHot = target_oneHot.view(-1, self.classes)

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)
        # put channel to the last position
        inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()
        inputs = inputs[target.view(N, h, w, 1).repeat(1, 1, 1, self.classes) < 23]
        inputs = inputs.view(-1, self.classes)

        # Numerator Product
        inter = inputs * target_oneHot

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)

        loss = inter / union

        # Return average loss over classes
        return 1 - loss.mean()/N
