from torchtools.models.paccrf.paccrf import PacCRF, create_YXRGB, create_position_feats
from torch.nn.functional import softmax
import torch.nn as nn


def get_crf(num_classes):
    compat = '2d'
    kernel_size = 11
    blur = 4
    dilation = 1
    crf_params = dict(num_steps=5, perturbed_init=True, fixed_weighting=False, unary_weight=0.8,
                      pairwise_kernels=[
                          dict(kernel_size=kernel_size, dilation=dilation, blur=blur, compat_type=compat,
                               spatial_filter=False,
                               pairwise_weight=2.0),
                          dict(kernel_size=kernel_size, dilation=dilation, blur=blur, compat_type=compat,
                               spatial_filter=False,
                               pairwise_weight=0.6)
                      ])
    crf = PacCRF(num_classes, final_output="log_Q", **crf_params)
    return crf


def crfarward(crf, unary, image, truth=None):
    edge_feat = [create_YXRGB(image, 80.0, 13.0),
                 create_position_feats(image.shape[2:], 3.0, bs=image.shape[0], device=image.device)]
    out = crf(unary, edge_feat, truth=truth)
    return out


class CRFModel(nn.Module):
    def __init__(self, previous_model, n_class):
        super().__init__()
        self.previous_model = previous_model
        self.crf = get_crf(n_class)

    def forward(self, x, image):
        x = self.previous_model(x)
        x = softmax(x, dim=1)
        x = crfarward(self.crf, x, image, truth=None)

        return x
