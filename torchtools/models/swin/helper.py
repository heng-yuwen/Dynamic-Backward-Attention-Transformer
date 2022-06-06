#!/usr/bin/env python
# coding: utf-8
# import cv2
import numpy as np
import torch
from torchvision import transforms
from mmcv.runner import build_optimizer
import segmentation_models_pytorch as smp
from torchtools.models.fpn.fpn_smp import FPN
from torchtools.models.swin.swin_transformer import SwinTransformer

# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
mean = torch.Tensor([124 / 255., 117 / 255., 104 / 255.])
std = torch.Tensor([1, 1, 1])
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


def _mask_transform(mask, training=True):
    target = np.array(mask).astype('int32')
    assert target[:,:,0].max() <= 27, "wrong index, max > 27"
    assert target.min() >=0, "wrong index, min < 0"
    target -= 1
    if training:
        return torch.LongTensor(target)
    else:
        return torch.LongTensor(target).unsqueeze(dim=0)


def masks_transform(masks, training=True):
    '''
    masks: list of PIL images
    '''
    targets = []
    for m in masks:
        targets.append(_mask_transform(m, training=training))
    if training:
        targets = torch.stack(targets, dim=0)
    return targets


def images_transform(images, training=True):
    '''
    images: list of PIL images
    '''
    inputs = []
    for img in images:
        inputs.append(transformer(img))
    if training:
        return torch.stack(inputs, dim=0)
    else:
        return inputs


def create_model_load_weights(net, n_class, device, img_size=(512, 512), mode=1):
    if net == "efficientnet":
        model = FPN(
            encoder_name="efficientnet-b7",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class,  # model output channels (number of classes in your dataset)
            mode=mode
        )

    elif net == "resnest":
        model = FPN(
            encoder_name="timm-resnest269e",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class,  # model output channels (number of classes in your dataset)
            mode=mode
        )

    elif net == "hrnet":
        model = FPN(
            encoder_name="tu-hrnet_w64",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class,  # model output channels (number of classes in your dataset)
            mode=mode
        )
    elif net == "resnet50":
        model = FPN(
            encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class,  # model output channels (number of classes in your dataset)
            mode=mode
        )

    elif net == "swin_7local":
        smp.encoders.encoders["swin_7local"] = {
            "encoder": SwinTransformer,  # encoder class here
            # "pretrained_settings": {
            #     "imagenet": {
            #         "mean": [0.485, 0.456, 0.406],
            #         "std": [0.229, 0.224, 0.225],
            #         "url": "https://some-url.com/my-model-weights",
            #         "input_space": "RGB",
            #         "input_range": [0, 1],
            #     },
            # },
            "params": {
                          # init params for encoder if any
                          "window_size": [2, 2, 7, 7, 7, 7],
                          "depths": [2, 2, 6, 2, 2, 2],
                          "out_channels": [3, 96, 192, 384, 768, 768, 768],
                          "out_indices": (0, 1, 2, 3, 4, 5),
                          "num_heads": [3, 6, 12, 24, 24, 24]
                      },
        }
        model = FPN(
            encoder_name="swin_7local",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class,  # model output channels (number of classes in your dataset)
            encoder_depth=7,
            mode = mode
        )

    elif net == "swin":
        smp.encoders.encoders["swin"] = {
            "encoder": SwinTransformer,  # encoder class here
            # "pretrained_settings": {
            #     "imagenet": {
            #         "mean": [0.485, 0.456, 0.406],
            #         "std": [0.229, 0.224, 0.225],
            #         "url": "https://some-url.com/my-model-weights",
            #         "input_space": "RGB",
            #         "input_range": [0, 1],
            #     },
            # },
            "params": {
                # init params for encoder if any
                "window_size": [7, 7, 7, 7],
                "depths": [2, 2, 6, 2],
                "out_channels": [3, 96, 192, 384, 768],
                "out_indices": (0, 1, 2, 3),
                "num_heads": [3, 6, 12, 24]
            },
        }
        model = FPN(
            encoder_name="swin",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class,  # model output channels (number of classes in your dataset)
            encoder_depth=5,
            mode=mode
        )

    else:
        raise Exception("not valid net")

    return model


def get_optimizer(model, mode=1, learning_rate=2e-5):
    if mode < 5:
        # train net
        # AdamW optimizer, no weight decay for position embedding & layer norm in backbone
        cfg_optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                         paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                         'relative_position_bias_table': dict(decay_mult=0.),
                                                         'norm': dict(decay_mult=0.)}))
        optimizer = build_optimizer(model, cfg_optimizer)

    return optimizer
