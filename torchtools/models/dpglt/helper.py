#!/usr/bin/env python
# coding: utf-8
# import cv2
import numpy as np
import torch
from torchvision import transforms
import segmentation_models_pytorch as smp
from torchtools.models.fpn.fpn_smp import FPN
from torchtools.models.swin.swin_transformer import SwinTransformer
from torchtools.models.crfhead import CRFModel

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
    target[target == 255] = -1
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


def create_model_load_weights(net, n_class, img_size=(512, 512), mode=1, crf=None, train_ops=False, train_ops_mat_only=False):
    print("Loading net archtecture: {}".format(net))
    if mode in [99]:
        from torchtools.models.dpglt.dpglt import DPGLTransformer
        smp.encoders.encoders["dpglt"] = {
            "encoder": DPGLTransformer,  # encoder class here
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
                "img_size": img_size
            },
        }

    elif mode in [98]:
        from torchtools.models.dpglt.dpglt_less_trans import DPGLTransformer
        print("Use DPGLT with less transformer block")
        smp.encoders.encoders["dpglt"] = {
            "encoder": DPGLTransformer,  # encoder class here
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
                "img_size": img_size
            },
        }
    elif mode in [97]:
        from torchtools.models.dpglt.dpglt_overlapcnn import DPGLTransformer
        print("Use DPGLT with overlapping cnn kernal")
        smp.encoders.encoders["dpglt"] = {
            "encoder": DPGLTransformer,  # encoder class here
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
                "img_size": img_size
            },
        }

    elif mode in [96]:
        from torchtools.models.dpglt.dpglt_nodp import DPGLTransformer
        print("Use DPGLT with no dynamic patch")
        smp.encoders.encoders["dpglt"] = {
            "encoder": DPGLTransformer,  # encoder class here
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
                "img_size": img_size
            },
        }

    elif mode in [95]:
        from torchtools.models.dpglt.dpglt_single_branch import DPGLTransformer
        print("Use DPGLT with single branch only")
        smp.encoders.encoders["dpglt"] = {
            "encoder": DPGLTransformer,  # encoder class here
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
                "img_size": img_size
            },
        }

    elif mode in [94]:
        from torchtools.models.dpglt.dpglt_conv_enlarge import DPGLTransformer
        print("Use DPGLT with conv enlarge module")
        smp.encoders.encoders["dpglt"] = {
            "encoder": DPGLTransformer,  # encoder class here
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
                "img_size": img_size
            },
        }

    elif mode in [93]:
        from torchtools.models.dpglt.dpglt_single_branch_reduce_tf import DPGLTransformer
        print("Use DPGLT with single branch, reduced tf")
        smp.encoders.encoders["dpglt"] = {
            "encoder": DPGLTransformer,  # encoder class here
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
                "img_size": img_size
            },
        }

    elif mode in [92]:
        from torchtools.models.dpglt.dpglt_single_branch_avgpool import DPGLTransformer
        print("Use DPGLT with single branch, reduced tf with average pooling")
        smp.encoders.encoders["dpglt"] = {
            "encoder": DPGLTransformer,  # encoder class here
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
                "img_size": img_size
            },
        }

    elif mode in [91]:
        from torchtools.models.dpglt.dpglt_single_branch_dilatedcnn import DPGLTransformer
        print("Use DPGLT with single branch, and dilated cnn")
        smp.encoders.encoders["dpglt"] = {
            "encoder": DPGLTransformer,  # encoder class here
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
                "img_size": img_size
            },
        }

    elif mode in [90]:
        from torchtools.models.dpglt.dpglt_single_branch import DPGLTransformer
        print("Use DPGLT with single branch, and small window")
        smp.encoders.encoders["dpglt"] = {
            "encoder": DPGLTransformer,  # encoder class here
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
                "img_size": img_size
            },
        }

    elif mode in [89]:
        from torchtools.models.dpglt.dpglt_single_branch_nomerge import DPGLTransformer
        print("Use DPGLT with single branch, and no merge between global and aggregated local")
        smp.encoders.encoders["dpglt"] = {
            "encoder": DPGLTransformer,  # encoder class here
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
                "img_size": img_size
            },
        }

    elif mode in [88]:
        from torchtools.models.dpglt.dpglt_single_branch_nomerge_butswattn import DPGLTransformer
        print("Use DPGLT with single branch, and no merge between global and aggregated local, but attention")
        smp.encoders.encoders["dpglt"] = {
            "encoder": DPGLTransformer,  # encoder class here
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
                "img_size": img_size
            },
        }

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
            "window_size": [7, 7, 7, 7] if mode != 90 else [2, 2, 2, 2],
            "depths": [2, 2, 6, 2],
            "out_channels": [3, 96, 192, 384, 768],
            "out_indices": (0, 1, 2, 3),
            "num_heads": [3, 6, 12, 24]
        },
    }

    if net == "swins":
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
                "depths": [2, 2, 18, 2],
                "out_channels": [3, 96, 192, 384, 768],
                "out_indices": (0, 1, 2, 3),
                "num_heads": [3, 6, 12, 24]
            },
        }

    if net == "efficientnet":
        model = FPN(
            encoder_name="efficientnet-b7",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class if not train_ops else None,  # model output channels (number of classes in your dataset)
            train_ops=train_ops,
            mode=mode,
            train_ops_mat_only=train_ops_mat_only
        )

    elif net == "efficientnet-b5":
        model = FPN(
            encoder_name="efficientnet-b5",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class if not train_ops else None,  # model output channels (number of classes in your dataset)
            train_ops=train_ops,
            mode=mode,
            train_ops_mat_only=train_ops_mat_only
        )

    elif net == "resnest101":
        model = FPN(
            encoder_name="timm-resnest101e",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class if not train_ops else None,  # model output channels (number of classes in your dataset)
            train_ops=train_ops,
            mode=mode,
            train_ops_mat_only=train_ops_mat_only
        )

    elif net == "resnest269":
        model = FPN(
            encoder_name="timm-resnest269e",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class if not train_ops else None,  # model output channels (number of classes in your dataset)
            train_ops=train_ops,
            mode=mode,
            train_ops_mat_only=train_ops_mat_only
        )

    elif net == "hrnet":
        model = FPN(
            encoder_name="tu-hrnet_w64",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class if not train_ops else None,  # model output channels (number of classes in your dataset)
            mode=mode,
            train_ops = train_ops,
            train_ops_mat_only=train_ops_mat_only
        )
    elif net == "resnet50":
        model = FPN(
            encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class if not train_ops else None,  # model output channels (number of classes in your dataset)
            mode=mode,
            train_ops=train_ops,
            train_ops_mat_only=train_ops_mat_only
        )

    elif net == "resnet152":
        model = FPN(
            encoder_name="resnet152",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class if not train_ops else None,  # model output channels (number of classes in your dataset)
            mode=mode,
            train_ops=train_ops,
            train_ops_mat_only=train_ops_mat_only
        )

    elif net == "swin_7local":

        model = FPN(
            encoder_name="swin_7local",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class if not train_ops else None,  # model output channels (number of classes in your dataset)
            encoder_depth=7,
            mode=mode,
            train_ops=train_ops,
            train_ops_mat_only=train_ops_mat_only
        )

    elif net == "swin":

        model = FPN(
            encoder_name="swin",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class if not train_ops else None,  # model output channels (number of classes in your dataset)
            encoder_depth=5,
            mode=mode,
            train_ops=train_ops,
            train_ops_mat_only=train_ops_mat_only
        )

    elif net == "swins":

        model = FPN(
            encoder_name="swin",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_class if not train_ops else None,  # model output channels (number of classes in your dataset)
            encoder_depth=5,
            mode=mode,
            train_ops=train_ops,
            train_ops_mat_only=train_ops_mat_only
        )

    elif net == "dpglt":

        model = FPN(encoder_name="dpglt", encoder_depth=5, classes=n_class if not train_ops else None, encoder_weights=None, train_ops=train_ops, mode=mode,
                    train_ops_mat_only=train_ops_mat_only)

    else:
        raise Exception("not valid net")

    if crf is not None:
        model = CRFModel(model, n_class)

    return model


def get_optimizer(model, mode=1, learning_rate=2e-5):
    if True:
        # train net
        from mmcv.runner import build_optimizer
        # AdamW optimizer, no weight decay for position embedding & layer norm in backbone
        cfg_optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                         paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                         'relative_position_bias_table': dict(decay_mult=0.),
                                                         'norm': dict(decay_mult=0.)}))
        optimizer = build_optimizer(model, cfg_optimizer)

    return optimizer
