from ..resnet.resnet import resnet50
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
import os
# import matplotlib.pyplot as plt
from torchtools.models.paccrf.paccrf import PacCRF, create_YXRGB, PacCRFLoose, create_position_feats
from torchvision.transforms.functional import resize


def segment2rgb(output, mask=False):
    color_plate = {0: [119, 17, 17], 1: [202, 198, 144], 2: [186, 200, 238], 3: [124, 143, 166], 4: [89, 125, 49],
                   5: [16, 68, 16], 6: [187, 129, 156], 7: [208, 206, 72], 8: [98, 39, 69], 9: [102, 102, 102],
                   10: [76, 74, 95], 11: [16, 16, 68], 12: [68, 65, 38], 13: [117, 214, 70], 14: [221, 67, 72],
                   15: [92, 133, 119]}
    if not mask:
        output = output.argmax(dim=1)
    output = output.squeeze().cpu()
    rgbmask = np.zeros([output.size()[0], output.size()[1], 3], dtype=np.uint8)
    for i in range(16):
        rgbmask[output == i] = color_plate[i]

    return rgbmask


def get_patch_info(patch_size, H, W):
    num_H = H // patch_size if H % patch_size == 0 else H // patch_size + 1
    num_W = W // patch_size if W % patch_size == 0 else W // patch_size + 1

    stride_H = H // num_H if H % num_H == 0 else H // num_H + 1
    stride_W = W // num_W if W % num_W == 0 else W // num_W + 1

    H_padded = stride_H * (num_H - 1) + patch_size
    W_padded = stride_W * (num_W - 1) + patch_size

    pad_H = H_padded - H
    pad_W = W_padded - W

    return pad_H, pad_W, stride_H, stride_W, H_padded, W_padded


def global2patch(images, segments, images_255, patch_size, mode, pad=True):
    # if mode == 1:  # multiple input images, return patch only to train local branch
    image_patches_batch = []
    segment_patches_batch = []
    count_mask_batch = []
    pad_image_batch = []
    pad_mask_batch = []
    images_255_batch = []

    for i in range(len(images)):
        image = images[i]
        _, C, H, W = image.size()
        pad_H, pad_W, stride_H, stride_W, H_padded, W_padded = get_patch_info(patch_size, H, W)
        if pad:
            image = F.pad(image, (pad_W // 2 + pad_W % 2, pad_W // 2, pad_H // 2 + pad_H % 2, pad_H // 2))
        else:
            image = F.interpolate(image, (H_padded, W_padded), mode='bilinear', align_corners=True)
        count_mask = torch.ones([1, H_padded, W_padded], device=image.device)

        image_patches = image.unfold(2, patch_size, stride_H).unfold(3, patch_size, stride_W).squeeze()
        count_mask = count_mask.unfold(1, patch_size, stride_H).unfold(2, patch_size, stride_W).squeeze()
        image_patches = image_patches.contiguous().view(C, -1, patch_size, patch_size)
        image_patches = image_patches.permute(1, 0, 2, 3)
        count_mask = count_mask.contiguous().view(-1, patch_size, patch_size)

        image_patches_batch.append(image_patches)
        count_mask_batch.append(count_mask)

        if segments is not None:
            segment = segments[i]
            image_255 = images_255[i]
            if H == segment.size()[-2] and W == segment.size()[-1]:
                segment = F.pad(segment, (pad_W // 2 + pad_W % 2, pad_W // 2, pad_H // 2 + pad_H % 2, pad_H // 2), value=-1)
            elif H_padded == segment.size()[-2] and W_padded == segment.size()[-1]:
                segment = segment
            else:
                exit(1)
            image_255 = F.pad(image_255, (pad_W // 2 + pad_W % 2, pad_W // 2, pad_H // 2 + pad_H % 2, pad_H // 2))

            pad_image_batch.append(image)
            images_255_batch.append(image_255)
            pad_mask_batch.append(segment)

            segment_patches = segment.unfold(1, patch_size, stride_H).unfold(2, patch_size, stride_W).squeeze()
            segment_patches = segment_patches.contiguous().view(-1, patch_size, patch_size)
            segment_patches_batch.append(segment_patches)

    image_patches_batch = torch.cat(image_patches_batch)
    count_mask_batch = torch.cat(count_mask_batch)
    if segments is not None:
        segment_patches_batch = torch.cat(segment_patches_batch)

    if mode == 1:  # remove invalid patches
        valid_index = []
        for i in range(len(segment_patches_batch)):
            if not (segment_patches_batch[i] == -1).all():
                valid_index.append(i)
        image_patches_batch = image_patches_batch[valid_index]
        segment_patches_batch = segment_patches_batch[valid_index]
    if segments is not None:
        return pad_image_batch, image_patches_batch, segment_patches_batch, count_mask_batch, images_255_batch, \
               pad_mask_batch, [pad_H, pad_W, stride_H, stride_W, H_padded, W_padded, patch_size]
    else:
        return image_patches_batch, count_mask_batch, [pad_H, pad_W, stride_H, stride_W, H_padded, W_padded, patch_size]


def patch2global(tensor_patches, count_mask, unfold_info):
    # restore global to original size for batch_size = 1 only
    pad_H, pad_W, stride_H, stride_W, H_padded, W_padded, patch_size = unfold_info
    C = tensor_patches.size()[1]

    tensor_patches = tensor_patches.permute(1, 2, 3, 0)
    count_mask = count_mask.permute(1, 2, 0)

    tensor_patches = tensor_patches.contiguous().view(C * patch_size ** 2, -1)
    count_mask = count_mask.contiguous().view(patch_size ** 2, -1)

    tensor_patches = F.fold(tensor_patches.unsqueeze(dim=0), output_size=(H_padded, W_padded),
                            kernel_size=patch_size, stride=(stride_H, stride_W))
    count_mask = F.fold(count_mask.unsqueeze(dim=0), output_size=(H_padded, W_padded), kernel_size=patch_size,
                        stride=(stride_H, stride_W))

    tensor_patches = tensor_patches / count_mask

    return tensor_patches


def get_paccrf(num_classes=16, pairwise="p4d5641p4d5161", loose=False, use_yx=False, shared_scales=False, adaptive_init=True, conv=False):
    if not conv:
        pw_strs = []
        for s in pairwise.split('p')[1:]:
            l_ = 3 if s[2] == 's' else 2
            pw_strs.append('_'.join((s[:l_], s[l_], s[(l_ + 1):-1], s[-1])))

        crf_params = dict(num_steps=5,
                          perturbed_init=True,
                          fixed_weighting=False,
                          unary_weight=1.0,
                          final_output="softmax",
                          pairwise_kernels=[])

        for pw_str in pw_strs:
            t_, k_, d_, b_ = pw_str.split('_')
            pairwise_param = dict(kernel_size=int(k_),
                                  dilation=int(d_),
                                  blur=int(b_),
                                  compat_type=('potts' if t_.startswith('0d') else t_[:2]),
                                  spatial_filter=t_.endswith('s'),
                                  pairwise_weight=1.0)
            crf_params['pairwise_kernels'].append(pairwise_param)

        CRF = PacCRFLoose if loose else PacCRF
        crf = CRF(num_classes, **crf_params)
        feat_scales = nn.ParameterList()

        for s in pw_strs:
            fs, dilation = float(s.split('_')[1]), float(s.split('_')[2])
            p_sc = (((fs - 1) * dilation + 1) / 4.0) if adaptive_init else 100.0
            c_sc = 30.0
            if use_yx:
                scales = torch.tensor([p_sc, c_sc] if shared_scales else ([p_sc] * 2 + [c_sc] * 3), dtype=torch.float32)
            else:
                scales = torch.tensor(c_sc if shared_scales else [c_sc] * 3, dtype=torch.float32)
            feat_scales.append(nn.Parameter(scales))

    else:
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
        crf = PacCRF(num_classes, final_output="softmax", **crf_params)
        feat_scales = None

    return crf, feat_scales


def crfarward(crf, feat_scales, unary, image, truth=None, use_yx=False, shared_scales=False, conv=False):
    if not conv:
        if feat_scales[0].device != unary.device:
            for i in range(len(feat_scales)):
                feat_scales[i] = feat_scales[i].to(unary.device)

        if use_yx:
            if shared_scales:
                edge_feat = [create_YXRGB(image, yx_scale=sc[0], rgb_scale=sc[1]) for sc in feat_scales]
            else:
                edge_feat = [create_YXRGB(image, scales=sc) for sc in feat_scales]
        else:
            edge_feat = [image * (1.0 / rgb_scale.view(-1, 1, 1)) for rgb_scale in feat_scales]
        out = crf(unary, edge_feat, truth=truth)
    else:
        edge_feat = [create_YXRGB(image, 80.0, 13.0),
                     create_position_feats(image.shape[2:], 3.0, bs=image.shape[0], device=image.device)]
        out = crf(unary, edge_feat, truth=truth)
    return out


class fpn_module_single(nn.Module):
    def __init__(self, numClass):
        super(fpn_module_single, self).__init__()
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        self.smooth1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        # Classify layers
        self.classify = nn.Conv2d(128 * 4, numClass, kernel_size=3, stride=1, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs)
        p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs)
        p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs)
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), **self._up_kwargs) + y

    def forward(self, c2, c3, c4, c5):
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        ps0 = [p5, p4, p3, p2]

        # Smooth
        p5 = self.smooth1_1(p5)
        p4 = self.smooth2_1(p4)
        p3 = self.smooth3_1(p3)
        p2 = self.smooth4_1(p2)

        ps1 = [p5, p4, p3, p2]

        p5 = self.smooth1_2(p5)
        p4 = self.smooth2_2(p4)
        p3 = self.smooth3_2(p3)
        p2 = self.smooth4_2(p2)

        ps2 = [p5, p4, p3, p2]

        # Classify
        ps3 = self._concatenate(p5, p4, p3, p2)
        output = self.classify(ps3)
        # _,_,H,W = input.size()
        # output = self.smooth_classify1(F.interpolate(output, size=(H, W), **self._up_kwargs))
        # output = self.smooth_classify2(output)

        return output, ps0, ps1, ps2, ps3


class fpn_module_double(nn.Module):
    def __init__(self, numClass):
        super(fpn_module_double, self).__init__()
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # Top layer
        fold = 2
        self.toplayer = nn.Conv2d(2048 * fold, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Lateral layers [C]
        self.latlayer1 = nn.Conv2d(1024 * fold, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512 * fold, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256 * fold, 256, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        # ps0
        self.smooth1_1 = nn.Conv2d(256 * fold, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(256 * fold, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(256 * fold, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(256 * fold, 256, kernel_size=3, stride=1, padding=1)
        # ps1
        self.smooth1_2 = nn.Conv2d(256 * fold, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(256 * fold, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(256 * fold, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(256 * fold, 128, kernel_size=3, stride=1, padding=1)
        # ps2 is concatenation
        # Classify layers
        self.smooth = nn.Conv2d(128 * 4 * fold, 128 * 4, kernel_size=3, stride=1, padding=1)
        self.classify = nn.Conv2d(128 * 4, numClass, kernel_size=3, stride=1, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs)
        p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs)
        p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs)
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), **self._up_kwargs) + y

    def first_forward(self, patch_l, patch_g):
        c5, c4, c3, c2 = patch_l
        p5 = self.toplayer(torch.cat([c5] + [F.interpolate(patch_g[0], size=c5.size()[2:], **self._up_kwargs)], dim=1))
        p4 = self._upsample_add(p5, self.latlayer1(
            torch.cat([c4] + [F.interpolate(patch_g[1], size=c4.size()[2:], **self._up_kwargs)], dim=1)))
        p3 = self._upsample_add(p4, self.latlayer2(
            torch.cat([c3] + [F.interpolate(patch_g[2], size=c3.size()[2:], **self._up_kwargs)], dim=1)))
        p2 = self._upsample_add(p3, self.latlayer3(
            torch.cat([c2] + [F.interpolate(patch_g[3], size=c2.size()[2:], **self._up_kwargs)], dim=1)))
        ps0 = [p5, p4, p3, p2]

        return ps0

    def second_forward(self, ps0, ps0_ext):
        # Smooth
        p5, p4, p3, p2 = ps0
        p5 = self.smooth1_1(
            torch.cat([p5] + [F.interpolate(ps0_ext[0], size=p5.size()[2:], **self._up_kwargs)], dim=1))
        p4 = self.smooth2_1(
            torch.cat([p4] + [F.interpolate(ps0_ext[1], size=p4.size()[2:], **self._up_kwargs)], dim=1))
        p3 = self.smooth3_1(
            torch.cat([p3] + [F.interpolate(ps0_ext[2], size=p3.size()[2:], **self._up_kwargs)], dim=1))
        p2 = self.smooth4_1(
            torch.cat([p2] + [F.interpolate(ps0_ext[3], size=p2.size()[2:], **self._up_kwargs)], dim=1))
        ps1 = [p5, p4, p3, p2]

        return ps1

    def third_forward(self, ps1, ps1_ext):
        # Smooth
        p5, p4, p3, p2 = ps1
        p5 = self.smooth1_2(
            torch.cat([p5] + [F.interpolate(ps1_ext[0], size=p5.size()[2:], **self._up_kwargs)], dim=1))
        p4 = self.smooth2_2(
            torch.cat([p4] + [F.interpolate(ps1_ext[1], size=p4.size()[2:], **self._up_kwargs)], dim=1))
        p3 = self.smooth3_2(
            torch.cat([p3] + [F.interpolate(ps1_ext[2], size=p3.size()[2:], **self._up_kwargs)], dim=1))
        p2 = self.smooth4_2(
            torch.cat([p2] + [F.interpolate(ps1_ext[3], size=p2.size()[2:], **self._up_kwargs)], dim=1))
        ps2 = [p5, p4, p3, p2]

        return ps2

    def final_forward(self, ps2, ps2_ext):
        # Classify
        # use ps2_ext
        p5, p4, p3, p2 = ps2
        ps3 = self._concatenate(
            torch.cat([p5] + [F.interpolate(ps2_ext[0], size=p5.size()[2:], **self._up_kwargs)], dim=1),
            torch.cat([p4] + [F.interpolate(ps2_ext[1], size=p4.size()[2:], **self._up_kwargs)], dim=1),
            torch.cat([p3] + [F.interpolate(ps2_ext[2], size=p3.size()[2:], **self._up_kwargs)], dim=1),
            torch.cat([p2] + [F.interpolate(ps2_ext[3], size=p2.size()[2:], **self._up_kwargs)], dim=1)
        )
        ps3 = self.smooth(ps3)
        output = self.classify(ps3)

        return output, ps3
        # return ps3


class GLNet(nn.Module):
    def __init__(self, numClass, mode):
        super(GLNet, self).__init__()
        self.resnet_local = resnet50(True, glnet=True)
        self.resnet_global = resnet50(True, glnet=True)

        self.fpn_local = fpn_module_double(numClass)
        self.fpn_global = fpn_module_double(numClass)

        self.ensemble_conv = nn.Conv2d(128 * 4 * 2, numClass, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.ensemble_conv.weight, mean=0, std=0.01)

        self.mode = mode
        self.mse = nn.MSELoss()

        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}

        if mode in [7, 12, 15]:
            self.crf, self.feat_scales = get_paccrf(numClass)
            self.conv = False
            # -------------PacCRF--------------#
        if mode in [8, 13, 16]:
            self.crf, self.feat_scales = get_paccrf(numClass, loose=True)
            self.conv = False
        if mode in [9, 10, 18, 19]:
            self.crf, self.feat_scales = get_paccrf(numClass, conv=True)
            self.conv = True

        # init fpn
        for m in self.fpn_global.children():
            if hasattr(m, 'weight'): nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'): nn.init.constant_(m.bias, 0)
        for m in self.fpn_local.children():
            if hasattr(m, 'weight'): nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'): nn.init.constant_(m.bias, 0)

    def prepare_feature_to_share(self, ls, gs, mode):
        # prepare the feature map between local and global branch.
        l_list = []
        g_list = []
        for l, g in zip(ls, gs):
            patch_g, patch_count_g, patch_config = global2patch([g], None, None, patch_size=l.size()[-1], mode=mode,
                                                                pad=False)
            full_l = patch2global(l, patch_count_g, patch_config)
            l_list.append(patch_g)
            g_list.append(full_l)
        return l_list, g_list

    def ensemble(self, f_global, f_local):
        return self.ensemble_conv(torch.cat((f_local, f_global), dim=1))

    def forward(self, images_global, segments_global, patch_size, images_255, names, mode=6):
        pad_image_batch, image_patches_batch, segment_patches_batch, count_mask_batch, images_255_batch, pad_mask_batch, unfold_info = global2patch(
            images_global,
            segments_global, images_255,
            patch_size,
            mode=mode)

        # first forward, run local and global
        c2_l, c3_l, c4_l, c5_l = self.resnet_local.forward(image_patches_batch)
        c2_g, c3_g, c4_g, c5_g = self.resnet_global.forward(pad_image_batch[0])

        l_list, g_list = self.prepare_feature_to_share([c5_l, c4_l, c3_l, c2_l], [c5_g, c4_g, c3_g, c2_g], mode=mode)

        ps0_l = self.fpn_local.first_forward([c5_l, c4_l, c3_l, c2_l], l_list)
        ps0_g = self.fpn_global.first_forward([c5_g, c4_g, c3_g, c2_g], g_list)

        # second forward, share info between local and global
        l_list_2, g_list_2 = self.prepare_feature_to_share(ps0_l, ps0_g, mode=mode)
        ps1_l = self.fpn_local.second_forward(ps0_l, l_list_2)
        ps1_g = self.fpn_global.second_forward(ps0_g, g_list_2)

        # third forward, the same as before
        l_list_3, g_list_3 = self.prepare_feature_to_share(ps1_l, ps1_g, mode=mode)
        ps2_l = self.fpn_local.third_forward(ps1_l, l_list_3)
        ps2_g = self.fpn_global.third_forward(ps1_g, g_list_3)

        # fourth forward, the same as before
        l_list_4, g_list_4 = self.prepare_feature_to_share(ps2_l, ps2_g, mode=mode)
        output_l, ps3_l = self.fpn_local.final_forward(ps2_l, l_list_4)
        output_g, ps3_g = self.fpn_global.final_forward(ps2_g, g_list_4)

        # ensemble as patches
        ps3_g2l, _, _ = global2patch([ps3_g], None, None, ps3_l.size()[-1], mode=mode, pad=False)
        output_ensemble = self.ensemble(ps3_g2l, ps3_l)
        output_ensemble2global = patch2global(F.interpolate(output_ensemble, patch_size, mode="nearest"), count_mask_batch, unfold_info)
        output_ensemble2global = F.softmax(output_ensemble2global, dim=1)
        # output_ensemble2global is used to produce final prediction.

        generate_pseudo = False
        # save no_crf

        # img = Image.fromarray(np.asarray(segment2rgb(output_ensemble2global)))
        # img.save("output/test/glnet_no_crf/{}.png".format(names))
        if mode in [7, 8, 9, 10, 12, 13, 15, 16, 18, 19]:
            crfrgb = images_255_batch[0] - torch.tensor([122.675, 116.669, 104.008],
                                                                     device=images_255[0].device).view(
                1, -1, 1, 1)
            if generate_pseudo:
                output_ensemble2global = crfarward(self.crf, self.feat_scales, output_ensemble2global, crfrgb, truth=pad_mask_batch[0], conv=self.conv)
            else:
                output_ensemble2global = crfarward(self.crf, self.feat_scales, output_ensemble2global, crfrgb,
                                                   truth=None, conv=self.conv)
        mse = self.mse(ps3_l, ps3_g2l)

        if True:
            output_index = output_ensemble2global.argmax(dim=1).squeeze()
            img = Image.fromarray(np.asarray(output_index.cpu()).astype(np.uint8), "L")
            img.save("output/predict/img3/{}.png".format(names[0]))
            img = Image.fromarray(np.asarray(segment2rgb(output_ensemble2global)))
            img.save("output/predict/img3/color_{}.png".format(names[0]))
            # img = Image.fromarray(np.asarray(images_255_batch[0].squeeze().permute(1, 2, 0).cpu()))
            # img.save("output/train/second_image/{}.png".format(names[0]))

        # if len(os.listdir("output/test/2nd_self_train")) > 400:
        #     exit(0)
        # img = Image.fromarray(np.asarray(segment2rgb(output_ensemble2global)))
        # img.save("output/test/student_s3/{}.png".format(names[0]))
        # img = Image.fromarray(np.asarray(images_255_batch[0].squeeze().permute(1, 2, 0).cpu()))
        # img.save("output/test/original/{}.jpg".format(names[0]))

        return output_ensemble2global, output_l, output_g, segment_patches_batch, pad_mask_batch[0], mse


class fpn(nn.Module):
    def __init__(self, numClass, mode=1):
        super(fpn, self).__init__()
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        self.mode = mode

        if self.mode < 6:
            self.resnet_local = resnet50(True)
            # self.resnet_local = resnest50(True, dilated=True)
            self.resnet_global = None
        if self.mode > 6:
            self.resnet_local = resnet50(True)
            self.resnet_global = resnet50(True)

        # fpn module
        self.fpn_local = None
        self.fpn_global = None

        if mode == 1:
            self.fpn_local = fpn_module_single(numClass)

        if mode == 2 or mode == 3 or mode == 4 or mode == 5:
            self.fpn_local = fpn_module_single(numClass)

            # -------------PacCRF--------------#
            self.crf, self.feat_scales = get_paccrf(numClass, loose=True)
            self.conv = False
            # -------------PacCRF--------------#

            # ConvCRF
            # self.crf, self.feat_scales = get_paccrf(numClass, conv=True)
            # self.conv = True

        # init fpn
        if self.fpn_global is not None:
            for m in self.fpn_global.children():
                if hasattr(m, 'weight'): nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias'): nn.init.constant_(m.bias, 0)
        for m in self.fpn_local.children():
            if hasattr(m, 'weight'): nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'): nn.init.constant_(m.bias, 0)

    def ensemble(self, f_local, f_global):
        return self.ensemble_conv(torch.cat((f_local, f_global), dim=1))

    def forward(self, images_global, segments_global, patch_size, images_255, mode=1):
        # crop images based on mode
        if mode == 1:
            _, image_patches_batch, segment_patches_batch, count_mask_batch, images_255_batch, pad_mask_batch, unfold_info = global2patch(
                images_global,
                segments_global, images_255,
                patch_size,
                mode=mode)
            # train local model with patches only
            c2_l, c3_l, c4_l, c5_l = self.resnet_local.forward(image_patches_batch)
            output_l, _, _, _, _ = self.fpn_local.forward(c2_l, c3_l, c4_l, c5_l)
            output_l = F.softmax(output_l, dim=1)
            segments_l = resize(segment_patches_batch, [patch_size // 4, patch_size // 4],
                                interpolation=0), segment_patches_batch

        # mode 2 with full size input to predict, batch_size=1
        if mode == 2:
            _, _, H, W = images_global[0].size()
            # plt.imshow(images_255[0].squeeze().permute(1,2,0))
            # plt.show()
            c2_l, c3_l, c4_l, c5_l = self.resnet_local.forward(images_global[0])
            output_l, _, _, _, _ = self.fpn_local.forward(c2_l, c3_l, c4_l, c5_l)
            output_l = F.interpolate(output_l, (H, W), mode='nearest')
            # ------------CRF-------------
            images_255[0] = images_255[0] - torch.tensor([122.675, 116.669, 104.008], device=images_255[0].device).view(1, -1, 1, 1)
            output_l = crfarward(self.crf, self.feat_scales, output_l, images_255[0], conv=self.conv)  # softmax is applied automatically
            # ------------CRF-------------
            # plt.imshow(self.segment2rgb(output_l))
            # plt.axis('off')
            # plt.show()
            segments_l = segments_global[0]

        # training with patch, inference with patch, by stitching back.
        if mode == 3:
            _, image_patches_batch, segment_patches_batch, count_mask_batch, images_255_batch, pad_mask_batch, unfold_info = global2patch(
                images_global,
                segments_global, images_255,
                patch_size,
                mode=mode)
            # train local model with patches only
            c2_l, c3_l, c4_l, c5_l = self.resnet_local.forward(image_patches_batch)
            output_l, _, _, _, _ = self.fpn_local.forward(c2_l, c3_l, c4_l, c5_l)

            segments_l = pad_mask_batch[0]  # segment with padding -1, so can work as a mask (now just ignore).

            # append crf
            output_l = F.interpolate(output_l, (patch_size, patch_size), mode='nearest')
            output_l = F.softmax(output_l, dim=1)
            # put back to the original shape
            output_l = patch2global(output_l, count_mask_batch, unfold_info)

            # ------------PacCrf-------------
            images_255_batch[0] = images_255_batch[0] - torch.tensor([122.675, 116.669, 104.008], device=images_255[0].device).view(
                1, -1, 1, 1)
            output_l = crfarward(self.crf, self.feat_scales, output_l, images_255_batch[0], conv=self.conv)
            # ------------PacCrf-------------

        # mode 4 do the self-training task, which takes full image as input, then replace the ground truth before CRF, and after CRF
        if mode == 4:
            idx = np.random.randint(0, 100000)
            _, _, H, W = images_global[0].size()

            # if len(os.listdir("output/train/full_convcrf")) > 400:
            #     exit(0)

            # save original
            # img = Image.fromarray(np.asarray(images_255[0].squeeze().permute(1, 2, 0).cpu()))
            # img.save("output/train/original/{}.jpg".format(idx))

            c2_l, c3_l, c4_l, c5_l = self.resnet_local.forward(images_global[0])
            output_l, _, _, _, _ = self.fpn_local.forward(c2_l, c3_l, c4_l, c5_l)
            output_l = F.interpolate(output_l, (H, W), mode='nearest')

            # save no_crf
            # img = Image.fromarray(np.asarray(segment2rgb(output_l)))
            # img.save("output/test/patch_no_crf/{}.png".format(idx))
            generate_pseudo = False
            # ------------CRF-------------
            segments_l = segments_global[0]
            crfrgb = images_255[0] - torch.tensor([122.675, 116.669, 104.008], device=images_255[0].device).view(
                1, -1, 1, 1)
            if generate_pseudo:
                output_l = crfarward(self.crf, self.feat_scales, output_l, crfrgb, truth=segments_l, conv=self.conv)  # softmax is applied automatically
            else:
                output_l = crfarward(self.crf, self.feat_scales, output_l, crfrgb, truth=None, conv=self.conv)  # soft

            if generate_pseudo:
                path = "output/train/"
                while os.path.isfile(os.path.join(path, "initial_image/{}.jpg".format(idx))):
                    idx = np.random.randint(0, 100000)
                output_index = output_l.argmax(dim=1).squeeze()
                img = Image.fromarray(np.asarray(output_index.cpu()).astype(np.uint8), "L")
                img.save("output/train/initial_label/{}.png".format(idx))
                img = Image.fromarray(np.asarray(images_255[0].squeeze().permute(1, 2, 0).cpu()))
                img.save("output/train/initial_image/{}.jpg".format(idx))

            # save initial prediction

            # save crf
            # img = Image.fromarray(np.asarray(segment2rgb(output_l)))
            # img.save("output/train/full_convcrf/{}.png".format(idx))
            # # save true_mask
            # img = Image.fromarray(np.asarray(segment2rgb(segments_l, mask=True)))
            # img.save("output/train/true/{}.png".format(idx))

        # mode 5 do the self-training task, which takes patch image as input, then replace the ground truth before CRF, and after CRF
        if mode == 5:
            idx = np.random.randint(0, 100000)
            _, image_patches_batch, segment_patches_batch, count_mask_batch, images_255_batch, pad_mask_batch, unfold_info = global2patch(
                images_global,
                segments_global, images_255,
                patch_size,
                mode=mode)
            # train local model with patches only
            c2_l, c3_l, c4_l, c5_l = self.resnet_local.forward(image_patches_batch)
            output_l, _, _, _, _ = self.fpn_local.forward(c2_l, c3_l, c4_l, c5_l)

            segments_l = pad_mask_batch[0]  # segment with padding -1, so can work as a mask (now just ignore).

            # append crf
            output_l = F.interpolate(output_l, (patch_size, patch_size), mode='nearest')
            # output_l = F.softmax(output_l, dim=1)
            # put back to the original shape
            output_l = patch2global(output_l, count_mask_batch, unfold_info)
            # img = Image.fromarray(np.asarray(self.segment2rgb(output_l)))
            # img.save("patch_no_crf/{}.jpg".format(idx))
            # ------------PacCrf-------------
            images_255_batch[0] = images_255_batch[0] - torch.tensor([122.675, 116.669, 104.008],
                                                                     device=images_255[0].device).view(
                1, -1, 1, 1)
            output_l = self.crfarward(output_l, images_255_batch[0], truth=None, conv=self.conv)
            # ------------PacCrf-------------
            # save pac
            img = Image.fromarray(np.asarray(segment2rgb(output_l)))
            img.save("patch_paccrf_test/{}.jpg".format(idx))

            # save predicted pseudo label
            # self.save_segment("/scratch/yh1n19/data/material/localmatdb", seg=output_l, idx=idx,
                              # origin=images_255_batch)

        return output_l, segments_l

    def save_segment(self, path, seg, idx, origin=None):
        if origin is not None:
            img = Image.fromarray(np.asarray(origin[0].squeeze().permute(1, 2, 0).cpu()))
            while os.path.isfile(os.path.join(path, "original_with_pad/{}.jpg".format(idx))):
                idx = np.random.randint(0, 100000)
            img.save(os.path.join(path, "original_with_pad/{}.jpg".format(idx)))
        seg = Image.fromarray(np.asarray(seg.argmax(dim=1).squeeze().cpu(), dtype=np.uint8))
        seg.save(os.path.join(path, "patch_crf_with_true_test/{}.png".format(idx)))
