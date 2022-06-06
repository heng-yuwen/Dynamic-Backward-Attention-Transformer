import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from torchtools.datasets.lmd import LMDSegment
from torchtools.models.dpglt.helper import images_transform, masks_transform
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import crop
from math import ceil


def collate(batch):
    images = [b[0] for b in batch]
    segments = [b[1] for b in batch]
    names = [b[2] for b in batch]
    assert len(images) == len(segments)

    images_tensor = images_transform(images)
    segments_tensor = masks_transform(segments)
    images_255 = torch.cat([torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(dim=0) for image in images])

    return images_tensor, segments_tensor, images_255, names


def get_patch_info(patch_size, H, W):
    num_H = H // patch_size if H % patch_size == 0 else H // patch_size + 1
    num_W = W // patch_size if W % patch_size == 0 else W // patch_size + 1

    stride_H = patch_size if H % patch_size == 0 else ceil((H-patch_size) / (num_H - 1))
    stride_W = patch_size if W % patch_size == 0 else ceil((W-patch_size) / (num_W - 1))

    H_padded = stride_H * (num_H - 1) + patch_size
    W_padded = stride_W * (num_W - 1) + patch_size

    pad_H = H_padded - H
    pad_W = W_padded - W

    return pad_H, pad_W, stride_H, stride_W, H_padded, W_padded


def patch2global(tensor_patches, count_masks, patch_count, resized_size, patch_size=512):
    # restore global to original size
    merged_tensors = []
    sum = 0
    for idx, count in enumerate(patch_count):
        tensors = tensor_patches[sum: sum+count]
        sum += count
        H, W = resized_size[idx]
        pad_H, pad_W, stride_H, stride_W, H_padded, W_padded = get_patch_info(patch_size, H, W)
        C = tensors.size()[1]
        count_mask = count_masks[idx]

        tensors = tensors.permute(1, 2, 3, 0)
        count_mask = count_mask.permute(1, 2, 0)

        assert count_mask.size() == tensors.size()[1:], "the number of patches do not match"

        tensors = tensors.contiguous().view(C * patch_size ** 2, -1)
        count_mask = count_mask.contiguous().view(patch_size ** 2, -1)

        tensors = F.fold(tensors.unsqueeze(dim=0), output_size=(H_padded, W_padded),
                                kernel_size=patch_size, stride=(stride_H, stride_W))
        count_mask = F.fold(count_mask.unsqueeze(dim=0), output_size=(H_padded, W_padded), kernel_size=patch_size,
                            stride=(stride_H, stride_W))

        tensors = tensors / count_mask
        tensors = crop(tensors, pad_H // 2 + pad_H % 2, pad_W // 2 + pad_W % 2, H, W)
        assert tensors.size(-1) == W and tensors.size(-2) == H, "Wrong cropped region. {} does not match {}".format(tensors.size(), (H, W))
        merged_tensors.append(tensors)

    return merged_tensors


def split_overlap_img_tensor(images_tensor, patch_size=512):
    splitted_tensors = []
    count_mask_tensors = []
    patch_count = []
    resized_size = []
    for img_tensor in images_tensor:
        img_tensor = img_tensor.unsqueeze(0)
        _, C, H, W = img_tensor.size()
        pad_H, pad_W, stride_H, stride_W, H_padded, W_padded = get_patch_info(patch_size, H, W)
        resized_size.append((H, W))
        img_tensor = F.pad(img_tensor, (pad_W // 2 + pad_W % 2, pad_W // 2, pad_H // 2 + pad_H % 2, pad_H // 2))
        count_mask = torch.ones([1, H_padded, W_padded], device=img_tensor.device)
        image_patches = img_tensor.unfold(2, patch_size, stride_H).unfold(3, patch_size, stride_W).squeeze()
        count_mask = count_mask.unfold(1, patch_size, stride_H).unfold(2, patch_size, stride_W).squeeze()
        image_patches = image_patches.contiguous().view(C, -1, patch_size, patch_size)
        image_patches = image_patches.permute(1, 0, 2, 3)
        count_mask = count_mask.contiguous().view(-1, patch_size, patch_size)

        patch_count.append(image_patches.size(0))
        count_mask_tensors.append(count_mask)
        splitted_tensors.append(image_patches)

    return torch.cat(splitted_tensors, dim=0), count_mask_tensors, patch_count, resized_size


def collate_test(batch):
    images = [b[0] for b in batch]
    segments = [b[1] for b in batch]
    names = [b[2] for b in batch]
    assert len(images) == len(segments)

    images_tensor = images_transform(images, training=False)
    segments_tensor = masks_transform(segments, training=False)
    images_255 = [torch.from_numpy(np.array(image)).permute(2, 0, 1) for image in images]
    splitted_tensors, count_mask_tensors, patch_count, resized_size = split_overlap_img_tensor(images_tensor)
    splitted_images_255, _, _, _ = split_overlap_img_tensor(images_255)

    return splitted_tensors, count_mask_tensors, patch_count, resized_size, segments_tensor, images_255, names


class LMDSegLoader(LightningDataModule):
    def __init__(self, data_root, batch_size=1, classes=16, split="1"):
        super().__init__()
        self._use_gpu = torch.cuda.is_available()
        self._data_root = data_root
        self._classes = classes
        self._batch_size = batch_size
        self.split = split
        # self._batch_size = 10

    def train_dataloader(self):
        train_set = LMDSegment(root_dir=self._data_root, set_type='train',
                               classes=self._classes, use_transform=True, split=self.split)
        print("Training set loaded, with {} samples".format(len(train_set)))
        # sampler = BalancedDistributedSampler(train_set, 46)
        # sampler = DistributedSampler(train_set, shuffle=True)

        return DataLoader(dataset=train_set,
                          batch_size=self._batch_size,
                          pin_memory=self._use_gpu,
                          # sampler=sampler,
                          collate_fn=collate)

    def val_dataloader(self, sampler=None):
        val_set = LMDSegment(root_dir=self._data_root, set_type='validate',
                             classes=self._classes, split=self.split)
        print("Validation set loaded, with {} samples".format(len(val_set)))
        # sampler = DistributedSampler(val_set, shuffle=False)
        # sampler = RandomSampler(val_set, replacement=True, num_samples=46)

        return DataLoader(dataset=val_set,
                          batch_size=self._batch_size,
                          shuffle=False,
                          pin_memory=self._use_gpu,
                          sampler=sampler,
                          collate_fn=collate_test)

    def val_fixed_dataloader(self, sampler=None):
        val_set = LMDSegment(root_dir=self._data_root, set_type='validate_fixed',
                             classes=self._classes, split=self.split)
        print("Validation set loaded, with {} samples".format(len(val_set)))
        # sampler = DistributedSampler(val_set, shuffle=False)
        # sampler = RandomSampler(val_set, replacement=True, num_samples=46)

        return DataLoader(dataset=val_set,
                          batch_size=self._batch_size,
                          shuffle=False,
                          pin_memory=self._use_gpu,
                          sampler=sampler,
                          collate_fn=collate_test)

    def test_dataloader(self):
        test_set = LMDSegment(root_dir=self._data_root, set_type='test',
                              classes=self._classes, split=self.split)
        print("Test set loaded, with {} samples".format(len(test_set)))
        # if torch.cuda.device_count() > 1:
        # sampler = DistributedSampler(test_set, shuffle=False)
        # sampler = RandomSampler(test_set, replacement=True, num_samples=46)
        return DataLoader(dataset=test_set,
                          batch_size=self._batch_size,
                          shuffle=False,
                          pin_memory=self._use_gpu,
                          # sampler=sampler,
                          collate_fn=collate_test)

    def test_indoor_dataloader(self):
        test_set = LMDSegment(root_dir=self._data_root, set_type='test_indoor',
                              classes=self._classes, split=self.split)
        print("Test indoor set loaded, with {} samples".format(len(test_set)))
        # if torch.cuda.device_count() > 1:
        # sampler = DistributedSampler(test_set, shuffle=False)
        # sampler = RandomSampler(test_set, replacement=True, num_samples=46)
        return DataLoader(dataset=test_set,
                          batch_size=self._batch_size,
                          shuffle=False,
                          pin_memory=self._use_gpu,
                          # sampler=sampler,
                          collate_fn=collate_test)

    def infer_dataloader(self, path):
        infer_set = LMDSegment(root_dir=path, set_type='infer',
                              classes=self._classes, split=self.split)
        print("Infer images loaded, with {} samples".format(len(infer_set)))
        # if torch.cuda.device_count() > 1:
        # sampler = DistributedSampler(test_set, shuffle=False)
        # sampler = RandomSampler(test_set, replacement=True, num_samples=46)
        return DataLoader(dataset=infer_set,
                          batch_size=self._batch_size,
                          shuffle=False,
                          pin_memory=self._use_gpu,
                          # sampler=sampler,
                          collate_fn=collate_test)
