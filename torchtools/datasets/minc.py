import os
import random

import torch
import numpy as np
import zipfile
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class MINC(Dataset):
    def __init__(self, root_dir, set_type='train', classes=23,
                 scale=0.233, transform=None):
        self.root_dir = os.path.join(root_dir, "minc")
        self.set_type = set_type
        self.transform = transform
        self.scale = scale
        self.classes = range(classes)
        # This value has been obtained from the MINC paper
        self.mean = torch.Tensor([124.0/255, 117.0/255, 104.0/255])
        self.std = torch.Tensor([1, 1, 1])
        # store the index list for each class image
        self.class_image_idx = None
        # Use zip to save space
        self.zipdata = zipfile.ZipFile(os.path.join(self.root_dir, "photo_orig.zip"), mode='r')
        # I exploit the fact that several patches are obtained from the
        # same image by saving the last used image and by reusing it
        # whenever possible
        self.last_img = dict()
        self.last_img["img_path"] = ''

        # Get the material categories from the categories.txt file
        file_name = os.path.join(self.root_dir, 'categories.txt')
        self.categories = dict()
        new_class_id = 0
        with open(file_name, 'r') as f:
            for class_id, class_name in enumerate(f):
                if class_id in self.classes:
                    # The last line char (\n) must be removed
                    self.categories[class_id] = [class_name[:-1], new_class_id]
                    new_class_id += 1

        # Load the image data
        set_types = ['train', 'validate', 'test']
        self.data = []
        if set_type == "train":
            set_num = range(1)
        elif set_type == "validate":
            set_num = range(1, 2)
        elif set_type == "test":
            set_num = range(2, 3)
        elif set_type == "all":
            set_num = range(3)
        else:
            raise RuntimeError("invalid data category")

        for i in set_num:
            file_name = os.path.join(self.root_dir, set_types[i] + '_clean.txt')
            with open(file_name, 'r') as f:
                for line in f:
                    # Each row in self.data is composed by:
                    # [label, img_id, patch_center]
                    tmp = line.split(',')
                    label = int(tmp[0])
                    # Check if the patch label is in the new class set
                    if label in self.categories:
                        img_id = tmp[1]
                        patch_x = float(tmp[2])
                        # The last line char (\n) must be removed
                        patch_y = float(tmp[3][:-1])
                        path = os.path.join(img_id[-1], img_id + '.jpg')
                        patch_center = [patch_x, patch_y]
                        self.data.append([self.categories[label][1], path,
                                          patch_center])

            if set_types[i] == "train":
                self.class_image_idx = self.init_class_index()
                # print("Class index initialised for training set")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx][1]
        if self.last_img["img_path"] != img_path:
            # Sometimes the images are opened as grayscale,
            # so I need to force RGB
            self.last_img["img_path"] = img_path
            self.last_img["image"] = Image.open(self.zipdata.open(self.last_img["img_path"])).convert('RGB')

        width, height = self.last_img["image"].size
        patch_center = self.data[idx][2]
        patch_center = [patch_center[0] * width,
                        patch_center[1] * height]
        if width < height:
            patch_size = int(width * self.scale)
        else:
            patch_size = int(height * self.scale)
        box = (patch_center[0] - patch_size / 2,
               patch_center[1] - patch_size / 2,
               patch_center[0] + patch_size / 2,
               patch_center[1] + patch_size / 2)
        # print(box, width, height)
        patch = self.last_img["image"].crop(box)
        if self.transform:
            patch = self.transform(patch)

        # subtract the mean for each channel
        patch = transforms.Compose([
            transforms.Normalize(self.mean, self.std)
        ])(patch)
        # patch = torch.from_numpy(np.array(patch, dtype=np.float32))
        # patch = patch - self.mean.view(1, 1, -1)
        # patch = patch.permute(2, 0, 1)

        label = self.data[idx][0]
        return patch, label

    def init_class_index(self):
        # remember all the index of images for each class label
        idx_dict = dict()
        class_list = np.array([row[0] for row in self.data])
        for class_idx in self.classes:
            idx_dict[class_idx] = np.argwhere(class_list == class_idx).reshape(-1)

        return idx_dict


class MINCPointSegment(Dataset):
    def __init__(self, root_dir, set_type='train', classes=23):
        self.root_dir = os.path.join(root_dir, "minc")
        self.set_type = set_type
        self.classes = range(classes)
        # This value has been obtained from the MINC paper
        self.mean = torch.Tensor([124.0/255, 117.0/255, 104.0/255])
        self.std = torch.Tensor([1, 1, 1])
        # store the index list for each class image
        self.size = 512

        # I exploit the fact that several patches are obtained from the
        # same image by saving the last used image and by reusing it
        # whenever possible
        self.last_img = dict()
        self.last_img["img_path"] = ''
        # Use zip to save space
        self.zipdata = zipfile.ZipFile(os.path.join(self.root_dir, "photo_orig.zip"), mode='r')
        # Get the material categories from the categories.txt file
        file_name = os.path.join(self.root_dir, 'categories.txt')
        self.categories = dict()
        new_class_id = 0
        with open(file_name, 'r') as f:
            for class_id, class_name in enumerate(f):
                if class_id in self.classes:
                    # The last line char (\n) must be removed
                    self.categories[class_id] = [class_name[:-1], new_class_id]
                    new_class_id += 1

        # Load the image data
        set_types = ['train', 'validate', 'test']
        self.data = {}
        if set_type == "train":
            set_num = range(1)
        elif set_type == "validate":
            set_num = range(1, 2)
        elif set_type == "test":
            set_num = range(2, 3)
        elif set_type == "all":
            set_num = range(3)
        else:
            raise RuntimeError("invalid data category")

        for i in set_num:
            file_name = os.path.join(self.root_dir, set_types[i] + '.txt')
            with open(file_name, 'r') as f:
                for row_id, line in enumerate(f):
                    # Each row in self.data is composed by:
                    # [label, img_id, patch_center]
                    tmp = line.split(',')
                    tmp_label = int(tmp[0])

                    # Check if the patch label is in the new class set
                    if tmp_label in self.categories:
                        img_id = tmp[1]
                        patch_x = float(tmp[2])
                        # The last line char (\n) must be removed
                        patch_y = float(tmp[3][:-1])
                        tmp_path = os.path.join(img_id[-1], img_id + '.jpg')
                        if img_id not in self.data.keys():
                            self.data[img_id] = [tmp_path, [[patch_x, patch_y]], [tmp_label]]
                        else:
                            self.data[img_id][1].append([patch_x, patch_y])
                            self.data[img_id][2].append(tmp_label)
        self.data_keys = list(self.data.keys())

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx):
        img_meta = self.data[self.data_keys[idx]]
        img_path = img_meta[0]
        if self.last_img["img_path"] != img_path:
            # Sometimes the images are opened as grayscale,
            # so I need to force RGB
            self.last_img["img_path"] = img_path
            self.last_img["image"] = Image.open(self.zipdata.open(self.last_img["img_path"])).convert('RGB')

        segment = np.ones((self.size, self.size), dtype=np.int16) * 255
        for patch_center, label in zip(img_meta[1], img_meta[2]):
            patch_center = [int(patch_center[1] * (self.size-1)),
                            int(patch_center[0] * (self.size-1))]
            segment[patch_center[0], patch_center[1]] = label

        origin, segment = self.transform(self.last_img["image"], Image.fromarray(segment))

        # patch = torch.from_numpy(np.array(patch, dtype=np.float32))
        # patch = patch - self.mean.view(1, 1, -1)
        # patch = patch.permute(2, 0, 1)

        return origin, segment

    def transform(self, origin, segment):
        origin = TF.resize(origin, [self.size, self.size])

        # Random horizontal flipping
        if random.random() > 0.5:
            origin = TF.hflip(origin)
            segment = TF.hflip(segment)

        # Random vertical flipping
        if random.random() > 0.5:
            origin = TF.vflip(origin)
            segment = TF.vflip(segment)

        # Transform to tensor
        origin = TF.to_tensor(origin)
        # subtract the mean for each channel
        origin = transforms.Compose([
            transforms.Normalize(self.mean, self.std)
        ])(origin)

        segment = np.array(segment, dtype=np.int64)
        segment[segment==255] = -1

        return origin, segment
