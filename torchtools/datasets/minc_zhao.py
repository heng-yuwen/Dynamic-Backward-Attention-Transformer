import os
import random

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class MINCPatch(Dataset):
    def __init__(self, root_dir, set_type='train', classes=23,
                 transform=None):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "patch_images_256_new")
        self.set_type = set_type
        self.transform = transform
        self.classes = range(classes)
        # This value has been obtained from the MINC paper
        self.mean = torch.Tensor([124 / 255., 117 / 255., 104 / 255.])
        self.std = torch.Tensor([1, 1, 1])

        # store the index list for each class image
        self.class_image_idx = None

        # Get the material categories from the categories.txt file
        file_name = os.path.join(self.root_dir, 'categories.txt')
        self.categories = dict()
        with open(file_name, 'r') as f:
            for row_id, idx_label in enumerate(f):
                if row_id != 0:
                    label_id = int(idx_label.split()[0])
                    label = idx_label.split()[1]
                    if label_id in self.classes:
                        # The last line char (\n) must be removed
                        self.categories[label_id] = label

        # Load the image data
        set_types = ['train', 'test']
        self.data = []
        if set_type == "train":
            set_num = range(1)
        elif set_type == "test":
            set_num = range(1, 2)
        elif set_type == "all":
            set_num = range(2)
        else:
            raise RuntimeError("invalid data category")

        for i in set_num:
            file_name = os.path.join(self.data_dir, set_types[i] + '.txt')
            with open(file_name, 'r') as f:
                for line in f:
                    # Each row in self.data is composed by:
                    # [label, img_id, patch_center]
                    tmp = line.split()
                    label = int(tmp[1])
                    img_id = tmp[0].split("/")[-1]
                    # Check if the patch label is in the new class set
                    if label in self.categories:
                        path = os.path.join(self.data_dir, str(label), img_id)

                        self.data.append([label, path])

            if set_types[i] == "train":
                self.class_image_idx = self.init_class_index()
                # print("Class index initialised for training set")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx][1]
        patch = Image.open(img_path).convert('RGB')

        if self.transform:
            patch = self.transform(patch)

        # subtract the mean for each channel
        patch = transforms.Compose([
            transforms.Normalize(self.mean, self.std)
        ])(patch)

        label = self.data[idx][0]
        return patch, label

    def init_class_index(self):
        # remember all the index of images for each class label
        idx_dict = dict()
        class_list = np.array([row[0] for row in self.data])
        for class_idx in self.classes:
            idx_dict[class_idx] = np.argwhere(class_list == class_idx).reshape(-1)

        return idx_dict


class MINCSegment(Dataset):
    def __init__(self, root_dir, set_type='train', classes=24):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "full_images")
        self.set_type = set_type
        self.classes = range(classes)
        # This value has been obtained from the MINC paper
        self.mean = torch.Tensor([124 / 255., 117 / 255., 104 / 255.])
        self.std = torch.Tensor([1, 1, 1])

        # Get the material categories from the categories.txt file
        file_name = os.path.join(self.root_dir, 'categories.txt')
        self.categories = dict()
        with open(file_name, 'r') as f:
            new_label_id = 0
            for row_id, idx_label in enumerate(f):
                if row_id != 0:
                    label_id = int(idx_label.split()[0])
                    label = idx_label.split()[1]
                    # if label_id in self.classes:
                        # The last line char (\n) must be removed
                    if label == 255:
                        self.categories[label_id] = [label, -1]
                    else:
                        self.categories[label_id] = [label, new_label_id]
                    new_label_id += 1

        # print(self.categories)

        # Load the image data
        set_types = ['train', 'test']
        self.data = []
        if set_type == "train":
            set_num = range(1)
        elif set_type == "test":
            set_num = range(1, 2)
        elif set_type == "all":
            set_num = range(2)
        else:
            raise RuntimeError("invalid data category")

        for i in set_num:
            file_name = os.path.join(self.data_dir, set_types[i] + '.txt')
            with open(file_name, 'r') as f:
                for line in f:
                    # Each row in self.data is the image name
                    tmp = line.strip()
                    segment_path = os.path.join(self.data_dir, "label_scaled", tmp+".png")
                    origin_path = os.path.join(self.data_dir, "original_scaled", tmp+".jpg")

                    self.data.append([segment_path, origin_path])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        segment_path = self.data[idx][0]
        segment_img = Image.open(segment_path)
        origin_path = self.data[idx][1]
        origin_img = Image.open(origin_path).convert('RGB')

        origin, segment = self.transform(origin_img, segment_img)

        segment = np.array(segment, dtype=np.int64)
        # Transfer to new id
        segment[segment == 255] = -1
        return origin, segment

    def transform(self, origin, segment):
        # Random crop
        # if random.random() > 0.8:
        #     i, j, h, w = transforms.RandomCrop.get_params(
        #         origin, output_size=(224, 224))
        #     origin = TF.crop(origin, i, j, h, w)
        #     segment = TF.crop(segment, i, j, h, w)

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

        return origin, segment
