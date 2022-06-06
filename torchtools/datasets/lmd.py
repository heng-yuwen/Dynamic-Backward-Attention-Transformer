import os
import random
import pickle
import PIL.Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import zipfile
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LMDSegment(Dataset):
    """
    Return the images based on mode given
    """
    def __init__(self, root_dir, set_type='train', classes=16, use_transform=True, split="1", required_H=512, required_W=512):
        if set_type != "infer":
            self.root_dir = os.path.join(root_dir, "localmatdb")
            self.img_dir = os.path.join("images")
            self.zipdata = zipfile.ZipFile(os.path.join(self.root_dir, "localmatdb.zip"), mode='r')
        self.set_type = set_type
        self.classes = range(classes)
        self.use_transform = use_transform
        # This value has been obtained from the MINC paper
        self.category2code = {"asphalt": 0, "ceramic": 1, "concrete": 2, "fabric": 3, "foliage": 4,
                              "food": 5, "glass": 6, "metal": 7, "paper": 8, "plaster": 9, "plastic": 10,
                              "rubber": 11, "soil": 12, "stone": 13, "water": 14, "wood": 15}
        self.required_H = required_H
        self.required_W = required_W

        # Load the image data
        set_types = ["train", "validate", "test", "test_indoor", "infer", "validate_fixed"]
        self.set_type = set_type
        self.size = required_H

        # Load image names
        if set_type not in set_types:
            raise RuntimeError("invalid data category")
        # with open(os.path.join(self.root_dir, set_type), 'rb') as fp:
        if set_type in ["train", "validate", "test", "validate_fixed"]:
            if set_type == "train":
                # format_string = "balanced_{}_{}"
                format_string = "{}_{}"
            else:
                format_string = "{}_{}"
            if set_type != "validate_fixed":
                self.image_names = list(pickle.load(self.zipdata.open(format_string.format(set_type, split))))
            else:
                self.image_names = list(pickle.load(self.zipdata.open(format_string.format("validate", split))))
            # Load all mask file paths
            self.mask_paths = [path for path in self.zipdata.namelist() if "mask" in path]

        elif set_type == "test_indoor":
            self.stage = "test_indoor"
            self.image_names = ["COCO_train2014_000000510577.jpg", "COCO_train2014_000000555211.jpg",
                                "COCO_train2014_000000417814.jpg", "COCO_train2014_000000203887.jpg",
                                "COCO_train2014_000000377583.jpg", "COCO_train2014_000000111022.jpg",
                                "COCO_train2014_000000341917.jpg", "2009_000212.jpg"]
            self.maskdata = zipfile.ZipFile(os.path.join(self.root_dir, "{}.zip".format(self.stage)), mode='r')

        elif set_type == "infer":
            self.stage = "infer"
            self.image_names = []
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                        self.image_names.append(os.path.join(root, file))


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.set_type == "test_indoor":
            origin_path = os.path.join(self.img_dir, self.image_names[idx])
            origin_img = Image.open(self.zipdata.open(origin_path)).convert('RGB')
            segment_paths = os.path.join("{}".format(self.stage), origin_path.split("/")[1][:-4] + ".png")
            segment_img = Image.open(self.maskdata.open(segment_paths))
            origin_img, segment_img = self._transform_test(origin_img, segment_img)
            name = origin_path.split("/")[-1][:-4]
        elif self.set_type != "infer":
            origin_path = os.path.join(self.img_dir, self.image_names[idx])
            origin_img = Image.open(self.zipdata.open(origin_path)).convert('RGB')
            segment_paths = [path for path in self.mask_paths if self.image_names[idx] in path]

            # get the category from path str like ./COCO_train2014_000000010073.jpg_food_mask.png
            segment_categories = []
            for seg_path in segment_paths:
                category = seg_path.split("_")[-2]
                if category not in segment_categories:
                    segment_categories.append(category)
                else:
                    not_refined_mask_path = [path for path in segment_paths if category in path and "refinedmask" not in path]
                    for path in not_refined_mask_path:
                        segment_paths.remove(path)

            # load the masks and merge them into one
            segment_img = np.full((origin_img.size[::-1]), 255, dtype=np.uint8) # default all 255, means unknown
            for segment_path in segment_paths:
                temp_mask = np.asarray(Image.open(self.zipdata.open(segment_path)))
                # print(segment_path)
                assert temp_mask.shape == segment_img.shape
                # If multiple, test segment_img==255 to avoid overwriting.
                segment_img[(temp_mask!=0) & (segment_img==255)] = self.category2code[segment_path.split("_")[-2]]

            segment_img = Image.fromarray(segment_img)
            # print(segment_categories)

            # Transform to argument the training data, and fit for batch training (same size, 512*640)
            if self.use_transform and self.set_type == "train":
                origin_img, segment_img = self._transform_train(origin_img, segment_img)
            elif self.use_transform and self.set_type == "validate_fixed":
                origin_img, segment_img = self._transform_validate_fixed(origin_img, segment_img)
            elif self.use_transform:
                origin_img, segment_img = self._transform_test(origin_img, segment_img)
            name = origin_path.split("/")[-1][:-4]
        else:
            origin_img = Image.open(self.image_names[idx]).convert('RGB')
            segment_img = None
            origin_path = self.image_names[idx]
            origin_img, segment_img = self._transform_test(origin_img, segment_img)
            segment_img = torch.randn(size=origin_img.size).permute(1,0)
            name = origin_path.split("/")
            name = name[-2] + "/" + name[-1][:-4]

        return origin_img, segment_img, name

    def _transform_validate_fixed(self, origin, segment):
        # resize
        origin, segment = self._resize(origin, segment)

        # Random crop
        center_crop = transforms.CenterCrop((self.required_H, self.required_W))
        origin = center_crop(origin)
        segment = center_crop(segment)

        return origin, segment

    def _transform_train(self, origin, segment):
        """
        Resize then randomly crop the image into required size.
        :param origin: Pillow img
        :param segment: Pillow img
        :return: cropped image
        """

        # resize
        origin, segment = self._resize(origin, segment)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            origin, output_size=(self.required_H, self.required_W))
        origin = TF.crop(origin, i, j, h, w)
        segment = TF.crop(segment, i, j, h, w)

        if random.random() > 0.5:
            degree = random.choice([90, 180, 270])
            origin = transforms.functional.rotate(origin, degree)
            segment = transforms.functional.rotate(segment, degree)

        return origin, segment

    def _resize(self, origin, segment):
        W, H = origin.size
        if W >= H:
            if H < self.required_H:
                origin = origin.resize((W * self.required_H // H, self.required_H), Image.BICUBIC)
                if segment is not None:
                    segment = segment.resize((W * self.required_H // H, self.required_H), resample=PIL.Image.NEAREST)
        else:
            if W < self.required_W:
                origin = origin.resize((self.required_W, H * self.required_W // W), Image.BICUBIC)
                if segment is not None:
                    segment = segment.resize((self.required_W, H * self.required_W // W), resample=PIL.Image.NEAREST)

        if segment is not None:
            return origin, segment
        else:
            return origin

    def _transform_test(self, origin, segment):
        # resize only the image
        origin = self._resize(origin, None)
        # cropped_origins = self._crop_overlay(origin)

        return origin, segment

    def _crop_overlay(self, origin):
        W, H = origin.size

        num_W = W // self.required_W if W % self.required_W == 0 else W // self.required_W + 1
        num_H = H // self.required_H if H % self.required_H == 0 else H // self.required_H + 1
        cropped_origins = []
        for col in range(num_W):
            for row in range(num_H):
                left_step = (W - self.required_W) // (num_W-1) if num_W != 1 else 0
                top_step = (H - self.required_H) // (num_H-1) if num_H != 1 else 0
                if col*left_step+self.required_W <= W and top_step*row+self.required_H <= H:
                    box = (col*left_step, top_step*row, col*left_step+self.required_W, top_step*row+self.required_H)
                else:
                    if col*left_step+self.required_W > W:
                        left_step = W - self.required_W
                    else:
                        left_step = col*left_step+self.required_W
                    if top_step*row+self.required_H > H:
                        top_step = H - self.required_H
                    else:
                        top_step = top_step*row+self.required_H
                    box = (left_step, top_step, left_step+self.required_W, top_step+self.required_H)

                cropped_origins.append(origin.crop(box))

        return cropped_origins
