import os
import random
import pickle

import PIL.Image
import torch
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from numpy.random import RandomState
import pandas as pd
import zipfile
import json


def parse_vertices(vertices_str):
    """
    Parse vertices stored as a string.

    :param vertices: "x1,y1,x2,y2,...,xn,yn"
    :param return: [(x1,y1), (x1, y2), ... (xn, yn)]
    """
    s = [float(t) for t in vertices_str.split(',')]
    return zip(s[::2], s[1::2])


def parse_triangles(triangles_str):
    """
    Parse a list of vertices.

    :param vertices: "v1,v2,v3,..."
    :return: [(v1,v2,v3), (v4, v5, v6), ... ]
    """
    s = [int(t) for t in triangles_str.split(',')]
    return zip(s[::3], s[1::3], s[2::3])


class OPSurface(Dataset):
    """
    Return the images based on mode given
    """

    def __init__(self, root_dir, set_type='train', use_transform=True, required_H=512,
                 required_W=512):
        self.root_dir = os.path.join(root_dir, "opensurface")
        self.img_dir = os.path.join("opensurface", "photos")
        self.zipdata = zipfile.ZipFile(os.path.join(self.root_dir, "opensurface.zip"), mode='r')
        self.set_type = set_type
        self.use_transform = use_transform

        # This value has been obtained from the MINC paper, only useful in saving the colored segments
        # self.material2code = pd.read_csv(self.zipdata.open("opensurface/label-substance-colors.csv"))
        # self.object2code = pd.read_csv(self.zipdata.open("opensurface/label-name-colors.csv"))
        # self.scene2code = pd.read_csv(self.zipdata.open("opensurface/label-scene-colors.csv"))

        self.photo_labels = self.load_photos()
        print("map shapes onto photos...")
        self.photo_shapes = {p['pk']: [] for p in self.photo_labels}
        for s in self.load_shapes():
            l = self.photo_shapes.get(s['photo'], None)
            if l is not None:
                l.append(s)

        material_ignore = ['Chalkboard/blackboard', 'Cork/corkboard', 'Dirt', 'Fire', 'Granite', 'Linoleum', 'Marble', 'Paper towel/tissue', 'Plaster', 'Sky', 'Sponge', 'Styrofoam', 'Wallboard - painted', 'Wallboard - unpainted', 'Water', 'Wax', 'Wood - natural color', 'Wood - painted']

        materials = [(p['pk'], p['name']) for p in self.load_json('shapes.shapesubstance.json') if not p['fail'] and not p["name"] in material_ignore]
        objects = [(p['pk'], p['name']) for p in self.load_json('shapes.shapename.json') if not p['fail']]
        scenes = [(p['pk'], p['name']) for p in self.load_json('photos.photoscenecategory.json')]

        materials.sort(key=lambda x: x[1])
        objects.sort(key=lambda x: x[1])
        scenes.sort(key=lambda x: x[1])

        # material index to colour (the training gt)
        # ignore_materials = ["Foliage", "Wax", "Linoleum", "Sponge", "Sky", "Water", "Chalkboard/blackboard", "Cork/corkboard",
        #                    "Fire", "Styrofoam"]
        # group_material = {"Granite/marble": "Granite", "Paper towel/tissue": "Paper/tissue", "Plastic - opaque": "Plastic - clear",
        #                   "Wallboard - unpainted": "Wallboard - painted"}

        red_color = 1
        self.material_to_red = {}
        for pk, name in materials:
            # if name not in ignore_materials:
            self.material_to_red[pk] = red_color
            red_color += 1

        # ignore_objects = []
        green_color = 1
        self.object_to_green = {}
        for pk, name in objects:
            self.object_to_green[pk] = green_color
            green_color += 1

        self.scene_to_blue = {}
        blue_color = 1
        for pk, name in scenes:
            self.scene_to_blue[pk] = blue_color
            blue_color += 1

        self.required_H = required_H
        self.required_W = required_W

        # Load the image data
        set_types = ["train", "validate", "test"]
        if set_type not in set_types:
            raise RuntimeError("invalid data category")

        idx_np = np.array(range(len(self.photo_labels)))
        prng = RandomState(42)
        prng.shuffle(idx_np)
        # print(idx_np[:10])
        # load the name of all images in zipfile
        # self.image_names = []
        # for zipinfo in self.zipdata.filelist:
        #     if "opensurface/photos/" in zipinfo.filename:
        #         self.image_names.append(zipinfo.filename)
        # self.image_names.remove("opensurface/photos/")
        print(len(idx_np))
        if set_type == "train":
            self.idx_list = idx_np[: int(len(idx_np) * 0.7)]
        elif set_type == "validate":
            self.idx_list = idx_np[int(len(idx_np) * 0.7): int(len(idx_np) * 0.85)]
        else:
            self.idx_list = idx_np[int(len(idx_np) * 0.85):]

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        photo_info = self.photo_labels[self.idx_list[idx]]
        origin_path = os.path.join(self.img_dir, str(photo_info['pk']) + ".jpg")
        origin_img = Image.open(self.zipdata.open(origin_path)).convert('RGB')
        origin_img = self._resize(origin_img, None)
        photo_info["image_width"], photo_info["image_height"] = origin_img.size
        # parse the labels of a single image, with material, object, scene in R, G, B channels.

        # get the category from path str like ./COCO_train2014_000000010073.jpg_food_mask.png
        segment_img = self.render_single_photo_labels(photo_info, self.photo_shapes[photo_info["pk"]],
                                                      self.material_to_red, self.object_to_green, self.scene_to_blue)

        # Transform to argument the training data, and fit for batch training (same size, such as 512*512)
        if self.use_transform and self.set_type == "train":
            origin_img, segment_img = self._transform_train(origin_img, segment_img)
        elif self.use_transform:
            origin_img, segment_img = self._transform_test(origin_img, segment_img)

        return origin_img, segment_img, photo_info["pk"]

    def render_single_photo_labels(self, photo, photo_shapes, material_to_red, object_to_green, scene_to_blue):
        """ Render labels for one photo """
        w = photo["image_width"]
        h = photo["image_height"]

        blue_color = scene_to_blue.get(photo['scene_category'], 0)
        labels_image = Image.new(mode='RGB', size=(w, h), color=(0, 0, blue_color))
        draw = ImageDraw.Draw(labels_image)

        # render shapes that are part of this photo
        for shape in photo_shapes:
            # red encodes substance (material)
            # green encodes name (object)
            color = (
                material_to_red.get(shape['substance'], 0),
                object_to_green.get(shape['name'], 0),
                blue_color
            )

            # extract triangles
            triangles = parse_triangles(shape['triangles'])

            # extract vertices and rescale to pixel coordinates
            vertices = parse_vertices(shape['vertices'])
            vertices = [(int(x * w), int(y * h)) for (x, y) in vertices]

            # render triangles
            for tri in triangles:
                points = [vertices[tri[t]] for t in (0, 1, 2)]
                draw.polygon(points, fill=color)

        del draw
        return labels_image

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
            if H != self.required_H:
                origin = origin.resize((W * self.required_H // H, self.required_H), Image.BICUBIC)
                if segment is not None:
                    segment = segment.resize((W * self.required_H // H, self.required_H), resample=PIL.Image.NEAREST)
        else:
            if W != self.required_W:
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
                left_step = (W - self.required_W) // (num_W - 1) if num_W != 1 else 0
                top_step = (H - self.required_H) // (num_H - 1) if num_H != 1 else 0
                if col * left_step + self.required_W <= W and top_step * row + self.required_H <= H:
                    box = (col * left_step, top_step * row, col * left_step + self.required_W,
                           top_step * row + self.required_H)
                else:
                    if col * left_step + self.required_W > W:
                        left_step = W - self.required_W
                    else:
                        left_step = col * left_step + self.required_W
                    if top_step * row + self.required_H > H:
                        top_step = H - self.required_H
                    else:
                        top_step = top_step * row + self.required_H
                    box = (left_step, top_step, left_step + self.required_W, top_step + self.required_H)

                cropped_origins.append(origin.crop(box))

        return cropped_origins

    def load_json(self, filename):
        """ Load a JSON object """

        if not os.path.isfile(filename):
            filename = os.path.join("opensurface", 'opensurfaces', filename)
        print('parsing %s...' % filename)
        items = json.loads(self.zipdata.open(filename).read().decode("utf-8"))
        return [dict(list(p['fields'].items()) + [('pk', p['pk'])]) for p in items]

    def load_photos(self):
        """ Load all photos """

        print('load photos...')
        photos = self.load_json('photos.photo.json')

        print('throw out photos with incorrect scene category')
        photos = filter(lambda x: x['scene_category_correct'], photos)

        print('throw out synthetic photos')
        photos = filter(lambda x: not x['special'], photos)

        print('sort by num_vertices, then scene_category_correct, then scene_category_correct_score')
        photos = list(photos)
        photos.sort(key=lambda x: (x['num_vertices'], x['scene_category_correct'], x['scene_category_correct_score']),
                    reverse=True)

        return photos

    def load_shapes(self):
        """ Load all material shapes """

        print('load shapes...')
        shapes = self.load_json('shapes.materialshape.json')

        print('throw out synthetic shapes')
        shapes = filter(lambda x: not x['special'], shapes)

        print('throw out low quality shapes')
        shapes = filter(lambda x: x['high_quality'], shapes)

        print('sort by num_vertices')
        shapes = list(shapes)
        shapes.sort(key=lambda x: (x['num_vertices']), reverse=True)

        return shapes
