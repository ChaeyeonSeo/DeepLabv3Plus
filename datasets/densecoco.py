import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class DenseCOCO(data.Dataset):

    DenseCOCOClass = namedtuple('DenseCOCOClass', ['name', 'id', 'color'])
    classes = [
        DenseCOCOClass('background',            0, (0, 0, 0)),
        DenseCOCOClass('trunk',                 1, (1, 1, 1)),
        DenseCOCOClass('right hand',            2, (2, 2, 2)),
        DenseCOCOClass('left hand',             3, (3, 3, 3)),
        DenseCOCOClass('left foot',             4, (4, 4, 4)),
        DenseCOCOClass('right foot',            5, (5, 5, 5)),
        DenseCOCOClass('right thigh',           6, (6, 6, 6)),
        DenseCOCOClass('left thigh',            7, (7, 7, 7)),
        DenseCOCOClass('right calf',            8, (8, 8, 8)),
        DenseCOCOClass('left calf',             9, (9, 9, 9)),
        DenseCOCOClass('left arm',              10, (10, 10, 10)),
        DenseCOCOClass('right arm',             11, (11, 11, 11)),
        DenseCOCOClass('left forearm',          12, (12, 12, 12)),
        DenseCOCOClass('right forearm',         13, (13, 13, 13)),
        DenseCOCOClass('head',                  14, (14, 14, 14)),
    ]

    id_to_color = [c.color for c in classes]
    id_to_color = np.array(id_to_color)
    
    def __init__(self, root, image_folder, mask_folder, transform=None):
        self.root = root

        self.images_dir = os.path.join(self.root, image_folder)

        self.targets_dir = os.path.join(self.root, mask_folder)
        self.transform = transform

        self.images = []
        self.targets = []

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        ori_list = os.listdir(self.images_dir)
        ori_list.sort()
        mask_list = os.listdir(self.targets_dir)
        mask_list.sort()

        for ori in ori_list:
            self.images.append(os.path.join(self.images_dir, ori))
        for mask in mask_list:
            self.targets.append(os.path.join(self.targets_dir, mask))

    @classmethod
    def encode_target(cls, target):
        return np.array(target)

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 0
        return cls.id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)