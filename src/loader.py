import os
import random
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
import cv2
"""Adapted from https://github.com/LeeJunHyun/Image_Segmentation/blob/master/data_loader.py"""


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=128, mode='train', augmentation_prob=0.4):  # TODO: change image size for patch
        """Initializes image paths and preprocessing module."""
        self.root = root

        # GT : Ground Truth
        self.GT_paths = root[:-1] + '_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(
            root)))
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""

        image_path = self.image_paths[index]
        filename = image_path[-17:]  # TODO: Change when using image patches because of naming convention: 17 for patch 9 for not patch
        GT_path = (self.GT_paths + filename)  # TODO: bug when crating images to name a space

        # for using Clahe
        image = cv2.imread(image_path)
        GT = cv2.imread(GT_path)

        # image
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # GT
        lab = cv2.cvtColor(GT, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        GT = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        p_transform = random.random()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:

            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)

            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)


        Transform = []
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        image = Transform(image)
        # Take green channel only
        image = image[1:2, :, :]

        GT = Transform(GT)
        GT = GT[0:1, :, :]

        return image, GT, filename

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=8, mode='train', augmentation_prob=0.4):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader
