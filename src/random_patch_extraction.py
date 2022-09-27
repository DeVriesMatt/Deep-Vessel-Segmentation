"""
Random patch extraction for training
"""

import csv
import os
import urllib.request
import zipfile
from random import randint
from shutil import copyfile

from PIL import Image
from tqdm import tqdm
import numpy as np

from util import create_dir_if_not_exist


def create_patch(image_path, gt_path, patch_dir, patch_size, patch_per_image, inside=True):
    # Create dirs
    gt_dir = patch_dir + "_GT"
    image_dir = patch_dir
    create_dir_if_not_exist(gt_dir)
    create_dir_if_not_exist(image_dir)
    create_dir_if_not_exist("random")

    # Iterate through files to split and group them
    image_files = os.listdir(image_path)
    print(len(image_files), "slide images found")
    iter_tot = 0
    for image_file in tqdm(image_files, desc="Splitting images"):
        if "DS_Store" not in image_file:
            image = Image.open(image_path + "/" + image_file)
            image_np = np.asarray(image)
            # print(image_np.shape)

            gt = Image.open(gt_path + "/" + image_file[:-3] + 'png')
            gt_np = np.asarray(gt)
            # print(gt_np.shape)
            gt_np = np.reshape(gt_np, (gt_np.shape[0], gt_np.shape[1], 1))
            # print(gt_np.shape)

            width, height = image.size

            save_dir_image = image_dir
            save_dir_gt = gt_dir

            # Round to lowest multiple of target width and height.
            # Will lead to a loss of image data around the edges, but ensures split images are all the same size.
            rounded_width = patch_size * (width // patch_size)
            rounded_height = patch_size * (height // patch_size)
            # TODO: Added from https://github.com/QTIM-Lab/retinaunet/blob/master/lib/extract_patches.py

            k = 0
            patches = []
            patches_gt = []
            while k < patch_per_image:
                x_center = randint(0 + int(patch_size / 2), width - int(patch_size / 2))
                # print "x_center " +str(x_center)
                y_center = randint(0 + int(patch_size / 2), height - int(patch_size / 2))
                # print "y_center " +str(y_center)
                # check whether the patch is fully contained in the FOV
                # if inside == True:
                #     if is_patch_inside_FOV(x_center, y_center, img_w, img_h, patch_h) == False:
                #         continue
                patch = image_np[x_center - int(patch_size / 2):x_center + int(patch_size / 2),
                        y_center - int(patch_size / 2):y_center + int(patch_size / 2),
                        :]
                patch_mask = gt_np[x_center - int(patch_size / 2):x_center + int(patch_size / 2),
                             y_center - int(patch_size / 2):y_center + int(patch_size / 2),
                             :]
                patches.append(patch)
                patches_gt.append(patch_mask)

                box = (x_center - int(patch_size / 2), y_center - int(patch_size / 2),
                       x_center + int(patch_size / 2), y_center + int(patch_size / 2))
                cropped_data = image.crop(box)
                cropped_data_gt = gt.crop(box)

                cropped_image = Image.new('RGB', (patch_size, patch_size), 255)
                cropped_image.paste(cropped_data)

                cropped_image_gt = Image.new('RGB', (patch_size, patch_size), 255)
                cropped_image_gt.paste(cropped_data_gt)

                # if inside:
                #     if is_patch_inside_FOV(x_center, y_center, width, height, patch_size) == False:
                #         continue

                # if np.mean(np.asarray(cropped_image_gt)) == 0:
                #     continue
                # else:
                # print(np.mean(np.asarray(cropped_image_gt)[:, :, :1]))

                iter_tot += 1  # total
                k += 1  # per full_img

                cropped_image.save(save_dir_image + "/" + str(iter_tot).zfill(5) + ".png")

                cropped_image_gt.save(save_dir_gt + "/" + str(iter_tot).zfill(5) + ".png")

            # print(patches)

            # return patches  #, patches_masks
    print('Created', iter_tot, 'split images')


# check if the patch is fully contained in the FOV
def is_patch_inside_FOV(x, y, img_w, img_h, patch_h):
    x_ = x - int(img_w / 2)  # origin (0,0) shifted to image center
    y_ = y - int(img_h / 2)  # origin (0,0) shifted to image center
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0)
    # radius is 270 (from DRIVE db docs), minus the patch diagonal
    # (assumed it is a square # this is the limit to contain the full
    # patch in the FOV
    radius = np.sqrt((x_ * x_) + (y_ * y_))
    if radius < R_inside:
        return True
    else:
        return False


if __name__ == "__main__":
    patch_size = 48

    DATA_RAW_DIR = "./data/DRIVE/training"
    IOSTAR_IMAGE = DATA_RAW_DIR + "/train"
    IOSTAR_GT = DATA_RAW_DIR + "/train_GT"

    PROCESSED_IOSTAR_DIR_IMAGE = "./random/DRIVE/train"
    PROCESSED_IOSTAR_DIR_GT = "./random/DRIVE/train_GT"

    IOSTAR_IMAGE_VAL = DATA_RAW_DIR + "/valid"
    IOSTAR_GT_VAL = DATA_RAW_DIR + "/valid_GT"

    PROCESSED_IOSTAR_DIR_IMAGE_VAL = "./random/DRIVE/valid"
    PROCESSED_IOSTAR_DIR_GT_VAL = "./random/DRIVE/valid_GT"

    IOSTAR_IMAGE_TEST = DATA_RAW_DIR + "/test"
    IOSTAR_GT_TEST = DATA_RAW_DIR + "/test_GT"

    PROCESSED_IOSTAR_DIR_IMAGE_TEST = "./random/DRIVE/test"
    PROCESSED_IOSTAR_DIR_GT_TEST = "./random/DRIVE/test_GT"

    print('.########  DRIVE  #######.')
    print('===================== splitting Train ====================================')
    create_patch(IOSTAR_IMAGE, IOSTAR_GT, PROCESSED_IOSTAR_DIR_IMAGE, patch_size, 2500)

    print('===================== splitting Validation ====================================')
    create_patch(IOSTAR_IMAGE_VAL, IOSTAR_GT_VAL, PROCESSED_IOSTAR_DIR_IMAGE_VAL, patch_size, 2500)

    print('===================== splitting Test ====================================')
    create_patch(IOSTAR_IMAGE_TEST, IOSTAR_GT_TEST, PROCESSED_IOSTAR_DIR_IMAGE_TEST, patch_size, 2500)

    # STARE
    DATA_RAW_DIR = "./data/STAREALL"
    # EXAMPLE_SLIDES_ZIP = DATA_RAW_DIR + "/example_slides.zip"
    IOSTAR_IMAGE = DATA_RAW_DIR + "/train"
    IOSTAR_GT = DATA_RAW_DIR + "/train_GT"

    PROCESSED_IOSTAR_DIR_IMAGE = "./random/STAREALL/train"
    PROCESSED_IOSTAR_DIR_GT = "./random/STAREALL/train_GT"

    IOSTAR_IMAGE_VAL = DATA_RAW_DIR + "/valid"
    IOSTAR_GT_VAL = DATA_RAW_DIR + "/valid_GT"

    PROCESSED_IOSTAR_DIR_IMAGE_VAL = "./random/STAREALL/valid"
    PROCESSED_IOSTAR_DIR_GT_VAL = "./random/STAREALL/valid_GT"

    IOSTAR_IMAGE_TEST = DATA_RAW_DIR + "/test"
    IOSTAR_GT_TEST = DATA_RAW_DIR + "/test_GT"

    PROCESSED_IOSTAR_DIR_IMAGE_TEST = "./random/STAREALL/test"
    PROCESSED_IOSTAR_DIR_GT_TEST = "./random/STAREALL/test_GT"

    print('.########  STARE  #######.')
    print('===================== splitting Train ====================================')
    create_patch(IOSTAR_IMAGE, IOSTAR_GT, PROCESSED_IOSTAR_DIR_IMAGE, patch_size, 2500)

    print('===================== splitting Validation ====================================')
    create_patch(IOSTAR_IMAGE_VAL, IOSTAR_GT_VAL, PROCESSED_IOSTAR_DIR_IMAGE_VAL, patch_size, 2500)

    print('===================== splitting Test ====================================')
    create_patch(IOSTAR_IMAGE_TEST, IOSTAR_GT_TEST, PROCESSED_IOSTAR_DIR_IMAGE_TEST, patch_size, 2500)

    # CHASEDB1
    DATA_RAW_DIR = "./data/CHASEDB1"
    # EXAMPLE_SLIDES_ZIP = DATA_RAW_DIR + "/example_slides.zip"
    IOSTAR_IMAGE = DATA_RAW_DIR + "/train"
    IOSTAR_GT = DATA_RAW_DIR + "/train_GT"

    PROCESSED_IOSTAR_DIR_IMAGE = "./random/CHASEDB1/train"
    PROCESSED_IOSTAR_DIR_GT = "./random/CHASEDB1/train_GT"

    IOSTAR_IMAGE_VAL = DATA_RAW_DIR + "/valid"
    IOSTAR_GT_VAL = DATA_RAW_DIR + "/valid_GT"

    PROCESSED_IOSTAR_DIR_IMAGE_VAL = "./random/CHASEDB1/valid"
    PROCESSED_IOSTAR_DIR_GT_VAL = "./random/CHASEDB1/valid_GT"

    IOSTAR_IMAGE_TEST = DATA_RAW_DIR + "/test"
    IOSTAR_GT_TEST = DATA_RAW_DIR + "/test_GT"

    PROCESSED_IOSTAR_DIR_IMAGE_TEST = "./random/CHASEDB1/test"
    PROCESSED_IOSTAR_DIR_GT_TEST = "./random/CHASEDB1/test_GT"

    print('.########  CHASEDB1  #######.')
    print('===================== splitting Train ====================================')
    create_patch(IOSTAR_IMAGE, IOSTAR_GT, PROCESSED_IOSTAR_DIR_IMAGE, patch_size, 2500)

    print('===================== splitting Validation ====================================')
    create_patch(IOSTAR_IMAGE_VAL, IOSTAR_GT_VAL, PROCESSED_IOSTAR_DIR_IMAGE_VAL, patch_size, 2500)

    print('===================== splitting Test ====================================')
    create_patch(IOSTAR_IMAGE_TEST, IOSTAR_GT_TEST, PROCESSED_IOSTAR_DIR_IMAGE_TEST, patch_size, 2500)

    # HRF
    DATA_RAW_DIR = "./data/HRF"
    # EXAMPLE_SLIDES_ZIP = DATA_RAW_DIR + "/example_slides.zip"
    IOSTAR_IMAGE = DATA_RAW_DIR + "/train"
    IOSTAR_GT = DATA_RAW_DIR + "/train_GT"

    PROCESSED_IOSTAR_DIR_IMAGE = "./random/HRF/train"
    PROCESSED_IOSTAR_DIR_GT = "./random/HRF/train_GT"

    IOSTAR_IMAGE_VAL = DATA_RAW_DIR + "/valid"
    IOSTAR_GT_VAL = DATA_RAW_DIR + "/valid_GT"

    PROCESSED_IOSTAR_DIR_IMAGE_VAL = "./random/HRF/valid"
    PROCESSED_IOSTAR_DIR_GT_VAL = "./random/HRF/valid_GT"

    IOSTAR_IMAGE_TEST = DATA_RAW_DIR + "/test"
    IOSTAR_GT_TEST = DATA_RAW_DIR + "/test_GT"

    PROCESSED_IOSTAR_DIR_IMAGE_TEST = "./random/HRF/test"
    PROCESSED_IOSTAR_DIR_GT_TEST = "./random/HRF/test_GT"

    print('.########  HRF  #######.')
    print('===================== splitting Train ====================================')
    create_patch(IOSTAR_IMAGE, IOSTAR_GT, PROCESSED_IOSTAR_DIR_IMAGE, patch_size, 1600)

    print('===================== splitting Validation ====================================')
    create_patch(IOSTAR_IMAGE_VAL, IOSTAR_GT_VAL, PROCESSED_IOSTAR_DIR_IMAGE_VAL, patch_size, 1600)

    print('===================== splitting Test ====================================')
    create_patch(IOSTAR_IMAGE_TEST, IOSTAR_GT_TEST, PROCESSED_IOSTAR_DIR_IMAGE_TEST, patch_size, 1600)

    print("DONE!!!")
