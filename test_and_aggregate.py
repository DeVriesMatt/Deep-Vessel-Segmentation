import os

import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import torchvision
from tqdm import tqdm

from loader import get_loader
from networks import UNet, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet, Iternet, AttUIternet, R2UIternet

from torchvision import transforms as T

# Gets the predicted patches
test_loader = get_loader(image_path="test_patches/HRF128/test/",
                            image_size=48,
                            batch_size=1,
                            num_workers=0,
                            mode='test',
                            augmentation_prob=0.)


for i, (images, GT, image_path) in tqdm(enumerate(test_loader)):
    model = AttU_Net(1, 1)
    model.load_state_dict(torch.load('./models/HRF/HRF128/AttU_Net-50-0.0002-10-0.4000_Index_BCE_NoDropout_HRF128.pkl',
                                     map_location=torch.device('cpu')))
    model.train(False)
    model.eval()

    SR = model(images)

    torchvision.utils.save_image(SR.data.cpu(), './result/test_output/HRF128/AttUNet/%s' % image_path)


results = []
for i in tqdm(range(39, 46)):
    DATA_RAW_DIR = "./data/HRF"
    IOSTAR_IMAGE_TEST = DATA_RAW_DIR + "/test_GT"

    image = Image.open(IOSTAR_IMAGE_TEST + "/" + str(i).zfill(5) + ".png")
    width, height = image.size

    rounded_width = 128 * (width // 128)
    rounded_height = 128 * (height // 128)

    trimmed_data = image.crop((0, 0, rounded_width, rounded_height))
    trimmed_image = Image.new('RGB', (rounded_width, rounded_height), 255)
    trimmed_image.paste(trimmed_data)
    slide_image = trimmed_image
    slide_width, slide_height = slide_image.size

    new_image = Image.new('RGB', slide_image.size, 0)
    new_true_GT = Image.new('RGB', slide_image.size, 0)

    # Split and save
    patch_size = 128
    for i_x in range(slide_width // patch_size):
        for i_y in range(slide_height // patch_size):

            patch_image = Image.open(
                "./result/test_output/HRF128/AttUNet/" + str(i).zfill(5) + "_x" + str(i_x).zfill(2) + "_y" + str(i_y).zfill(2) + ".png")
            true_GT = Image.open(
                "./test_patches/HRF128/test_GT/" + str(i).zfill(5) + "_x" + str(i_x).zfill(2) + "_y" + str(i_y).zfill(2) + ".png")
            # black_image =
            x = patch_size * i_x
            y = patch_size * i_y
            box = (x, y, x + patch_size, y + patch_size)
            new_image.paste(patch_image, box)
            new_true_GT.paste(true_GT, box)

    new_image.save("result/test_whole_image/HRF128/AttUNet/" + str(i).zfill(5) + ".png")
    new_true_GT.save("result/test_whole_image_true/HRF128/" + str(i).zfill(5) + ".png")

    from metrics import *

    SR = Image.open("result/test_whole_image/HRF128/AttUNet/" + str(i).zfill(5) + ".png")
    GT = Image.open("result/test_whole_image_true/HRF128/" + str(i).zfill(5) + ".png")

    Transform = []
    Transform.append(T.ToTensor())
    Transform = T.Compose(Transform)
    SR = Transform(SR)
    # print(torch.max(SR[3]))

    GT = Transform(GT)
    # print(torch.max(GT[0]))

    acc = get_accuracy(SR[0], GT[0])
    sensitivity = get_sensitivity(SR[0], GT[0])
    specificity = get_specificity(SR[0], GT[0])
    dice = get_DC(SR[0], GT[0])
    jaccard = get_JS(SR[0], GT[0])
    results.append({'acc': acc,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'dice': dice,
                    'jaccard': jaccard})
    print(get_accuracy(SR[0], GT[0]))
    print(get_sensitivity(SR[0], GT[0]))
    print(get_specificity(SR[0], GT[0]))
    print(get_DC(SR[0], GT[0]))
    print(get_JS(SR[0], GT[0]))

print(results)
