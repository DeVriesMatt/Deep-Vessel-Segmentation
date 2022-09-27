import os
import re
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.nn import functional as F
from time import sleep
import resource


_nsre = re.compile('([0-9]+)')


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def listdir_sorted_alphanumeric(dir_path):
    return sorted_alphanumeric(os.listdir(dir_path))


def sorted_alphanumeric(items):
    return sorted(items, key=natural_sort_key)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def is_axis(event_axis, name):
    if event_axis is None:
        return False
    if not hasattr(event_axis, 'label'):
        return False
    if event_axis.label is None:
        return False
    return event_axis.label == name


def trim_image(image_path, tile_size):
    """
    Load and trim and image so it is a multiple of the tile size.
    """
    image = Image.open(image_path)
    width, height = image.size
    rounded_width = tile_size * (width // tile_size)
    rounded_height = tile_size * (height // tile_size)
    trimmed_data = image.crop((0, 0, rounded_width, rounded_height))
    trimmed_image = Image.new('RGB', (rounded_width, rounded_height), 255)
    trimmed_image.paste(trimmed_data)
    return trimmed_image


def calc_model_metrics(pl_module, x, y):
    is_training = pl_module.training
    pl_module.eval()
    probas = pl_module(x)
    if is_training:
        pl_module.train()
    return calc_metrics(probas, y)


def calc_metrics(probas, y):
    probas = probas.detach()
    clz = (probas > 0.5).float() * 1
    acc = accuracy_score(y, clz)
    try:
        auc = roc_auc_score(y, probas)
    except ValueError:
        auc = acc
    loss = F.binary_cross_entropy(probas, y).numpy()
    return acc, auc, loss


def calc_roc_curve(probas, y):
    return roc_curve(y, probas)


def print_pause(text):
    sleep(0.5)
    print(text)
    sleep(0.5)


def log_memory_usage(prefix=""):
    if prefix is not "":
        prefix += " "
    print("{:s}Memory Usage: {:.3f}GB".format(prefix, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000000))


def get_full_well_nums():
    """
    Get a list of well numbers that have the full amount of data.
    """
    well_nums = []
    with open("./data/full_well_numbers.csv", 'r') as f:
        for well_num in f:
            well_nums.append(well_num.rstrip())
    print('Using {:d} well numbers'.format(len(well_nums)))
    return well_nums
