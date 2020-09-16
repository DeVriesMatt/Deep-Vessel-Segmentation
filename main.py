import argparse
import os
import time

from trainer import Solver
from loader import get_loader
from torch.backends import cudnn

"""Adapted from https://github.com/LeeJunHyun/Image_Segmentation"""


def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['UNet', 'R2U_Net', 'AttU_Net', 'R2AttU_Net', 'Iternet', 'AttUIternet', 'R2UIternet',
                                 'NestedUNet']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net/Iternet/AttUIternet/R2UIternet/NestedUNet')
        print('Your input for model_type was %s' % config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    print(config)

    train_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0.)
    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    start = time.time()
    history, results = solver.train()
    stop = time.time()
    training_time = stop - start
    print(history)
    print(results)
    print("Training time: {}".format(training_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=48)  # TODO: change for image patches
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)  # TODO: change for image channel to be green only
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_epochs_decay', type=int, default=10)
    parser.add_argument('--decay_factor', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.002)  # Original LR 0.0002
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='UNet',
                        help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/Iternet/AttUIternet/R2UItenet/NestedUNet')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--train_path', type=str, default='./processed/STARE/train/')
    parser.add_argument('--valid_path', type=str, default='./processed/STARE/valid/')
    parser.add_argument('--test_path', type=str, default='./processed/STARE/test/')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
