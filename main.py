import os
import torch
import train
import argparse
import numpy as np

from train import GAN_CLS
from torch.utils.data import DataLoader
from data_loader import Text2ImageDataset


def check_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print ('{} created'.format(dir_name))

'''
def check_args(args):
    # Make all directories if they don't exist

    # --checkpoint_dir
    check_dir(args.checkpoint_dir)

    # --sample_dir
    check_dir(args.sample_dir)

    # --log_dir
    check_dir(args.log_dir)

    # --final_model dir
    check_dir(args.final_model)

    # --epoch
    assert args.num_epochs > 0, 'Number of epochs must be greater than 0'

    # --batch_size
    assert args.batch_size > 0, 'Batch size must be greater than zero'

    # --z_dim
    assert args.z_dim > 0, 'Size of the noise vector must be greater than zero'

    return args
'''

def main():
    dataset = Text2ImageDataset('flowers/')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    gan = GAN_CLS(data_loader)

    gan.build_model()
    gan.train_model()


if __name__ == '__main__':
    main()
