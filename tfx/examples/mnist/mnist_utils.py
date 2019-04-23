from __future__ import absolute_import, division

import os

from logzero import logger
from tensorflow.contrib.learn.python.learn.datasets import mnist


def maybe_download(data_dir='/tmp/data/mnist'):
    """
    Will download MNIST gzip files and move them to the correct
    locations if not already present
    """
    should_download = True

    for x in ['train', 'val']:
        target_dir = os.path.join(data_dir, x)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        else:
            # dir's exist, so we assume data is already present
            # and in the correct locations
            should_download = False

    if not should_download:
        logger.info('data already present at: %s; no need to download', data_dir)
        return

    # downloads MNIST datasets
    data_sets = mnist.read_data_sets(data_dir)

    # move to desired locations and rename
    def move_to_dest(target_dir, from_file, to_file):
        try:
            os.rename(
                os.path.join(data_dir, from_file),
                os.path.join(target_dir, to_file)
            )
        except OSError:
            # file already moved
            pass

    # train
    train_dir = os.path.join(data_dir, 'train')
    move_to_dest(train_dir, 'train-images-idx3-ubyte.gz', 'images.gz')
    move_to_dest(train_dir, 'train-labels-idx1-ubyte.gz', 'labels.gz')

    # test
    val_dir = os.path.join(data_dir, 'val')
    move_to_dest(val_dir, 't10k-images-idx3-ubyte.gz', 'images.gz')
    move_to_dest(val_dir, 't10k-labels-idx1-ubyte.gz', 'labels.gz')

    logger.info('data successfully downloaded and moved to: %s', data_dir)
