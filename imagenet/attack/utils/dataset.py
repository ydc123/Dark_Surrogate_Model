import os
import cv2
import numpy as np
import pandas as pd
import scipy.misc
from imageio import imread
from imageio import imsave

class DatasetMetadata_Imagenet():
    def __init__(self, csv):
        self.df = df = pd.read_csv(csv)

    def get_true_label(self, image_id, targeted=False):
        if targeted:
            row = self.df[self.df.ImageId == image_id]['TargetClass']
        else:
            row = self.df[self.df.ImageId == image_id]['TrueLabel']
        assert(len(row) == 1)
        return row.values[0] - 1


def load_images(input_dir, batch_shape, targeted, img_num):
    """Read png images from input directory in batches.

    Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, 3, height, width]

    Yields:
    filenames: list file names without path of each image
                    Lenght of this list could be less than batch_size, in this case only
                    first few images of the result are elements of the minibatch.
    images: array with all images from this batch
    """
    if  batch_shape[2] is None:
        filepath = '{}/{}'.format(input_dir, os.listdir(input_dir)[0])
        with open(filepath,'rb') as f:
            image = np.array(imread(f))
        if batch_shape[-1] is not None:
            batch_shape = (batch_shape[0], image.shape[0], image.shape[1], batch_shape[-1])
        else:
            batch_shape = (batch_shape[0], batch_shape[1], image.shape[0], image.shape[1])
    images = []
    y_dev = []
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    csv_dir = 'input/dev_dataset.csv'
    datameta = DatasetMetadata_Imagenet(csv_dir)
    filename_list = sorted(os.listdir(input_dir))[0:img_num]
    for fname in filename_list:
        image_id = fname[:-4]
        filepath = '{}/{}'.format(input_dir, fname)
        with open(filepath,'rb') as f:
            image = np.array(imread(f))

            if batch_shape[1] == 3:
                image = cv2.resize(image, (batch_shape[2], batch_shape[3]))
                image = image.transpose([2, 0, 1])
            else:
                image = cv2.resize(image, (batch_shape[1], batch_shape[2]))
            image = image.astype(np.float32) / 255
        images.append(image)
        filenames.append(os.path.basename(filepath))
        y_dev.append(datameta.get_true_label(image_id, targeted))
        idx += 1
        if idx == batch_size:
            images = np.array(images)
            y_dev = np.array(y_dev)
            yield filenames, images, y_dev
            filenames = []
            images = []
            y_dev = []
            idx = 0
    if idx > 0:
        images = np.array(images)
        y_dev = np.array(y_dev)
        yield filenames, images, y_dev


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        with open(os.path.join(output_dir, filename), 'wb') as f:
            img = (images[i, :, :, :] * 255).astype(np.uint8).transpose([1, 2, 0])
            imsave(f, img, format='png')
