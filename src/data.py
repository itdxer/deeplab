import os
import sys
import os.path
from collections import namedtuple

import cv2
import numpy as np
from tqdm import tqdm


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
BASEPATH = os.path.join(CURRENT_DIR, "..", "dataset", "VOCdevkit", "VOC2012")

IMAGESETS = os.path.join(BASEPATH, "ImageSets", "Segmentation")
IMAGES = os.path.join(BASEPATH, "JPEGImages")
ANNOTATIONS = os.path.join(BASEPATH, "SegmentationClass")

TRAIN_SET = os.path.join(IMAGESETS, "train.txt")
TRAINVAL_SET = os.path.join(IMAGESETS, "trainval.txt")
VALIDATION_SET = os.path.join(IMAGESETS, "val.txt")

VOCClass = namedtuple("VOCClass", "index color name")
voc_classes = [
    VOCClass(index=0, color=(0, 0, 0), name="background"),
    VOCClass(index=1, color=(128, 0, 0), name="aeroplane"),
    VOCClass(index=2, color=(0, 128, 0), name="bicycle"),
    VOCClass(index=3, color=(128, 128, 0), name="bird"),
    VOCClass(index=4, color=(0, 0, 128), name="boat"),
    VOCClass(index=5, color=(128, 0, 128), name="bottle"),
    VOCClass(index=6, color=(0, 128, 128), name="bus"),
    VOCClass(index=7, color=(128, 128, 128), name="car"),
    VOCClass(index=8, color=(64, 0, 0), name="cat"),
    VOCClass(index=9, color=(192, 0, 0), name="chair"),
    VOCClass(index=10, color=(64, 128, 0), name="cow"),
    VOCClass(index=11, color=(192, 128, 0), name="diningtable"),
    VOCClass(index=12, color=(64, 0, 128), name="dog"),
    VOCClass(index=13, color=(192, 0, 128), name="horse"),
    VOCClass(index=14, color=(64, 128, 128), name="motorbike"),
    VOCClass(index=15, color=(192, 128, 128), name="person"),
    VOCClass(index=16, color=(0, 64, 0), name="potted plant"),
    VOCClass(index=17, color=(128, 64, 0), name="sheep"),
    VOCClass(index=18, color=(0, 192, 0), name="sofa"),
    VOCClass(index=19, color=(128, 192, 0), name="train"),
    VOCClass(index=20, color=(0, 64, 128), name="tv/monitor"),
]


def iter_data(filepath):
    with open(filepath) as f:
        for line in f:
            image_path = os.path.join(IMAGES, line.strip() + '.jpg')
            annotation_path = os.path.join(ANNOTATIONS, line.strip() + '.png')

            yield (
                cv2.imread(image_path),
                process_annotation(cv2.imread(annotation_path)),
            )


def read_data(filepath):
    images = []
    annotations = []

    for image, annotation in tqdm(iter_data(filepath)):
        images.append(image)
        annotations.append(annotation)

    return images, annotations


def find_pads(difference):
    half_difference = int(difference // 2)

    if difference % 2 == 0:
        return (half_difference, half_difference)

    return (half_difference, half_difference + 1)


def pad_values(image, size, value):
    in_height, in_width, _ = image.shape

    return np.pad(
        image, [
            find_pads(size - in_height),
            find_pads(size - in_width),
            (0, 0),  # no padding for channels
        ],
        mode='constant',
        constant_values=value,
    )


def crop_image(image, size):
    in_height, in_width, _ = image.shape

    if in_height > size:
        left_diff, right_diff = find_pads(in_height - size)
        image = image[left_diff:in_height - right_diff, :, :]

    if in_width > size:
        top_diff, bottom_diff = find_pads(in_width - size)
        image = image[:, top_diff:in_width - bottom_diff, :]

    return image


def process_annotation(annotation_3d):
    # From: https://github.com/wuhuikai/DeepGuidedFilter
    annotation_3d = annotation_3d[:, :, (2, 1, 0)]  # use only RGB
    height, width, _ = annotation_3d.shape
    annotation_2d = np.zeros((height, width), dtype=np.uint8)

    # Unknown colors will not get any matches with existed casses and
    # they will be automatically marked as 0 by default. For this reason
    # we add +1 to every index to make sure that we can differentiate them
    # from unknown classes.
    for voc_class in voc_classes:
        class_color = np.array(voc_class.color).reshape(1, 1, 3)
        mask = np.all(annotation_3d == class_color, axis=2)
        annotation_2d[mask] = voc_class.index + 1

    return annotation_2d


def make_annotation_one_hot_encoded(annotation_2d):
    height, width = annotation_2d.shape
    n_classes = len(voc_classes)

    xs, ys = np.nonzero(annotation_2d)

    annotation_onehot = np.zeros((height, width, n_classes), dtype=np.uint8)
    annotation_onehot[xs, ys, annotation_2d[xs, ys] - 1] = 1

    return annotation_onehot


def subtract_channel_mean(image):
    image = image.astype(np.float32)
    # Per channel normalization, input BGR
    image[:, :, 2] -= 123.68
    image[:, :, 1] -= 116.78
    image[:, :, 0] -= 103.94
    return image
