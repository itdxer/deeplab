import argparse

import numpy as np
from tqdm import tqdm

from data import (
    VALIDATION_SET, iter_data,
    make_annotation_one_hot_encoded, subtract_channel_mean,
)
from model import create_deeplab_model, download_resnet50_weights
from utils import get_confusion_matrix, segmentation_metrics, show


parser = argparse.ArgumentParser()
parser.add_argument('--deeplab-weights', required=True)


def summarize(confusion, iteration):
    accuracy, miou = segmentation_metrics(confusion)
    print('\n' + '-' * 40)
    print("Validation after {} iterations".format(iteration))
    print("Val accuracy: {:.3f}".format(accuracy))
    print("Val miou: {:.3f}".format(miou))


if __name__ == '__main__':
    args = parser.parse_args()
    resnet50_weights = download_resnet50_weights()

    resnet50, deeplab = create_deeplab_model(
        resnet50_weights, args.deeplab_weights)

    deeplab = resnet50 > deeplab
    confusion = np.zeros((21, 21))

    print("Start validation")
    for i, (image, annotation) in tqdm(enumerate(iter_data(VALIDATION_SET))):
        annotation = make_annotation_one_hot_encoded(annotation)
        annotation = np.expand_dims(annotation, axis=0)

        image = subtract_channel_mean(image)
        image = np.expand_dims(image, axis=0).astype(np.float32)

        segmentation = deeplab.predict(image)
        confusion += get_confusion_matrix(annotation, segmentation)

        if i % 20 == 0 and i != 0:
            summarize(confusion, i + 1)

    summarize(confusion, i + 1)
