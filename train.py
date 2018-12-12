import os
import argparse
from random import shuffle

import numpy as np
from imgaug import augmenters as iaa
from neupy import algorithms, storage

from data import (
    read_data, subtract_channel_mean, crop_image, pad_values,
    make_annotation_one_hot_encoded, TRAIN_SET, VALIDATION_SET
)
from model import create_deeplab_model, download_resnet50_weights
from utils import get_confusion_matrix, segmentation_metrics


parser = argparse.ArgumentParser()
parser.add_argument('--storage-folder', '-s', required=True)
parser.add_argument('--image-size', '-i', type=int, default=513)
parser.add_argument('--batch-size', '-b', type=int, default=10)
parser.add_argument('--epochs', '-e', type=int, default=30)

image_augmentation = iaa.Sequential([
    iaa.Add((-32, 32)),  # change brightness
    iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
    iaa.ContrastNormalization((0.5, 1.5)),  # improve or worsen the contrast
])
affine = iaa.Affine(scale=(0.5, 2))  # zoom-in and zoom-out on image
flip = iaa.Fliplr(0.5)  # left-right flip with 50% probability


def create_data_iterator(datapath, crop_size, batch_size, use_augmentation=True):
    images, annotations = read_data(datapath)
    n_samples = len(images)

    def iter_batches():
        indices = np.arange(n_samples)
        shuffle(indices)  # randomize the order

        # Note: We will exclude last incomplete batch
        for i in range(0, n_samples, batch_size):
            batch_indeces = indices[i:i + batch_size]
            image_batch, annotation_batch = [], []

            if len(batch_indeces) < batch_size:
                # Skip batches that smaller than expected batch size
                # These batches might have negative effect on batch norm
                # layers and gradient estimation.
                continue

            for index in batch_indeces:
                image, annotation = images[index], annotations[index]

                if use_augmentation:
                    affine_det = affine.to_deterministic()
                    flip_det = flip.to_deterministic()

                    image = image_augmentation.augment_image(image)
                    image = affine_det.augment_image(image)
                    image = flip_det.augment_image(image)

                    annotation = affine_det.augment_image(annotation)
                    annotation = flip_det.augment_image(annotation)

                image = crop_image(image, crop_size)
                image = pad_values(image, crop_size, value=0)
                image = subtract_channel_mean(image)

                # crop image function assumes that input has channel
                annotation = crop_image(np.expand_dims(annotation, axis=2), crop_size)[:, :, 0]
                annotation = make_annotation_one_hot_encoded(annotation)
                annotation = pad_values(annotation, crop_size, value=0)

                image_batch.append(image)
                annotation_batch.append(annotation)

            yield (
                np.stack(image_batch, axis=0),
                np.stack(annotation_batch, axis=0),
            )

    return iter_batches


if __name__ == '__main__':
    args = parser.parse_args()
    storage_folder = args.storage_folder

    if not os.path.exists(storage_folder):
        os.mkdir(storage_folder)

    resnet50, deeplab = create_deeplab_model(
        download_resnet50_weights(),
        size=args.image_size,
    )

    print("Loading training data...")
    training_iterator = create_data_iterator(
        TRAIN_SET,
        args.image_size,
        args.batch_size,
        use_augmentation=True,
    )

    print("Loading validation data...")
    vaidation_iterator = create_data_iterator(
        VALIDATION_SET,
        args.image_size,
        batch_size=60,
        use_augmentation=False,
    )

    optimizer = algorithms.Adam(
        deeplab,

        error='categorical_crossentropy',
        step=0.00001,
        verbose=True,

        addons=[algorithms.WeightDecay],
        decay_rate=0.0001,
    )

    for i in range(args.epochs):
        print("Epoch #{}".format(i + 1))

        for x_batch, y_batch in training_iterator():
            x_batch = resnet50.predict(x_batch)
            optimizer.train(x_batch, y_batch, epochs=1, summary='inline')

        print("Start validation")
        val_images, val_annotations = next(vaidation_iterator())
        segmentation = deeplab.predict(resnet50.predict(val_images))
        confusion = get_confusion_matrix(val_annotations, segmentation)

        accuracy, miou = segmentation_metrics(confusion)
        print("Val accuracy: {:.3f}".format(accuracy))
        print("Val miou: {:.3f}".format(miou))

        filename = 'deeplab_{:0>3}_{:.3f}_{:.3f}.hdf5'.format(i, accuracy, miou)
        filepath = os.path.join(storage_folder, filename)

        print("Saved: {}".format(filepath))
        storage.save(deeplab, filepath)
