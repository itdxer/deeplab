import os
import random
import unittest
from collections import Counter

import cv2
import numpy as np

from src.data import (
    subtract_channel_mean, add_zero_paddings, find_missing_borders_along_dimension, crop_image,
    process_annotation, make_annotation_one_hot_encoded,
)


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
ANNOTATION_DIR = os.path.join(CURRENT_DIR, 'annotated-images')


class DataProcessingTestCase(unittest.TestCase):
    def test_image_cropping(self):
        for _ in range(100):
            height = random.randint(10, 30)
            width = random.randint(10, 30)
            size = random.randint(20, 40)

            original_image = np.ones((height, width, 3))
            cropped_image = crop_image(original_image, size)

            self.assertEqual(
                cropped_image.shape,
                (min(height, size), min(width, size), 3),
            )
            np.testing.assert_array_almost_equal(np.unique(cropped_image), [1])

    def test_find_missing_borders_along_dimension(self):
        self.assertEqual(find_missing_borders_along_dimension(0), (0, 0))
        self.assertEqual(find_missing_borders_along_dimension(4), (2, 2))
        self.assertEqual(find_missing_borders_along_dimension(5), (2, 3))

    def test_image_padding(self):
        for _ in range(100):
            height = random.randint(10, 30)
            width = random.randint(10, 30)
            size = max(height, width, random.randint(20, 40))

            original_image = np.ones((height, width, 3))
            padded_image = add_zero_paddings(original_image, size)

            self.assertEqual(padded_image.shape, (size, size, 3))
            self.assertEqual(original_image.sum(), padded_image.sum())

            expected_values = [1]
            if original_image.shape != padded_image.shape:
                expected_values = [0, 1]

            np.testing.assert_array_almost_equal(
                np.unique(padded_image), expected_values)

    def test_image_channel_subtraction(self):
        origina_image = 255 * np.ones((10, 10, 3)).astype(np.uint8)
        image = subtract_channel_mean(origina_image)

        self.assertTrue(np.allclose(255 - 103.94, image[:, :, 0]))
        self.assertTrue(np.allclose(255 - 116.78, image[:, :, 1]))
        self.assertTrue(np.allclose(255 - 123.68, image[:, :, 2]))

    def test_image_channel_subtraction_error(self):
        with self.assertRaises(ValueError):
            subtract_channel_mean(np.random.random((1, 10, 10, 3)))

        with self.assertRaises(ValueError):
            subtract_channel_mean(np.random.random((10, 10)))

    def test_annotation_processing_image_1(self):
        annotation_path = os.path.join(ANNOTATION_DIR, "2007_000033.png")
        labeled_image = process_annotation(cv2.imread(annotation_path))

        actual_class_counters = Counter(labeled_image.ravel())
        expected_class_counters = {
            1: 143868,  # background class
            2: 30937,  # plane class
            0: 8195,  # unknown class
        }
        self.assertDictEqual(
            actual_class_counters,
            expected_class_counters,
        )

    def test_annotation_processing_image_2(self):
        annotation_path = os.path.join(ANNOTATION_DIR, "2007_001420.png")
        labeled_image = process_annotation(cv2.imread(annotation_path))

        actual_class_counters = Counter(labeled_image.ravel())

        expected_class_counters = {
            0: 7589,  # unknown class
            1: 137802,  # background class
            14: 10990,  # horse
            16: 5706,  # person
            17: 3913,  # potted plant
        }
        self.assertDictEqual(
            actual_class_counters,
            expected_class_counters,
        )

    def test_annotation_onehot_encoding(self):
        annotation_path = os.path.join(ANNOTATION_DIR, "2007_000033.png")
        labeled_image = process_annotation(cv2.imread(annotation_path))
        onehot_annotation = make_annotation_one_hot_encoded(labeled_image)

        actual_class_counters = Counter(onehot_annotation.argmax(axis=-1).ravel())
        expected_class_counters = {
            0: 143868 + 8195,  # background + unknown class
            1: 30937,  # plane class
        }

        self.assertEqual(onehot_annotation.shape, (366, 500, 21))
        self.assertEqual(0, onehot_annotation.max(axis=-1).min())
        self.assertEqual(8195, (onehot_annotation.max(axis=-1) == 0).sum())
        self.assertDictEqual(
            actual_class_counters,
            expected_class_counters,
        )
