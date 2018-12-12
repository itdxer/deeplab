import unittest

import numpy as np

from src.utils import score_segmentation, reverse_indeces


class UtilsTestCase(unittest.TestCase):
    def test_segmentation_scores(self):
        expected = np.array([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        expected = np.stack([expected, 1 - expected], axis=2)
        predicted = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        predicted = np.stack([predicted, 1 - predicted], axis=2)

        accuracy, miou = score_segmentation(expected, predicted)
        self.assertAlmostEqual(accuracy, 20 / 30)

        iou_ones = 4 / 14
        iou_zeros = 16 / 26
        self.assertAlmostEqual(miou, (iou_ones + iou_zeros) / 2)

    def test_reverse_indeces(self):
        C0 = [0, 0, 0]
        C1 = [128, 0, 0]
        C2 = [0, 128, 0]

        onehot_labels = np.array([[
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ], [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ]])
        onehot_labels = np.transpose(onehot_labels, (1, 2, 0))
        actual_colors = reverse_indeces(onehot_labels)

        expected_colors = np.array([
            [C0, C0, C0, C1, C1, C1],
            [C0, C0, C0, C1, C1, C1],
            [C1, C1, C1, C2, C2, C2],
            [C1, C1, C1, C2, C2, C2],
        ])
        np.testing.assert_array_almost_equal(
            expected_colors, actual_colors)
