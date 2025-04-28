"""
This module contains functions for calculating common similarity metrics between two 3D volumes: Dice Similarity Coefficient (Dice3d) and Jaccard Similarity Coefficient (Jaccard3d). These functions are designed to compare binary 3D masks, treating 0 as background and any non-zero value as part of the structure.

Dice3d(a, b): Computes the Dice Similarity Coefficient, which is a measure of the overlap between two binary 3D volumes. A value of 1.0 indicates perfect overlap, while a value of 0 indicates no overlap. It is often used in medical imaging for evaluating segmentation results.

Jaccard3d(a, b): Computes the Jaccard Similarity Coefficient, which is another metric to measure the similarity between two binary 3D volumes. A value of 1.0 indicates perfect similarity (complete overlap), and a value of 0 indicates no similarity.

Both functions handle inputs of 3D arrays (volumes) and ensure they are of the same shape and dimensionality before proceeding. If both volumes are empty (no structures), both functions return a similarity score of 1.0, considering it a perfect match.
"""

"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes.
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data.

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # Convert to binary mask: 0 = background, 1 = structure
    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)

    intersection = np.sum(a_bin * b_bin)
    sum_volumes = np.sum(a_bin) + np.sum(b_bin)

    if sum_volumes == 0:
        return 1.0  # If both are empty, consider perfect match

    dice_score = 2.0 * intersection / sum_volumes
    return dice_score


def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes.
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data.

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)

    intersection = np.sum(a_bin * b_bin)
    union = np.sum((a_bin + b_bin) > 0)

    if union == 0:
        return 1.0  # If both are empty, perfect match

    jaccard_score = intersection / union
    return jaccard_score

