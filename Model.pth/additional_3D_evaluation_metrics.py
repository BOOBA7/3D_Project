"""
This module provides functions for evaluating 3D segmentation performance using additional metrics beyond Dice and Jaccard:

Sensitivity3d: Computes the Sensitivity (True Positive Rate) of the segmentation.

Specificity3d: Computes the Specificity (True Negative Rate) of the segmentation.

LikelihoodRatio3d: Computes the Likelihood Ratio, combining sensitivity and specificity.

full_additional_report: Generates a report with sensitivity, specificity, likelihood ratio, and ground truth sum.

save_report_as_json: Saves the evaluation report as a JSON file.

These metrics provide a more comprehensive assessment of the model's segmentation performance.
"""

"""
Module for additional 3D evaluation metrics
"""
import numpy as np
import json
import os

def Sensitivity3d(y_pred, y_true):
    y_pred_bin = (y_pred > 0).astype(np.uint8)
    y_true_bin = (y_true > 0).astype(np.uint8)

    TP = np.sum((y_pred_bin == 1) & (y_true_bin == 1))
    FN = np.sum((y_pred_bin == 0) & (y_true_bin == 1))

    if (TP + FN) == 0:
        return 1.0
    return TP / (TP + FN)

def Specificity3d(y_pred, y_true):
    y_pred_bin = (y_pred > 0).astype(np.uint8)
    y_true_bin = (y_true > 0).astype(np.uint8)

    TN = np.sum((y_pred_bin == 0) & (y_true_bin == 0))
    FP = np.sum((y_pred_bin == 1) & (y_true_bin == 0))

    if (TN + FP) == 0:
        return 1.0
    return TN / (TN + FP)

def LikelihoodRatio3d(y_pred, y_true):
    sens = Sensitivity3d(y_pred, y_true)
    spec = Specificity3d(y_pred, y_true)

    if spec == 1.0:
        return float('inf')
    return sens / (1.0 - spec)

def full_additional_report(y_pred, y_true, sample_id=None):
    return {
        "Sample_ID": sample_id,
        "Sensitivity": Sensitivity3d(y_pred, y_true),
        "Specificity": Specificity3d(y_pred, y_true),
        "Likelihood_Ratio": LikelihoodRatio3d(y_pred, y_true),
        "Ground_Truth_Sum": int(np.sum((y_true > 0).astype(np.uint8)))
    }

def save_report_as_json(report, save_dir="evaluation_reports", epoch=None):
    os.makedirs(save_dir, exist_ok=True)
    filename = f"report_{report['Sample_ID']}_epoch{epoch}.json" if epoch is not None else f"report_{report['Sample_ID']}.json"
    with open(os.path.join(save_dir, filename), "w") as f:
        json.dump(report, f, indent=4)
