"""Training loss for iterative 3D trajectory estimates."""

import torch

from .utils import masked_mean


def sequence_loss(xyz_predictions, xyz_targets, valid_masks, gamma: float = 0.8):
    if not xyz_targets:
        raise ValueError("at least one target window is required")

    total_loss = 0.0
    if not (len(xyz_predictions) == len(xyz_targets) == len(valid_masks)):
        raise ValueError("prediction, target and validity window counts must match")
    for predictions, target, valid in zip(xyz_predictions, xyz_targets, valid_masks):
        if not predictions:
            raise ValueError("each target window requires at least one prediction")
        window_loss = 0.0
        prediction_count = len(predictions)
        for index, prediction in enumerate(predictions):
            weight = gamma ** (prediction_count - index - 1)
            error = torch.mean(torch.abs(prediction - target), dim=3)
            window_loss += weight * masked_mean(error, valid)
        total_loss += window_loss / prediction_count
    return total_loss / len(xyz_targets)
