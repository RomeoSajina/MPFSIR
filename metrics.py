import numpy as np
import torch


def VIM(pred, GT, calc_per_frame=True, return_last=True):
    if calc_per_frame:
        pred = pred.reshape(-1, 39)
        GT = GT.reshape(-1, 39)
    errorPose = np.power(GT - pred, 2)
    errorPose = np.sum(errorPose, 1)
    errorPose = np.sqrt(errorPose)

    if return_last:
        errorPose = errorPose[-1]
    return errorPose


def keypoint_mpjpe(pred, gt):
    error = np.linalg.norm(pred - gt, ord=2, axis=-1).mean()
    return error
