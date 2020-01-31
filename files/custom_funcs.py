from keras import backend as K
import numpy as np

# not tested
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def threshold(pred, thresh=0.5):
    pred[pred<thresh] = 0
    pred[pred>=thresh] = 1
    return pred

# https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137
def dice_coefficient_test(gt, pred, thresh=True):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    gt : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    pred : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    """
    if thresh:
        pred = threshold(pred)
    gt = np.asarray(gt).astype(np.bool)
    pred = np.asarray(pred).astype(np.bool)

    if gt.shape != pred.shape:
        raise ValueError("Shape mismatch: gt and pred must have the same shape.")

    im_sum = gt.sum() + pred.sum()
    if im_sum == 0:
        raise ValueError("Both arrays are empty.")

    # Compute Dice coefficient
    intersection = np.logical_and(gt, pred)

    return 2. * intersection.sum() / im_sum
