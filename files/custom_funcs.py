from keras import backend as K
import tensorflow as tf
import numpy as np

epsilon = 1e-5
smooth = 1

def get_bbox_metrics(tensor):
    Z = K.any(tensor, axis=(1,2))
    Y = K.any(tensor, axis=(0,2))
    X = K.any(tensor, axis=(0,1))
    #
    Z_min, Z_max = tf.where(Z)[0][0], tf.where(Z)[-1][0]
    Y_min, Y_max = tf.where(Y)[0][0], tf.where(Y)[-1][0]
    X_min, X_max = tf.where(X)[0][0], tf.where(X)[-1][0]
    #
    Z_center = tf.math.ceil(tf.subtract(Z_max,Z_min)/2 + tf.cast(Z_min, tf.float64))
    Y_center = tf.math.ceil(tf.subtract(Y_max,Y_min)/2 + tf.cast(Y_min, tf.float64))
    X_center = tf.math.ceil(tf.subtract(X_max,X_min)/2 + tf.cast(X_min, tf.float64))
    #
    return Z_center, Y_center, X_center

def bbox_distance_loss(y_true, y_pred):
    '''
        y_true should already only contain 0's and 1's
        spacing is currently hard-coded.
        To call in test phase:
        K.eval(bbox_distance_loss(y_true, y_pred))
    '''
    # y_pred = tf.cast(y_pred > 0.5, tf.float32)
    # #
    # spacing=(6,3,3)
    # #
    # Z_center_true, Y_center_true, X_center_true = get_bbox_metrics(y_true)
    # Z_center_pred, Y_center_pred, X_center_pred = get_bbox_metrics(y_pred)
    # #
    # Z_distance = (Z_center_true -  Z_center_pred) * spacing[0]
    # Y_distance = (Y_center_true -  Y_center_pred) * spacing[1]
    # X_distance = (X_center_true -  X_center_pred) * spacing[2]
    # #
    # total = K.pow(Z_distance, 2) + K.pow(Y_distance, 2) + K.pow(X_distance, 2)
    #
    # distance = tf.cast(K.sqrt(total), tf.float32)

    y_true_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos)
    return true_pos

def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred, axis=axis) + smooth/2)/(K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def focal_tversky_inverse(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), 1/gamma)


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



#
# # not tested
# def iou_coef(y_true, y_pred, smooth=1):
#   intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
#   union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
#   iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
#   return iou
#
#
# def dice_coefficient(y_true, y_pred, smooth=1.):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#
#
# def dice_coefficient_loss(y_true, y_pred):
#     return -dice_coefficient(y_true, y_pred)
#
#
