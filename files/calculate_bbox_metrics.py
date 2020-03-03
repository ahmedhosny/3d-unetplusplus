import numpy as np
import math
import SimpleITK as sitk
import utils

def get_bbox_metrics(mask_data):
    # crop maskData to only the 1's
    # http://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    # maskData order is z,y,x because we already rolled it
    Z = np.any(mask_data, axis=(1, 2))
    Y = np.any(mask_data, axis=(0, 2))
    X = np.any(mask_data, axis=(0, 1))
    #
    X_min, X_max = np.where(X)[0][[0, -1]]
    Y_min, Y_max = np.where(Y)[0][[0, -1]]
    Z_min, Z_max = np.where(Z)[0][[0, -1]]
    # 1 is added to account for the final slice also including the mask
    # geometric center
    return {
        "Z":{
        "min": Z_min,
        "max": Z_max,
        "length": Z_max-Z_min+1,
        "center": (Z_max-Z_min)/2 + Z_min
        },
        "Y":{
        "min": Y_min,
        "max": Y_max,
        "length": Y_max-Y_min+1,
        "center": (Y_max-Y_min)/2 + Y_min
        },
        "X":{
        "min": X_min,
        "max": X_max,
        "length": X_max-X_min+1,
        "center": (X_max-X_min)/2 + X_min
        }
    }

def get_arr_from_nrrd(link_to_nrrd, thresh=True):
    '''
    Used for predictions.
    '''
    image = sitk.ReadImage(link_to_nrrd)
    spacing = image.GetSpacing()
    arr = sitk.GetArrayFromImage(image)
    if thresh:
        arr = utils.threshold(arr)
    assert arr.min() == 0, "minimum value is not 0"
    assert arr.max() == 1, "minimum value is not 1"
    assert len(np.unique(arr)) == 2, "arr does not contain 2 unique values"
    return arr, spacing

def calculate_bbox_metrics(gt, pr, spacing):
    """
    Calculates the distance between the centers of the bounding boxes of the ground truth and precited label.
    Args:
        gt (numpy array): ground truth label.
        pr (numpy array): Predicted label.
        spacing (list): 3 values for z,y,x spacing (to calculate real values)
    Returns:
        Euclidan distance
    """

    assert gt.shape==pr.shape, "gt and pr do not have the same shape"

    gt_bbox_metrics = get_bbox_metrics(gt)
    pr_bbox_metrics = get_bbox_metrics(pr)

    # https://hlab.stanford.edu/brian/euclidean_distance_in.html
    # e2 = (APHWcell1 - APHWcell2)2 + (mTaucell1 - mTaucell2)2 + (eEPSCcell1 - eEPSCcell2)2

    Z_distance = (gt_bbox_metrics["Z"]["center"] -  pr_bbox_metrics["Z"]["center"]) * spacing[2]

    Y_distance = (gt_bbox_metrics["Y"]["center"] -  pr_bbox_metrics["Y"]["center"]) * spacing[1]

    X_distance = (gt_bbox_metrics["X"]["center"] -  pr_bbox_metrics["X"]["center"]) * spacing[0]

    distance = math.sqrt(pow(Z_distance, 2) +
                         pow(Y_distance, 2) +
                         pow(X_distance, 2))

    return {
        "ground_truth_bbox_metrics": gt_bbox_metrics,
        "prediction_bbox_metrics": pr_bbox_metrics,
        "x_distance": abs(X_distance),
        "y_distance": abs(Y_distance),
        "z_distance": abs(Z_distance),
        "distance": distance
    }
