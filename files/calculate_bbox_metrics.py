import numpy as np
import math
import SimpleITK as sitk



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

def threshold(pred, thresh=0.5):
    pred[pred<thresh] = 0
    pred[pred>=thresh] = 1
    return pred

def get_arr_from_nrrd(link_to_nrrd, thresh=True):
    '''
    Used for predictions.
    '''
    image = sitk.ReadImage(link_to_nrrd)
    spacing = image.GetSpacing()
    arr = sitk.GetArrayFromImage(image)
    if thresh:
        arr = threshold(arr)
    assert arr.min() == 0, "minimum value is not 0"
    assert arr.max() == 1, "minimum value is not 1"
    assert len(np.unique(arr)) == 2, "arr does not contain 2 unique values"
    return arr, spacing

def calculate_bbox_metrics(gt, path_to_nrrd):
    """
    Calculates the distance between the centers of the bounding boxes of the ground truth and precited label.
    Args:
        gt (numpy array): ground truth label.
        path_to_nrrd (string): Path to nrrd file containing the predicted label.
    Returns:
        Euclidan distance
    """

    # get arrays
    pred, pred_spacing = get_arr_from_nrrd(path_to_nrrd)
    assert gt.shape==pred.shape, "gt and pred do not have the same shape"

    gt_bbox_metrics = get_bbox_metrics(gt)
    pred_bbox_metrics = get_bbox_metrics(pred)

    # https://hlab.stanford.edu/brian/euclidean_distance_in.html
    # e2 = (APHWcell1 - APHWcell2)2 + (mTaucell1 - mTaucell2)2 + (eEPSCcell1 - eEPSCcell2)2

    Z_distance = (gt_bbox_metrics["Z"]["center"] -  pred_bbox_metrics["Z"]["center"]) * pred_spacing[2]

    Y_distance = (gt_bbox_metrics["Y"]["center"] -  pred_bbox_metrics["Y"]["center"]) * pred_spacing[1]

    X_distance = (gt_bbox_metrics["X"]["center"] -  pred_bbox_metrics["X"]["center"]) * pred_spacing[0]

    distance = math.sqrt(pow(Z_distance, 2) +
                         pow(Y_distance, 2) +
                         pow(X_distance, 2))

    return {
    "ground_truth_bbox_metrics": gt_bbox_metrics,
    "prediction_bbox_metrics": pred_bbox_metrics,
    "distance": distance
    }
