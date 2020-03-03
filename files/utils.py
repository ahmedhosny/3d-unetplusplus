import numpy as np
import SimpleITK as sitk
import itertools

def threshold(pred, thresh=0.5):
    pred[pred<thresh] = 0
    pred[pred>=thresh] = 1
    return pred

def reduce_arr_dtype(arr, verbose=False):
    """ Change arr.dtype to a more memory-efficient dtype, without changing
    any element in arr. """

    if np.all(arr-np.asarray(arr,'uint8') == 0):
        if arr.dtype != 'uint8':
            if verbose:
                print('Converting '+str(arr.dtype)+' to uint8 np.ndarray')
            arr = np.asarray(arr, dtype='uint8')
    elif np.all(arr-np.asarray(arr,'int8') == 0):
        if arr.dtype != 'int8':
            if verbose:
                print('Converting '+str(arr.dtype)+' to int8 np.ndarray')
            arr = np.asarray(arr, dtype='int8')
    elif np.all(arr-np.asarray(arr,'uint16') == 0):
        if arr.dtype != 'uint16':
            if verbose:
                print('Converting '+str(arr.dtype)+' to uint16 np.ndarray')
            arr = np.asarray(arr, dtype='uint16')
    elif np.all(arr-np.asarray(arr,'int16') == 0):
        if arr.dtype != 'int16':
            if verbose:
                print('Converting '+str(arr.dtype)+' to int16 np.ndarray')
            arr = np.asarray(arr, dtype='int16')

    return arr

def generate_sitk_obj_from_npy_array(image_sitk_obj, pred_npy_array, output_dir=""):

    """
    image_sitk_obj: sitk object of input to model
    pred_npy_array: returned prediction from model - should be squeezed.
    NOTE: image_arr.shape will always be equal or larger than pred_npy_array.shape, but never smaller given that
    we are always cropping in data.py
    """
    # get array from sitk object
    image_arr = sitk.GetArrayFromImage(image_sitk_obj)
    # change pred_npy_array.shape to match image_arr.shape
    # getting amount of padding needed on each side
    z_diff = int((image_arr.shape[0] - pred_npy_array.shape[0]) / 2)
    y_diff = int((image_arr.shape[1] - pred_npy_array.shape[1]) / 2)
    x_diff = int((image_arr.shape[2] - pred_npy_array.shape[2]) / 2)
    # pad, defaults to 0
    pred_npy_array = np.pad(pred_npy_array, ((z_diff, z_diff), (y_diff, y_diff), (x_diff, x_diff)), 'constant')
    # save sitk obj
    new_sitk_object = sitk.GetImageFromArray(pred_npy_array)
    new_sitk_object.SetSpacing(image_sitk_obj.GetSpacing())
    new_sitk_object.SetOrigin(image_sitk_obj.GetOrigin())
    assert new_sitk_object.GetSize() == image_sitk_obj.GetSize(), "oops.. The shape of the returned array does not match your requested shape."
    if output_dir != "":
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_dir)
        writer.SetUseCompression(True)
        writer.Execute(new_sitk_object)
    return new_sitk_object


def combine_masks(mask_list):
    if len(mask_list) >= 2:
        for a, b in itertools.combinations(mask_list, 2):
            assert a.shape == b.shape, "masks do not have the same shape"
            assert a.max() == b.max(), "masks do not have the same max value (1)"
            assert a.min() == b.min(), "masks do not have the same min value (0)"

        # we will ignore the fact that 2 masks at the same voxel will overlap and
        # cause that vixel to have a value of 2.
        # The get_bbox function doesnt really care about that - it just evaluates
        # zero vs non-zero
        combined = np.zeros((mask_list[0].shape))
        for mask in mask_list:
            if mask is not None:
                combined = combined + mask
        return combined
    elif len(mask_list) == 1:
        return mask_list[0]
    else:
        print ("No masks provided!")

def get_bbox(mask_data):
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
    return Z_min, Z_max, Y_min, Y_max, X_min, X_max, Z_max-Z_min+1, Y_max-Y_min+1, X_max-X_min+1

def append_helper(result, key_list, obj):
    for key in key_list:
        result[key] = obj[key]
    return result
