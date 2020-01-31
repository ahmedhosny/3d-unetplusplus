import numpy as np
import SimpleITK as sitk

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

def generate_sitk_obj_from_npy_array(image_sitk_obj, image_arr, pred_npy_array, output_dir=""):
    """
    image_sitk_obj: sitk object of input to model
    image_arr: array of sitk object of input to model
    pred_npy_array: returned prediction from model
    NOTE: image_arr.shape will always be equal or larger than pred_npy_array.shape, but never smaller gievn that
    we are always cropping in data.py
    """
    # change pred_npy_array.shape to match image_arr.shape
    # getting amount of padding needed on each side
    z_diff = int((image_arr.shape[0] - pred_npy_array.shape[0]) / 2)
    y_diff = int((image_arr.shape[1] - pred_npy_array.shape[1]) / 2)
    x_diff = int((image_arr.shape[2] - pred_npy_array.shape[2]) / 2)
    # pad, defaults to 0
    pred_npy_array = np.pad(pred_npy_array, ((z_diff, z_diff), (y_diff, y_diff), (x_diff, x_diff), (0, 0)), 'constant')
    # remove channel
    pred_npy_array = pred_npy_array.reshape(*image_arr.shape)
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
    else:
        return new_sitk_object
