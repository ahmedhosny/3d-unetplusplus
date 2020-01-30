import numpy as np

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
