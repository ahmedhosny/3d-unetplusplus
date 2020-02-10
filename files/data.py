import numpy as np
import pandas as pd
import SimpleITK as sitk
from collections import defaultdict

df = pd.read_csv("/data/0_curation/rtog_final_curation_ctv.csv")
crop_3d = [0,80,2,98,6,102] # 80, 100, 108

paths = {
    "image": "/data/14_image_interpolated_resized_rescaled/rtog_{}_image_interpolated_resized_rescaled_xx.nrrd",
    "label": "/data/23_oar_interpolated_resized_rescaled_ctv_binary/rtog_{}_oar_interpolated_resized_rescaled_xx.nrrd"
}

def print_shape(obj, mode):
    print ("{} image shape :: {} \n{} label shape :: {}".format(
        mode, obj["images"].shape,
        mode, obj["labels"].shape))

def get_arr(patient_id, key):
    """
    Reads a nrrd file and spits out a numpy array.
    """
    path_to_nrrd = paths[key].format(patient_id)
    image = sitk.ReadImage(path_to_nrrd)
    return sitk.GetArrayFromImage(image)


def format_3d_array(arr, crop_3d, mode="label"):
    """
    Crops, reshapes, and creates a RGB image (or binary label) of a 3D array.
    crop_3d [start_z, end_z, start_y, end_y, start_x, end_x]
    """
    if mode == "image":
        arr = np.interp(arr,[-1024,3071],[0,1])
        arr = arr[crop_3d[0]:crop_3d[1], crop_3d[2]:crop_3d[3], crop_3d[4]:crop_3d[5]]
        arr = arr.reshape(1, *arr.shape)
        return arr
    elif mode == "label":
        # change channel here
        arr = arr[crop_3d[0]:crop_3d[1], crop_3d[2]:crop_3d[3], crop_3d[4]:crop_3d[5], 1]
        arr = arr.reshape(1, *arr.shape)
        return arr

def generate_train_tune_data(start, end):
    """
    For 3d models. (ignores patients)
    """
    images = []
    labels = []
    # read dataframe
    for idx, pat in enumerate(df["patid"].tolist()[start:end]):
        # read image and label
        arr_image = get_arr(pat, "image")
        arr_label = get_arr(pat, "label")
        # format
        arr_image = format_3d_array(arr_image, crop_3d, mode="image")
        arr_label = format_3d_array(arr_label, crop_3d)
        # append to list
        images.append(arr_image)
        labels.append(arr_label)
        print ("{}_{}".format(idx, pat))

    return {
            "images": np.array(images),
            "labels": np.array(labels)
           }

def generate_test_data(start, end):
    """
    For 2d slice-by-slice models (will bundle slices per patient).
    Unlike generate_train_tune_data, this will return all labels.
    """
    test_data = []
    # read dataframe
    for idx, pat in enumerate(df["patid"].tolist()[start:end]):
        # read image and label
        image_sitk_obj = sitk.ReadImage(paths["image"].format(pat))
        image = sitk.GetArrayFromImage(image_sitk_obj)
        label = get_arr(pat, "label")
        test_data.append(
         {"patid": pat,
          "image_sitk_obj" : image_sitk_obj,
          "image_arr": image,
          "image": format_3d_array(image, crop_3d, mode="image"),
          "label" : format_3d_array(label, crop_3d)
          }
        )
        print ("{}_{}".format(idx, pat))
    return test_data

def get_data(mode="train"):
    """
    For mode == "train":
        data["train"]["images"][i] returns single 3d 1-channel array
        data["train"]["labels"][i] returns single 3d 1-channel array
        data["tune"]["images"][i] returns single 3d 1-channel array
        data["tune"]["labels"][i] returns single 3d 1-channel array
    For mode == "test":
        data[i]["patid"]
        data[i]["image_sitk_obj"] returns sitk object
        data[i]["image_arr"] returns raw numpy array
        data[i]["image"] returns single 3d 1-channel array
        data[i]["label"] returns single 3d 1-channel array
    """
    if mode=="train":
        data = {
            "train": generate_train_tune_data(0, 5), # 330
            "tune": generate_train_tune_data(330, 333) # 360
        }
        print_shape(data["train"], "train")
        print_shape(data["tune"], "tune")
    elif mode=="test":
        data = generate_test_data(360, 363) #426
        print ("test cases :: {}\ntest image shape :: {}\ntest label shape :: {}".format(len(data), data[0]["image"].shape, data[0]["label"].shape))
    return data
