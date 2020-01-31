import numpy as np
import pandas as pd
import SimpleITK as sitk
from collections import defaultdict


df = pd.read_csv("/data/0_curation/rtog_final_curation_ctv.csv")
crop_3d = [0,160,6,294,2,322] # 160, 288, 320
crop_2d = [6,294,2,322] # 288, 320

paths = {
    "image": "/data/7_image_interpolated_resized/rtog_{}_image_interpolated_resized_raw_xx.nrrd",
    "lung": "/data/10_lung_interpolated_resized/rtog_{}_lung_interpolated_resized_raw_xx.nrrd",
    "heart": "/data/9_heart_interpolated_resized/rtog_{}_heart_interpolated_resized_raw_xx.nrrd",
    "cord": "/data/11_cord_interpolated_resized/rtog_{}_cord_interpolated_resized_raw_xx.nrrd",
    "esophagus": "/data/8_esophagus_interpolated_resized/rtog_{}_esophagus_interpolated_resized_raw_xx.nrrd",
    "ctv": "/data/12_ctv_interpolated_resized/rtog_{}_ctv_interpolated_resized_raw_xx.nrrd"
}

def print_shape(obj, mode):
    print ("{} image shape :: {} \n{} label shape :: {}".format(
        mode, obj["images"].shape,
        mode, obj["labels"].shape))

def get_arr(patient_id, label):
    """
    Reads a nrrd file and spits out a numpy array.
    """
    path_to_nrrd = paths[label].format(patient_id)
    label = sitk.ReadImage(path_to_nrrd)
    return sitk.GetArrayFromImage(label)

def format_2d_array(arr, crop_2d, mode="label"):
    """
    Crops, reshapes, and creates a RGB image (or binary label) of a 2D array.
    crop_2d [start_y, end_y, start_x, end_x]
    """
    arr = arr[crop_2d[0]:crop_2d[1], crop_2d[2]:crop_2d[3]]
    if mode == "image":
        arr = arr.reshape(*arr.shape, 1)
        arr = np.interp(arr,[-1024,3071],[0,1])
        arr = np.repeat(arr, 3, axis=2)
    return arr

def format_3d_array(arr, crop_3d, mode="label"):
    """
    Crops, reshapes, and creates a RGB image (or binary label) of a 3D array.
    crop_3d [start_z, end_z, start_y, end_y, start_x, end_x]
    """
    arr = arr[crop_3d[0]:crop_3d[1], crop_3d[2]:crop_3d[3], crop_3d[4]:crop_3d[5]]
    arr = arr.reshape(*arr.shape, 1)
    if mode == "image":
        arr = np.interp(arr,[-1024,3071],[0,1])
        arr = np.repeat(arr, 3, axis=3)
    return arr

def generate_train_tune_data(start, end):
    """
    For 2d slice-by-slice models. (ignores patients)
    """
    images = []
    labels = []
    label_order = ["lung", "heart", "cord", "esophagus", "ctv"]
    # read dataframe
    for idx,pat in enumerate(df["patid"].tolist()[start:end]):
        # read image and get array
        path_to_nrrd = paths["image"].format(pat)
        image = sitk.ReadImage(path_to_nrrd)
        arr_image = sitk.GetArrayFromImage(image)
        # read label and get array
        label_arrays = {
            "lung": get_arr(pat, "lung"),
            "heart": get_arr(pat, "heart"),
            "cord": get_arr(pat, "cord"),
            "esophagus": get_arr(pat, "esophagus"),
            "ctv": get_arr(pat, "ctv")
        }
        counter = 0
        # filter out all blank slices
        for i in range(arr_image.shape[0]):
            # get that slice
            slice_image = arr_image[i]
            if np.unique(slice_image).size != 1:
                counter += 1
                slice_image = format_2d_array(slice_image, crop_2d, mode="image")
                images.append(slice_image)
                combined_label = []
                # get 5 labels and append them
                for label in label_order:
                    slice_label = format_2d_array(label_arrays[label][i], crop_2d)
                    combined_label.append(slice_label)
                combined_label = np.stack(combined_label, axis=2)
                labels.append(combined_label)
        print ("{}_{}_{}/160".format(idx, pat, counter))

    return {
            "images": np.array(images),
            "labels": np.array(labels)
           }

# def generate_test_data(start, end):
#     """
#     For 2d slice-by-slice models (will bundle slices per patient).
#     Unlike generate_train_tune_data, this will return all labels.
#     """
#     test_data = []
#     # read dataframe
#     for idx,pat in enumerate(df["patid"].tolist()[start:end]):
#         image_sitk_obj = sitk.ReadImage(paths["image"].format(pat))
#         image_arr = sitk.GetArrayFromImage(image_sitk_obj)
#         test_data.append(
#          {"patid": pat,
#           "image_sitk_obj" : image_sitk_obj,
#           "image_arr": image_arr,
#           "image": format_3d_array(image_arr, crop_3d, mode="image"),
#           "labels" : {
#                 "lung": format_3d_array(get_arr(pat, "lung"), crop_3d),
#                 "heart": format_3d_array(get_arr(pat, "heart"), crop_3d),
#                 "cord": format_3d_array(get_arr(pat, "cord"), crop_3d),
#                 "esophagus": format_3d_array(get_arr(pat, "esophagus"), crop_3d),
#                 "ctv": format_3d_array(get_arr(pat, "ctv"), crop_3d)
#                 }
#           }
#         )
#         print ("{}_{}".format(idx, pat))
#     return test_data

def get_data(mode="train"):
    """
    For mode == "train":
        data["train"]["images"][i]
        data["train"]["labels"][i]
    For mode == "test":
        data[i]["patid"]
        data[i]["image_sitk_obj"] # sitk object
        data[i]["image_arr"]..[j] for slice # raw numpy array
        data[i]["image"]..[j] for slice # formatted ready for model
        data[i]["labels"]["lung"]..[j] for slice
    """
    if mode=="train":
        data = {
            "train": generate_train_tune_data(0, 330), # 330
            "tune": generate_train_tune_data(330, 360) # 360
        }
        print_shape(data["train"], "train")
        print_shape(data["tune"], "tune")
    elif mode=="test":
        data = generate_test_data(360, 426) #426
        print ("test cases :: {}\ntest image shape :: {}\ntest label shape :: {}".format(len(data), data[0]["image"].shape, data[0]["labels"]["lung"].shape))
    return data



















# print(data["train"]["images"].shape)
