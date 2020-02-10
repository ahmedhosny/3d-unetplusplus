import numpy as np
import pandas as pd
import SimpleITK as sitk
from collections import defaultdict

df = pd.read_csv("/data/0_curation/rtog_final_curation_ctv.csv")
crop_3d = [0,160,6,294,2,322] # 160, 288, 320
crop_2d = [6,294,2,322] # 288, 320

paths = {
    "image": "/data/7_image_interpolated_resized/rtog_{}_image_interpolated_resized_raw_xx.nrrd",
    "label": "/data/013_oar_interpolated_resized_ctv_binary/rtog_{}_oar_interpolated_resized_rescaled_xx.nrrd"
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

def format_image(arr, repeat_axis):
    arr = arr.reshape(*arr.shape, 1)
    arr = np.interp(arr,[-1024,3071],[0,1])
    arr = np.repeat(arr, 3, axis=repeat_axis)
    return arr

def format_2d_array(arr, crop_2d, mode="label"):
    """
    Crops, reshapes, and creates a RGB image (or binary label) of a 2D array.
    crop_2d [start_y, end_y, start_x, end_x]
    """
    if mode == "image":
        arr = arr[crop_2d[0]:crop_2d[1], crop_2d[2]:crop_2d[3]]
        return format_image(arr, 2)
    return arr[crop_2d[0]:crop_2d[1], crop_2d[2]:crop_2d[3], [0,1,3,4]] # : # skip lung i=2

def format_3d_array(arr, crop_3d, mode="label"):
    """
    Crops, reshapes, and creates a RGB image (or binary label) of a 3D array.
    crop_3d [start_z, end_z, start_y, end_y, start_x, end_x]
    """
    if mode == "image":
        arr = arr[crop_3d[0]:crop_3d[1], crop_3d[2]:crop_3d[3], crop_3d[4]:crop_3d[5]]
        return format_image(arr, 3)
    return arr[crop_3d[0]:crop_3d[1], crop_3d[2]:crop_3d[3], crop_3d[4]:crop_3d[5], [0,1,3,4]] # :

def generate_train_tune_data(start, end):
    """
    For 2d slice-by-slice models. (ignores patients)
    """
    images = []
    labels = []
    # read dataframe
    for idx, pat in enumerate(df["patid"].tolist()[start:end]):
        # read image and label
        arr_image = get_arr(pat, "image")
        arr_label = get_arr(pat, "label")
        counter = 0
        # filter out all blank slices
        for i in range(arr_image.shape[0]):
            # get that slice
            slice_image = arr_image[i]
            if np.unique(slice_image).size != 1:
                counter += 1
                # image
                slice_image = format_2d_array(slice_image, crop_2d, mode="image")
                images.append(slice_image)
                # label
                slice_label = arr_label[i]
                slice_label = format_2d_array(slice_label, crop_2d)
                labels.append(slice_label)
        print ("{}_{}_{}/160".format(idx, pat, counter))

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
        data["train"]["images"][i] returns single 2d 3-channel array
        data["train"]["labels"][i] returns single 2d 5-channel array
        data["tune"]["images"][i] returns single 2d 3-channel array
        data["tune"]["labels"][i] returns single 2d 5-channel array
    For mode == "test":
        data[i]["patid"]
        data[i]["image_sitk_obj"] returns sitk object
        data[i]["image_arr"] returns raw numpy array (..[j] for slice)
        data[i]["image"] returns single 3d 3-channel array per patient (..[j] for slice)
        data[i]["label"] returns single 3d 5-channel array per patient (..[j] for slice)
    """
    if mode=="train":
        data = {
            "train": generate_train_tune_data(0, 330), # 330
            "tune": generate_train_tune_data(330, 360) # 360
        }
        print_shape(data["train"], "train")
        print_shape(data["tune"], "tune")
    elif mode=="test":
        data = generate_test_data(360, 363) #426
        print ("test cases :: {}\ntest image shape :: {}\ntest label shape :: {}".format(len(data), data[0]["image"].shape, data[0]["label"].shape))
    return data
