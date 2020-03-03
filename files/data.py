import numpy as np
import pandas as pd
import SimpleITK as sitk

df = pd.read_csv("/mnt/aertslab/USERS/Ahmed/0_FINAL_SEGMENTAION_DATA/data.csv")
print("data.csv shape :: ", df.shape)

data_split = {
    "train": df[(df["dataset"]=="maastro")][:330], 
    "tune" : df[(df["dataset"]=="maastro")][330:360],
    "test" : df[(df["dataset"]=="maastro")][360:],
}

version = "interpolated_resized_rescaled"

def print_shape(obj, mode):
    print ("{} image shape :: {} \n{} label shape :: {}".format(
        mode, obj["images"].shape,
        mode, obj["labels"].shape))

def get_arr(path_to_nrrd, mode, model_input_size):
    """
    Reads a nrrd file and spits out a numpy array.
    path_to_nrrd: path_to_nrrd
    type: image or label
    """
    sitk_image = sitk.ReadImage(path_to_nrrd)
    arr = sitk.GetArrayFromImage(sitk_image)
    if mode == "tune":
        arr = format_arr(arr, model_input_size)
    return arr

def crop_arr(arr, model_input_size):
    start_z = arr.shape[0]//2 - model_input_size[0]//2
    start_y = arr.shape[1]//2 - model_input_size[1]//2
    start_x = arr.shape[2]//2 - model_input_size[2]//2
    #
    arr = arr[start_z:start_z+model_input_size[0],
              start_y:start_y+model_input_size[1],
              start_x:start_x+model_input_size[2]]
    return arr

def format_arr(arr, model_input_size):
    """
    Used for test mode. Crops and reshapes array.
    Also remaps image values.
    """
    arr = crop_arr(arr, model_input_size)
    arr = arr.reshape(1, *arr.shape)
    return arr

def generate_train_tune_data(data_split, mode, model_input_size):
    """
    Used for training and tuning only.
    data_split: dictionary of train, tune, and test split.
    mode: train, tune, or test
    """
    images = []
    labels = []
    for idx, patient in data_split[mode].iterrows():
        # get arr
        arr_image = get_arr(patient["image_"+version], mode, model_input_size)
        arr_image = np.interp(arr_image,[-1024,3071],[0,1])
        arr_label = get_arr(patient["label_"+version], mode, model_input_size)
        # append to list
        images.append(arr_image)
        labels.append(arr_label)
        print ("{}_{}".format(idx, patient["patient_id"]))
    print("-------------")
    return {
            "images": np.array(images),
            "labels": np.array(labels)
           }

def generate_test_data(data_split, model_input_size):
    """
    Used for testing only. The image sitk object info is needed during test time. To avoid reading the image nrrd twice, it is read here.
    """
    test = []
    for idx, patient in data_split["test"].iterrows():
        # get image
        image_sitk_obj = sitk.ReadImage(patient["image_"+version])
        arr_image = sitk.GetArrayFromImage(image_sitk_obj)
        arr_image_interp = np.interp(arr_image,[-1024,3071],[0,1])
        # get label
        label_sitk_obj = sitk.ReadImage(patient["label_"+version])
        arr_label = sitk.GetArrayFromImage(image_sitk_obj)
        # append to list
        test.append(
         {"patient_id": patient["patient_id"],
          "dataset": patient["dataset"],
          "image_sitk_obj": image_sitk_obj,
          "image": format_arr(arr_image_interp, model_input_size),
          "label_sitk_obj": label_sitk_obj,
          "label": format_arr(arr_label, model_input_size),
          }
        )
        print ("{}_{}".format(idx, patient["patient_id"]))
    return test

def get_data(mode, model_input_size):
    """
    to call:
        model_input_size = (80, 96, 96)
        data_train_tune = get_data("train_tune", model_input_size)
        data_test = get_data("test", model_input_size)

    for access:
        data_train_tune["train"]["images"][patient, axial_slice, :, :]
        data_train_tune["train"]["labels"][patient, axial_slice, :, :]
        data_train_tune["tune"]["images"][patient, 0, axial_slice, :, :]
        data_train_tune["tune"]["labels"][patient, 0, axial_slice, :, :]
        data_test[patient]["image"][0, axial_slice, :, :]
        data_test[patient]["label"][0, axial_slice, :, :]

    intended behaviour:
        train image shape :: (L, 84, 108, 108)
        train label shape :: (L, 84, 108, 108)
        tune image shape :: (L, 1, 80, 96, 96)
        tune label shape :: (L, 1, 80, 96, 96)
        test cases :: 3
        test image shape :: (1, 80, 96, 96)
        test label shape :: (1, 80, 96, 96)

    for mode == "test", objects are returned:
        data[i]["patient_id"]
        data[i]["image_sitk_obj"] returns sitk object of image
        data[i]["image"] returns 3d 1-channel array
        data[i]["label_sitk_obj"] returns sitk object of label
        data[i]["label"] returns 3d 1-channel array
    """
    if mode=="train_tune":
        data = {
            "train": generate_train_tune_data(data_split, "train", model_input_size),
            "tune": generate_train_tune_data(data_split, "tune", model_input_size)
        }
        print_shape(data["train"], "train")
        print_shape(data["tune"], "tune")
    elif mode=="test":
        data = generate_test_data(data_split, model_input_size)
        print ("test cases :: {}\ntest image shape :: {}\ntest label shape :: {}".format(len(data), data[0]["image"].shape, data[0]["label"].shape))
    return data
