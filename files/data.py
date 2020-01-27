import numpy as np
import pandas as pd
import SimpleITK as sitk
from collections import defaultdict


df = pd.read_csv("/data/0_curation/rtog_final_curation_ctv.csv")

label_paths = {
    "image": "/data/7_image_interpolated_resized/rtog_{}_image_interpolated_resized_raw_xx.nrrd",
    "lung": "/data/10_lung_interpolated_resized/rtog_{}_lung_interpolated_resized_raw_xx.nrrd",
    "heart": "/data/9_heart_interpolated_resized/rtog_{}_heart_interpolated_resized_raw_xx.nrrd",
    "cord": "/data/11_cord_interpolated_resized/rtog_{}_cord_interpolated_resized_raw_xx.nrrd",
    "esophagus": "/data/8_esophagus_interpolated_resized/rtog_{}_esophagus_interpolated_resized_raw_xx.nrrd",
    "ctv": "/data/12_ctv_interpolated_resized/rtog_{}_ctv_interpolated_resized_raw_xx.nrrd"
}

def print_shape(obj, mode):
    print ("{} image shape :: {} \n{} {} label shape :: {}\n{} {} label shape :: {}\n{} {} label shape :: {}\n{} {} label shape :: {}\n{} {} label shape :: {}".format(
        mode, obj["images"].shape,
        mode, "lung", obj["labels"]["lung"].shape,
        mode, "heart", obj["labels"]["heart"].shape,
        mode, "cord", obj["labels"]["cord"].shape,
        mode, "esophagus", obj["labels"]["esophagus"].shape,
        mode, "ctv", obj["labels"]["ctv"].shape)
        )

def get_arr(patient_id, label):
    path_to_nrrd = label_paths[label].format(patient_id)
    label = sitk.ReadImage(path_to_nrrd)
    return sitk.GetArrayFromImage(label)

def add_slice(label_lists, arr, index, label):
    slice_label = arr[index]
    slice_label = slice_label[6:294,2:322]
    slice_label = slice_label.reshape(288, 320, 1)
    label_lists[label].append(slice_label)

def generate_data(labels, start, end):
    images = []
    label_lists = defaultdict(list)
    # read dataframe
    for idx,pat in enumerate(df["patid"].tolist()[start:end]):
        # read image and get array
        path_to_nrrd = label_paths["image"].format(pat)
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
                # remap to 0 to 1
                slice_image =  np.interp(slice_image,[-1024,3071],[0,1])
                # slim down slightly so dimensions work with network
                slice_image = slice_image[6:294,2:322]
                # connvert to RGB and append
                slice_image = np.repeat(slice_image.reshape(288, 320, 1), 3, axis=2)
                images.append(slice_image)
                # get label and append
                for l in labels:
                    add_slice(label_lists, label_arrays[l], i, l)

        print ("{}_{}_{}/160".format(idx, pat, counter))

    return {"images": np.array(images),
            "labels": {
                "lung": np.array(label_lists["lung"]),
                "heart": np.array(label_lists["heart"]),
                "cord": np.array(label_lists["cord"]),
                "esophagus": np.array(label_lists["esophagus"]),
                "ctv": np.array(label_lists["ctv"])
            }
           }


def get_data(labels):
    data = {
        "train": generate_data(labels, 0, 330), # 330
        "tune": generate_data(labels, 330, 360), # 360
        # "test": generate_data(350,426)
    }
    print_shape(data["train"], "train")
    print_shape(data["tune"], "tune")
    # print_shape(data["test"], "test")
    return data
