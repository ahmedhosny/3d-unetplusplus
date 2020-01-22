import numpy as np
import pandas as pd
import SimpleITK as sitk

def print_shape(obj, mode):
    print ("{} image shape :: {} \n{} label shape :: {}".format(
        mode, obj["images"].shape, mode, obj["labels"].shape))

def generate_data(start, end):
    # read dataframe
    df = pd.read_csv("/data/0_curation/rtog_final_curation_ctv.csv")
    images = []
    labels = []
    for pat in df["patid"].tolist()[start:end]:
        # read image and get array
        path_to_nrrd = "/data/7_image_interpolated_resized/rtog_{}_image_interpolated_resized_raw_xx.nrrd".format(pat)
        image = sitk.ReadImage(path_to_nrrd)
        arr_image = sitk.GetArrayFromImage(image)
        # read label and get array
        path_to_nrrd = "/data/10_lung_interpolated_resized/rtog_{}_lung_interpolated_resized_raw_xx.nrrd".format(pat)
        label = sitk.ReadImage(path_to_nrrd)
        arr_label = sitk.GetArrayFromImage(label)
        # filter out all blank slices
        for i in range(arr_image.shape[0]):
            # get that slice
            slice_image = arr_image[i]
            if np.unique(slice_image).size != 1:
                # remap to 0 to 1
                slice_image =  np.interp(slice_image,[-1024,3071],[0,1])
                # slim down slightly so dimensions work with network
                slice_image = slice_image[6:294,2:322]
                # connvert to RGB and append
                slice_image = np.repeat(slice_image.reshape(288, 320, 1), 3, axis=2)
                images.append(slice_image)
                # get label and append
                slice_label = arr_label[i]
                slice_label = slice_label[6:294,2:322]
                slice_label = slice_label.reshape(288, 320, 1)
                labels.append(slice_label)

    return {"images": np.array(images), "labels": np.array(labels)}


def get_data():
    data = {
        "train": generate_data(0,10), #300
        "tune": generate_data(300,310), #350
        # "test": generate_data(350,426)
    }
    print_shape(data["train"], "train")
    print_shape(data["tune"], "tune")
    # print_shape(data["test"], "test")
    return data
