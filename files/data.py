import numpy as np
import pandas as pd
import SimpleITK as sitk

def print_shape(obj, mode):
    print ("{} image shape :: {} \n{} label shape :: {}".format(
        mode, obj["images"].shape, mode, obj["labels"].shape))
#
# def generate_data(size):
#     _IMAGE_CHANNELS = 3
#     _IMAGE_SIZE_X = 256
#     _IMAGE_SIZE_Y = 256
#     return {
#         "images": np.random.rand(size,_IMAGE_SIZE_X, _IMAGE_SIZE_Y, _IMAGE_CHANNELS), # 0 to 1
#         "labels": np.random.randint(2, size=(size,_IMAGE_SIZE_X, _IMAGE_SIZE_Y, 1))
#     }
#
# def get_data():
#     data = {
#         "train": generate_data(10),
#         "tune": generate_data(5),
#         "test": generate_data(6)
#     }
#     print_shape(data["train"], "train")
#     print_shape(data["tune"], "tune")
#     print_shape(data["test"], "test")
#     return data
#
# data = get_data()
# print (data["train"]["images"])

# https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another




def generate_data(start, end):

    df = pd.read_csv("/data/0_curation/rtog_final_curation_ctv.csv")

    images = []
    labels = []
    for pat in df["patid"].tolist()[start:end]:
        #
        path_to_nrrd = "/data/7_image_interpolated_resized/rtog_{}_image_interpolated_resized_raw_xx.nrrd".format(pat)
        image = sitk.ReadImage(path_to_nrrd)
        arr = sitk.GetArrayFromImage(image)
        arr =  np.interp(arr,[-1024,3071],[0,1])
        arr = arr[80,:,:]
        arr = arr[6:294,2:322]
        arr = np.repeat(arr.reshape(288, 320, 1), 3, axis=2)
        images.append(arr)
        #
        path_to_nrrd = "/data/10_lung_interpolated_resized/rtog_{}_lung_interpolated_resized_raw_xx.nrrd".format(pat)
        image = sitk.ReadImage(path_to_nrrd)
        arr = sitk.GetArrayFromImage(image)
        arr = arr[80,:,:]
        arr = arr[6:294,2:322]
        arr = arr.reshape(288, 320, 1)
        labels.append(arr)

    return {"images": np.array(images), "labels": np.array(labels)}


def get_data():
    data = {
        "train": generate_data(0,300) #,
        # "tune": generate_data(300,350),
        # "test": generate_data(350,426)
    }
    print_shape(data["train"], "train")
    # print_shape(data["tune"], "tune")
    # print_shape(data["test"], "test")
    return data


# get_data()

# patids_add = [
# "0617-292370",
# "0617-309598",
# "0617-404249",
# "0617-446756",
# "0617-503047",
# "0617-503777",
# "0617-543978",
# "0617-547163",
# "0617-579583",
# "0617-613919",
# "0617-619856",
# "0617-639047",
# "0617-657637",
# "0617-682513",
# "0617-792061"
# ]
#
# patids += patids_add

# interpolate image
# for pat in patids:
#     dataset = "rtog"
#     patient_id = pat
#     data_type = "image"
#     path_to_nrrd = "/mnt/aertslab/DATA/Lung/RTOG_0617/2_image_nrrd/rtog_{}_image_raw_raw_xx.nrrd".format(pat)
#     interpolation_type = "linear" #"nearest_neighbor"
#     spacing = (1,1,3)
#     return_type = "sitk_object"
#     output_dir = "/mnt/aertslab/DATA/Lung/RTOG_0617/0_processed/0_image_interpolated"
#     interpolated_nrrd = interpolate(dataset, patient_id, data_type, path_to_nrrd, interpolation_type, spacing, return_type, output_dir)
#     print (pat)
