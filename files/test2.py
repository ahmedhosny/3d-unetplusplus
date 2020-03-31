import os
from scipy import ndimage
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pprint import pprint
#
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
#
from data import get_data
from utils import generate_sitk_obj_from_npy_array, threshold, get_spacing, calculate_metrics, save_candidate_roi
from plot_images import plot_images

# always make sure to run test on one dataset only!

# segmentation model to run
RUN = "103"
NAME = "focal-tversky-loss-0.0005"
#
LOCALIZATION_RUN = "95_bce-dice-loss-0.001" # "92_tversky-loss-0.001"
SAVE_CSV = True
DATASET = "harvard-rt"

# metric-specific
HAUSDORFF_PERCENT = 95
OVERLAP_TOLERANCE = 5
SURFACE_DICE_TOLERANCE = 6

MASTER_FOLDER = "/mnt/aertslab/USERS/Ahmed/0_FINAL_SEGMENTAION_DATA"

# file to read from
_from = "/mnt/aertslab/USERS/Ahmed/0_FINAL_SEGMENTAION_DATA/{}/11_image_interpolated_roi_pr/{}".format(DATASET, LOCALIZATION_RUN)

# file to save to
dir_name = "/mnt/aertslab/USERS/Ahmed/0_FINAL_SEGMENTAION_DATA/{}/12_label_interpolated_pr/{}_{}_{}".format(DATASET, LOCALIZATION_RUN, RUN, NAME)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
    print("directory {} created".format(dir_name))

print("test run # {}".format(RUN))

# initiate vars
results = []
no_results = []

# load model
model = os.path.join("/output/{}_{}".format(RUN, NAME), "{}.h5".format(RUN))
original_model = load_model(model, custom_objects={'InstanceNormalization': InstanceNormalization})


def pad_helper(center, original, prediction, resized):
    diff = original - resized
    alpha = diff // 2
    beta = center - (prediction//2)
    pad_l = alpha + beta
    #
    gamma = resized - ( center + (prediction//2) )
    pad_r = alpha + gamma
    return pad_l, pad_r



for roi_image in os.listdir(_from):
    #
    dataset = roi_image.split("_")[0]
    patient_id = roi_image.split("_")[1]
    # if patient_id != "1530-TC330" or patient_id != "15-TC027":
    try:
        print (patient_id)
        Z = int(roi_image.split("_")[7])
        Y = int(roi_image.split("_")[8])
        X = int(roi_image.split("_")[9].split(".")[0])
        #
        image_obj = sitk.ReadImage( os.path.join(_from, roi_image) )
        image_arr = sitk.GetArrayFromImage(image_obj)
        print (image_obj.GetSpacing(), image_arr.shape) # size should match model
        # interpolate
        image_arr = np.interp(image_arr,[-1024,3071],[0,1])
        label_prediction = np.squeeze(original_model.predict(image_arr.reshape(1,1, *image_arr.shape)))
        label_prediction = threshold(label_prediction)


        ###############
        if label_prediction[label_prediction==1].sum() > 0:

            # get image_interpolated and label_interpolated
            path_to_image = "{}/{}/{}/{}_{}_{}".format(MASTER_FOLDER, dataset, "3_image_interpolated", dataset, patient_id, "image_interpolated_raw_raw_xx.nrrd")
            image_obj_org = sitk.ReadImage(path_to_image)
            image_arr_org = sitk.GetArrayFromImage(sitk.ReadImage(path_to_image))
            print (image_arr_org.shape)
            #
            path_to_label = "{}/{}/{}/{}_{}_{}".format(MASTER_FOLDER, dataset, "4_label_interpolated", dataset, patient_id, "label_interpolated_raw_raw_xx.nrrd")
            label_arr_org = sitk.GetArrayFromImage(sitk.ReadImage(path_to_label))
            print (label_arr_org.shape)


            # enlarge prediction and save as nrrd in _to

            Z_diff = image_arr_org.shape[0] - label_prediction.shape[0]
            print (image_arr_org.shape[0], label_prediction.shape[0])
            print(Z_diff, Z)
            if Z_diff >= 0:
                pad_l, pad_r = pad_helper(Z, image_arr_org.shape[0], label_prediction.shape[0], 168)
                label_prediction = np.pad(label_prediction,((pad_l, pad_r), (0, 0),   (0, 0)), 'constant', constant_values=0)
                print("more")
            else:
                alpha = abs(Z_diff // 2)
                label_prediction = label_prediction[alpha:alpha+image_arr_org.shape[0],:,:]
                print("less")

            print (label_prediction.shape)


            Y_diff = image_arr_org.shape[1] - label_prediction.shape[1]
            print (image_arr_org.shape[1], label_prediction.shape[1])
            print(Y_diff)
            if Y_diff >= 0:
                pad_l, pad_r = pad_helper(Y, image_arr_org.shape[1], label_prediction.shape[1], 324)
                label_prediction = np.pad(label_prediction,((0,0), (pad_l, pad_r),   (0, 0)), 'constant', constant_values=0)
                print("more")
            else:
                alpha = abs(Y_diff // 2)
                label_prediction = label_prediction[:,alpha:alpha+image_arr_org.shape[1],:]
                print("less")

            print (label_prediction.shape)


            X_diff = image_arr_org.shape[2] - label_prediction.shape[2]
            print (image_arr_org.shape[2], label_prediction.shape[2])
            print(X_diff)
            if X_diff >= 0:
                pad_l, pad_r = pad_helper(X, image_arr_org.shape[2], label_prediction.shape[2], 324)
                label_prediction = np.pad(label_prediction,((0,0), (0, 0), (pad_l, pad_r)), 'constant', constant_values=0)
                print("more")
            else:
                alpha = abs(X_diff // 2)
                label_prediction = label_prediction[:,:,alpha:alpha+image_arr_org.shape[2]]
                print("less")

            print (label_prediction.shape)


            # check size
            if label_arr_org.shape[0] - label_prediction.shape[0] == 1:
                label_prediction = np.pad(label_prediction,((1, 0), (0, 0),   (0, 0)), 'constant', constant_values=0)

            if label_arr_org.shape[1] - label_prediction.shape[1] == 1:
                label_prediction = np.pad(label_prediction,((0, 0), (1, 0),   (0, 0)), 'constant', constant_values=0)

            if label_arr_org.shape[2] - label_prediction.shape[2] == 1:
                label_prediction = np.pad(label_prediction,((0, 0), (0, 0),   (1, 0)), 'constant', constant_values=0)

            pred_sitk_obj = sitk.GetImageFromArray(label_prediction)
            pred_sitk_obj.SetSpacing(image_obj_org.GetSpacing())
            pred_sitk_obj.SetOrigin(image_obj_org.GetOrigin())

            writer = sitk.ImageFileWriter()
            writer.SetFileName("{}/{}_{}_{}.nrrd".format(dir_name, dataset, patient_id, "label_interpolated_raw_raw_pr"))
            writer.SetUseCompression(True)
            writer.Execute(pred_sitk_obj)

            # # get arrays from data
            # image_arr_org = sitk.GetArrayFromImage(image_sitk_obj)
            # label_arr_org = sitk.GetArrayFromImage(label_sitk_obj)
            # get arrays from prediction
            pred_arr_org = sitk.GetArrayFromImage(pred_sitk_obj)
            spacing = get_spacing(image_obj_org)

            # metrics
            result, dice, bbox_metrics = calculate_metrics(patient_id, spacing, label_arr_org, pred_arr_org, HAUSDORFF_PERCENT, OVERLAP_TOLERANCE, SURFACE_DICE_TOLERANCE)
            # append
            results.append(result)

            # plot 5x3 views
            plot_images(dataset,
                        patient_id,
                        image_arr_org,
                        label_arr_org,
                        pred_arr_org,
                        dir_name,
                        True,
                        bbox_metrics,
                        dice)
            print ("{} done. dice :: {}".format(patient_id, result["dice"]))

        else:
            no_results.append(patient_id)
            result = {}
            result["patient_id"] = patient_id
            result["precision"] = 0
            result["recall"] = 0
            result["jaccard"] = 0
            result["dice"] = 0
            result["segmentation_score"] = 0
            result["x_distance"] = 0
            result["y_distance"] = 0
            result["z_distance"] = 0
            result["distance"] = 0
            result["average_surface_distance_gt_to_pr"] = 0
            result["average_surface_distance_pr_to_gt"] = 0
            result["robust_hausdorff"] = 0
            result["overlap_fraction_gt_with_pr"] = 0
            result["overlap_fraction_pr_with_gt"] = 0
            result["surface_dice"] = 0
            for axes in ["X", "Y", "Z"]:
                for location in ["min", "center", "max", "length"]:
                    result["prediction_{}_{}".format(axes, location)] = 0
            results.append(result)

    except Exception as e:
        print(e)

# populate df
if SAVE_CSV:
    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.join(dir_name, "{}_{}.csv".format(RUN, NAME)))

# def path_helper(folder, file_tail):
#     return "{}/{}/{}/{}_{}_{}.nrrd".format(MASTER_FOLDER, dataset, folder, dataset, patient_id, file_tail)
#
# # LOCALIZATION or SEGMENTATION_GT or SEGMENTATION_PR
# TASK = "LOCALIZATION"
# MASTER_FOLDER = "/mnt/aertslab/USERS/Ahmed/0_FINAL_SEGMENTAION_DATA"
# IMAGE_INTERPOLATED_ROI_PR_FOLDER = "11_image_interpolated_roi_pr"
#
# if TASK == "LOCALIZATION":
#     IMAGE_SHAPE = (80, 96, 96) # model input
#     SAVE_CANDIDATES = True
#     IMAGE_INTERPOLATED_RESIZED_FOLDER = "5_image_interpolated_resized"
#     CROP_SHAPE = (64, 160, 160) # segmentation model size
#     data = get_data("test", IMAGE_SHAPE, TASK)
# elif TASK == "SEGMENTATION_GT":
#     IMAGE_SHAPE = (64, 160, 160)
#     SAVE_CANDIDATES = False
#     data = get_data("test", IMAGE_SHAPE, TASK)
    # TASK == "SEGMENTATION_PR":
    # DATASET = "harvard-rt"




# /mnt/aertslab/USERS/Ahmed/0_FINAL_SEGMENTAION_DATA/harvard-rt/11_image_interpolated_roi_pr/92_tversky-loss-0.001/harvard-rt_12-TC235_image_interpolated_roi_raw_pr_66_195_249.nrrd







#
#
#
#
#
#
#
#
#
#
#
# #############################################################################
# if TASK == "SEGMENTATION_PR":
#     image_interpolated_roi_pr_folder = "{}/{}/{}".format(MASTER_FOLDER, DATASET,IMAGE_INTERPOLATED_ROI_PR_FOLDER)
#     for file in image_interpolated_roi_pr_folder:
#         # read image
#         image = sitk.GetArrayFromImage(sitk.ReadImage(file))
#
#
# /mnt/aertslab/USERS/Ahmed/0_FINAL_SEGMENTAION_DATA/harvard-rt/11_image_interpolated_roi_pr
#
#
# if TASK == "LOCALIZATION" or TASK == "SEGMENTATION_GT":
#
#     for patient in data:
#         #### VARIABLES
#         patient_id = patient["patient_id"]
#         dataset = patient["dataset"]
#         # formatted (cropped & reshaped)
#         image = patient["image"]
#         # original size
#         image_sitk_obj = patient["image_sitk_obj"]
#         label_sitk_obj = patient["label_sitk_obj"]
#         spacing = get_spacing(image_sitk_obj)
#
#         #### PREDICT
#         label_prediction = np.squeeze(original_model.predict(image.reshape(1, *image.shape)))
#         label_prediction = threshold(label_prediction)
#
#         # if there are voxels predicted:
#         if label_prediction[label_prediction==1].sum() > 0:
#
#             # save model output as nrrd
#             # this will pad the prediction to match the size of the originals
#             # for localization, 80, 96, 96 => 84, 108, 108
#             # for segmentation, 64, 160, 160 => 76, 196, 196
#             pred_sitk_obj = generate_sitk_obj_from_npy_array(
#             image_sitk_obj,
#             label_prediction,
#             True,
#             os.path.join(dir_name, "{}_{}_prediction.nrrd".format(dataset, patient_id)))
#
#             # get arrays from data
#             image_arr_org = sitk.GetArrayFromImage(image_sitk_obj)
#             label_arr_org = sitk.GetArrayFromImage(label_sitk_obj)
#             # get arrays from prediction
#             pred_arr_org = sitk.GetArrayFromImage(pred_sitk_obj)
#
#             # metrics
#             result, dice, bbox_metrics = calculate_metrics(patient_id, spacing, label_arr_org, pred_arr_org, HAUSDORFF_PERCENT, OVERLAP_TOLERANCE, SURFACE_DICE_TOLERANCE)
#             # append
#             results.append(result)
#
#             # plot 5x3 views
#             plot_images(dataset,
#                         patient_id,
#                         image_arr_org,
#                         label_arr_org,
#                         pred_arr_org,
#                         dir_name,
#                         True,
#                         bbox_metrics,
#                         dice)
#             print ("{} done. dice :: {}".format(patient_id, result["dice"]))
#
#             # TODO: deal with empty predictions, bbox is None
#             # # extract ROI from image_interpolated_resized and label_interpolated_resized
#             if SAVE_CANDIDATES:
#                 save_candidate_roi(bbox_metrics,
#                    spacing,
#                    path_helper(IMAGE_INTERPOLATED_RESIZED_FOLDER, "image_interpolated_resized_raw_xx"),,
#                    CROP_SHAPE,
#                    path_helper(IMAGE_INTERPOLATED_ROI_PR_FOLDER, "image_interpolated_roi_raw_pr"))
#
#         else:
#             no_results.append(patient_id)
#             bbox_metrics = None
#
#
#
#     print ("no results :: ", no_results)
#
