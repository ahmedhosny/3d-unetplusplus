import os
from scipy import ndimage
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pprint import pprint
#
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
#
from data import get_data
from utils import generate_sitk_obj_from_npy_array, threshold, get_spacing, calculate_metrics, save_candidate_roi, multi_prediction
from plot_images import plot_images

def path_helper(folder, file_tail):
    return "{}/{}/{}/{}_{}_{}.nrrd".format(MASTER_FOLDER, dataset, folder, dataset, patient_id, file_tail)

# TODO: deal with empty predictions, for localization

# LOCALIZATION or SEGMENTATION
TASK = "SEGMENTATION"

RUN = "112"
NAME = "focal-tversky-loss-0.0005-augment-rt-maastro"
SAVE_CSV = True
print("{} test run # {}".format(TASK, RUN))

if TASK == "LOCALIZATION":
    IMAGE_SHAPE = (80, 96, 96)
    SAVE_CANDIDATES = True
    CROP_SHAPE = (64, 160, 160) # to save ROI images for segmentation model
    MASTER_FOLDER = "/mnt/aertslab/USERS/Ahmed/0_FINAL_SEGMENTAION_DATA"
    IMAGE_INTERPOLATED_RESIZED_FOLDER = "5_image_interpolated_resized"
    IMAGE_INTERPOLATED_ROI_PR_FOLDER = "11_image_interpolated_roi_pr/{}_{}".format(RUN, NAME)
elif TASK == "SEGMENTATION":
    IMAGE_SHAPE = (64, 160, 160)
    SAVE_CANDIDATES = False # always false


MULTI_PREDICTION = False
MODEL_TO_USE = "_final" # "" or "_final"

# metric-specific
HAUSDORFF_PERCENT = 95
OVERLAP_TOLERANCE = 5
SURFACE_DICE_TOLERANCE = 6

# get data
data = get_data("test", IMAGE_SHAPE, TASK, MULTI_PREDICTION)

# folder should already exist from training run
dir_name = "/output/{}_{}".format(RUN, NAME)

# initiate vars
results = []
no_results = []

# load model
model = os.path.join(dir_name, "{}{}.h5".format(RUN, MODEL_TO_USE))
original_model = load_model(model, custom_objects={'InstanceNormalization': InstanceNormalization})

for patient in data:
    #### VARIABLES
    patient_id = patient["patient_id"]
    dataset = patient["dataset"]
    # formatted (cropped & reshaped) if MULTI_PREDICTION = False
    # not cropped or reshaped if MULTI_PREDICTION = True
    image = patient["image"]
    # original size
    image_sitk_obj = patient["image_sitk_obj"]
    label_sitk_obj = patient["label_sitk_obj"]
    spacing = get_spacing(image_sitk_obj)

    #### PREDICT
    if MULTI_PREDICTION:
        label_prediction = multi_prediction(image, original_model, IMAGE_SHAPE)
        label_prediction = threshold(np.squeeze(label_prediction), 4.5)
    else:
        label_prediction = original_model.predict(image.reshape(1,*image.shape))
        label_prediction = threshold(np.squeeze(label_prediction)) # 0.5



    # if there are voxels predicted:
    if label_prediction[label_prediction==1].sum() > 0:

        # save model output as nrrd
        # this will pad the prediction to match the size of the originals
        # for localization, 80, 96, 96 => 84, 108, 108
        # for segmentation, 64, 160, 160 => 76, 196, 196
        pred_sitk_obj = generate_sitk_obj_from_npy_array(
        image_sitk_obj,
        label_prediction,
        True,
        os.path.join(dir_name, "{}_{}_prediction.nrrd".format(dataset, patient_id)))

        # get arrays from data
        image_arr_org = sitk.GetArrayFromImage(image_sitk_obj)
        label_arr_org = sitk.GetArrayFromImage(label_sitk_obj)
        # get arrays from prediction
        pred_arr_org = sitk.GetArrayFromImage(pred_sitk_obj)

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


        # extract ROI from image_interpolated_resized
        if SAVE_CANDIDATES:
            # create folder
            dir = "{}/{}/{}".format(MASTER_FOLDER, dataset, IMAGE_INTERPOLATED_ROI_PR_FOLDER)
            if not os.path.exists(dir):
                os.mkdir(dir)
                print("directory {} created".format(dir))
            # save candidates
            save_candidate_roi(bbox_metrics,
               spacing,
               path_helper(IMAGE_INTERPOLATED_RESIZED_FOLDER, "image_interpolated_resized_raw_xx"),
               CROP_SHAPE,
               "{}/{}_{}_{}".format(dir, dataset, patient_id, "image_interpolated_roi_raw_pr<>.nrrd"))

    else:
        no_results.append(patient_id)
        # temporary for segmentation task
        if TASK == "SEGMENTATION":
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


print ("no results :: ", no_results)

# populate df
if SAVE_CSV:
    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.join(dir_name, "{}_{}.csv".format(RUN, NAME)))
