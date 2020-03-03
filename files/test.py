import os
from scipy import ndimage
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pprint
#
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
#
from data import get_data
from utils import generate_sitk_obj_from_npy_array, threshold, append_helper
from metrics import precision, recall, jaccard, dice, segmentation_score, bbox_distance, surface_dice
from plot_images import plot_images
from calculate_bbox_metrics import calculate_bbox_metrics

runs = ["60_precision-loss-0.0001",
"61_precision-loss-0.0005",
"62_precision-loss-0.001",
"63_recall-loss-0.0001",
"64_recall-loss-0.0005",
"65_recall-loss-0.001",
"66_dice-loss-0.0001",
"67_dice-loss-0.0005",
"68_dice-loss-0.001",
"69_tversky-loss-0.0001",
"70_tversky-loss-0.0005",
"71_tversky-loss-0.001",
"72_focal-tversky-loss-0.0001",
"73_focal-tversky-loss-0.0005",
"74_focal-tversky-loss-0.001",
"75_wce-loss-0.0001",
"76_wce-loss-0.0005",
"77_wce-loss-0.001",
"78_balanced-cross-entropy-loss-0.0001",
"79_balanced-cross-entropy-loss-0.0005",
"80_balanced-cross-entropy-loss-0.001",
"81_focal-loss-0.0001",
"82_focal-loss-0.0005",
"83_focal-loss-0.001",
"84_bce-dice-loss-0.0001",
"85_bce-dice-loss-0.0005",
"86_bce-dice-loss-0.001",
"87_wce-dice-loss-0.0001",
"88_wce-dice-loss-0.0005",
"89_wce-dice-loss-0.001"]


for run in runs:

    cnfg = dict()
    cnfg["run"] = run.split("_")[0]
    cnfg["name"] = run.split("_")[1]
    cnfg["image_shape"] = (80, 96, 96)
    cnfg["save_prediction"] = True # must be true to plot images and calculate all metrics!
    cnfg["save_csv"] = True
    cnfg["plot_results"] = True
    cnfg["hausdorff_percent"] = 95
    cnfg["overlap_tolerance"] = 5
    cnfg["surface_dice_tolerance"] = 6

    print("test run # {}".format(cnfg["run"]))

    # folder should already exist from training run
    dir_name = "/output/{}_{}".format(cnfg["run"], cnfg["name"])


    # initiate vars
    results = []
    no_results = []

    # get data
    data = get_data("test", cnfg["image_shape"])

    # load model
    model = os.path.join(dir_name, "{}.h5".format(cnfg["run"]))
    original_model = load_model(model, custom_objects={'InstanceNormalization': InstanceNormalization})

    for patient in data:
        #### VARIABLES
        result = {}
        patient_id = patient["patient_id"]
        dataset = patient["dataset"]
        result["patient_id"] = patient_id
        # formatted
        image = patient["image"]
        label_ground_truth = np.squeeze(patient["label"])
        # original size
        image_sitk_obj = patient["image_sitk_obj"]
        label_sitk_obj = patient["label_sitk_obj"]
        spacing = image_sitk_obj.GetSpacing()

        #### PREDICT
        label_prediction = np.squeeze(original_model.predict(image.reshape(1, *image.shape)))
        label_prediction = threshold(label_prediction)
        # if there are voxels predicted:
        if label_prediction[label_prediction==1].sum() > 0:
            # save as nrrd
            if cnfg["save_prediction"]:
                pred_sitk_obj = generate_sitk_obj_from_npy_array(
                image_sitk_obj,
                label_prediction,
                os.path.join(dir_name, "{}_{}_prediction.nrrd".format(dataset, patient_id)))

            #### METRICS
            # calculate metrics based on arrays of original size
            image_arr_org = sitk.GetArrayFromImage(image_sitk_obj) # from data
            label_arr_org = sitk.GetArrayFromImage(label_sitk_obj) # from data
            pred_arr_org = sitk.GetArrayFromImage(pred_sitk_obj) # just saved abv

            #
            result["precision"] = precision(label_arr_org, pred_arr_org)
            result["recall"] = recall(label_arr_org, pred_arr_org)
            result["jaccard"] = jaccard(label_arr_org, pred_arr_org)
            result["dice"] = dice(label_arr_org, pred_arr_org)
            result["segmentation_score"] = segmentation_score(label_arr_org, pred_arr_org, spacing)
            bbox_metrics = calculate_bbox_metrics(label_arr_org,
                                                pred_arr_org,
                                                spacing)
            result = append_helper(result, ["x_distance", "y_distance", "z_distance", "distance"], bbox_metrics)
            surface_dice_metrics = surface_dice(label_arr_org,
                                            pred_arr_org,
                                            spacing,
                                            cnfg["hausdorff_percent"], cnfg["overlap_tolerance"],
                                            cnfg["surface_dice_tolerance"])
            result = append_helper(result, ["average_surface_distance_gt_to_pr", "average_surface_distance_pr_to_gt", "robust_hausdorff", "overlap_fraction_gt_with_pr", "overlap_fraction_pr_with_gt", "surface_dice"], surface_dice_metrics)
            # get bbox center (indices) of prediction for next segmentation step
            for axes in ["X", "Y", "Z"]:
                for location in ["min", "center", "max", "length"]:
                    result["prediction_{}_{}".format(axes, location)] = bbox_metrics["prediction_bbox_metrics"][axes][location]
            # append
            results.append(result)
            # pprint.pprint(result)
            # plot image
            plot_images(dataset,
                        patient_id,
                        image_arr_org,
                        label_arr_org,
                        pred_arr_org,
                        dir_name,
                        True,
                        bbox_metrics,
                        result["dice"])
            print ("{} done. dice :: {}".format(patient_id, result["dice"]))

            # extract ROI from image_interpolated and label_interpolated
        else:
            no_results.append(patient_id)

    print ("no results :: ", no_results)

    # populate df
    if cnfg["save_csv"]:
        df = pd.DataFrame.from_dict(results)
        df.to_csv(os.path.join(dir_name, "{}_{}.csv".format(cnfg["run"], cnfg["name"])))
