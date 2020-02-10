from keras.models import load_model
from data import get_data
import tensorflow as tf
import numpy as np
from custom_funcs import dice_coefficient_test, threshold
import pandas as pd
from common_utils import generate_sitk_obj_from_npy_array
from keras.utils import multi_gpu_model
from scipy import ndimage

RUN = 22
GPUS = 8
BATCH_SIZE = int(160/GPUS)
print("test run # {}".format(RUN))

model = "/output/36_epoch_3_val-loss_-0.2641610581429618.h5"
# model = "/output/34_epoch_9_val-loss_-0.6120161452185066.h5"

prediction_dict = [
{"label": "overall", "output": []},
{"label": "ctv", "output": []},
{"label": "heart", "output": []},
{"label": "lung", "output": []},
{"label": "esophagus", "output": []},
{"label": "cord", "output": []}
]
#
# {"label": "heart",
# "model": "/output/21_label_heart_epoch_2_val-loss_-0.7868379491494388.h5",
# "output": []},
# {"label": "cord",
# "model": "/output/22_label_cord_epoch_2_val-loss_-0.7181856234372912.h5",
# "output": []},
# {"label": "esophagus",
# "model": "/output/23_label_esophagus_epoch_4_val-loss_-0.7761369094848717.h5",
# "output": []},
# {"label": "ctv",
# "model": "/output/24_label_ctv_epoch_3_val-loss_-0.7424983113985474.h5",
# "output": []}
# ]
patids = []

data = get_data(mode="test")
df = pd.DataFrame()

# with tf.device('/gpu:7'):
# for index, obj in enumerate(prediction_dict):

original_model = load_model(model)
parallel_model = multi_gpu_model(original_model, gpus=GPUS)

for index, patient in enumerate(data):
    patid = patient["patid"]
    if index == 0:
        patids.append(patid)
    image = patient["image"]
    ground_truth = patient["label"]
    prediction = parallel_model.predict(image, batch_size=BATCH_SIZE)
    np.save("/output/36_test_prediction_{}.npy".format(patid), prediction)
    print (prediction.shape)
#     #########
#     # post-processing
#     """
#     Will remove single dimension, threshold, fill holes, then add single dimension back.
#     """
#     original_shape = prediction.shape
#     prediction = np.squeeze(prediction)
#     prediction = threshold(prediction)
#     prediction = ndimage.binary_fill_holes(prediction).astype(int)
#     prediction = prediction.reshape(*original_shape)
#     # calculate dice and append to result
#     dice = dice_coefficient_test(ground_truth, prediction)
#     print("{} {}".format(patid, dice))
#     obj["output"].append(dice)
#     # save nrrd
#     generate_sitk_obj_from_npy_array(patient["image_sitk_obj"], patient["image_arr"], prediction, "/output/2d_multi_model/{}_{}_prediction.nrrd".format(patid, obj["label"]))
# df[obj["label"]] = obj["output"]
#
# df["patids"] = patids
# df.to_csv("/output/2d_multi_model/2d_multi_model.csv")
