from keras.models import load_model
from data import get_data
import tensorflow as tf
import numpy as np
from custom_funcs import dice_coefficient_test
import pandas as pd
from common_utils import reduce_arr_dtype

prediction_dict = [
{"label": "lung",
"model": "/output/20_label_lung_epoch_2_val-loss_-0.6884740935425411.h5",
"output": []},
{"label": "heart",
"model": "/output/21_label_heart_epoch_2_val-loss_-0.7868379491494388.h5",
"output": []},
{"label": "cord",
"model": "/output/22_label_cord_epoch_2_val-loss_-0.7181856234372912.h5",
"output": []},
{"label": "esophagus",
"model": "/output/23_label_esophagus_epoch_4_val-loss_-0.7761369094848717.h5",
"output": []},
{"label": "ctv",
"model": "/output/24_label_ctv_epoch_3_val-loss_-0.7424983113985474.h5",
"output": []}
]
patids = []


RUN = 9
print("test run # {}".format(RUN))

data = get_data(mode="test")
df = pd.DataFrame()

# with tf.device('/gpu:7'):
for index, obj in enumerate(prediction_dict):
    model = load_model(obj["model"])
    print(obj["model"])
    for patient in data:
        patid = patient["patid"]
        if index == 0:
            patids.append(patid)
        image = patient["image"]
        ground_truth = patient["labels"][obj["label"]]
        prediction = model.predict(image)
        np.save("/output/2d_multi_model/{}_{}.npy".format(patid, obj["label"]), reduce_arr_dtype(prediction, verbose=False))
        dice = dice_coefficient_test(ground_truth, prediction)
        print("{} {}".format(patid, dice))
        obj["output"].append(dice)
    df[obj["label"]] = obj["output"]

df["patids"] = patids
df.to_csv("/output/2d_multi_model/2d_multi_model.csv")
