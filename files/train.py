import os
import glob
import numpy as np
import math
#
from keras import backend as K
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
K.set_image_dim_ordering('th')
#
from model_3d.model import isensee2017_model
from data import get_data
from generator import generator
from model_callbacks import model_callbacks
from losses import precision_loss, recall_loss, dice_loss, tversky_loss, focal_tversky_loss, wce, balanced_cross_entropy, focal, bce_dice_loss, wce_dice_loss

cnfg = dict()
cnfg["image_shape"] = (80, 96, 96)
# max 20 for 80, 96, 96
cnfg["batch_size_per_gpu"] = 20
cnfg["n_gpus"] = 4
cnfg["batch_size"] = cnfg["batch_size_per_gpu"] * cnfg["n_gpus"]
cnfg["epochs"] = 50
cnfg["labels"] = [1]
cnfg["n_labels"] = len(cnfg["labels"])
cnfg["n_base_filters"] = 16
cnfg["input_shape"] = tuple([1] + list(cnfg["image_shape"]))
# AUGMENTATION
cnfg["augment"] = True
cnfg["rotation_angle_range"] = 8
cnfg["blur_multiplier"] = 2.0
cnfg["blur_random_range"] = 0.6 # 60% + or -
# get data
data = get_data("train_tune", cnfg["image_shape"])
# precision_loss, recall_loss, dice_loss, tversky_loss, focal_tversky_loss, wce(), balanced_cross_entropy(), focal(), bce_dice_loss,
all_losses = [wce_dice_loss]
all_lrs = [0.0001,  0.0005, 0.001]
first_run = 87

for loss in all_losses:
    for lr in all_lrs:

        cnfg["run"] = first_run
        cnfg["name"] = loss.__name__.replace("_", "-") + "-" + str(lr)
        # original = 5e-4
        cnfg["initial_learning_rate"] = lr

        print("train run # {}".format(cnfg["run"]))

        # create folder
        dir_name = "/output/{}_{}".format(cnfg["run"], cnfg["name"])
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print("directory {} created".format(dir_name))

        cnfg["steps_per_epoch"] = math.floor(
        len(data["train"]["images"]) / cnfg["batch_size"])
        print("steps_per_epoch :: ", cnfg["steps_per_epoch"])

        # model
        original_model = isensee2017_model(input_shape=cnfg["input_shape"], n_labels=cnfg["n_labels"],
        initial_learning_rate=cnfg["initial_learning_rate"],
        n_base_filters=cnfg["n_base_filters"])
        # print(original_model.summary(line_length=150))

        # parallel model
        parallel_model = multi_gpu_model(original_model, gpus=cnfg["n_gpus"])
        parallel_model.compile(optimizer=Adam(lr=cnfg["initial_learning_rate"]), loss=loss)

        # callbacks
        cbk = model_callbacks(original_model, cnfg["run"], dir_name)
        csv_logger = CSVLogger(dir_name + '/{}.csv'.format(cnfg["run"]), append=True, separator=',')

        gen = generator(data["train"]["images"],
        data["train"]["labels"],
        cnfg["batch_size"],
        cnfg["rotation_angle_range"],
        cnfg["image_shape"],
        cnfg["blur_multiplier"],
        cnfg["blur_random_range"],
        cnfg["augment"])

        # fit
        parallel_model.fit_generator(
            generator=gen,
            steps_per_epoch=cnfg["steps_per_epoch"],
            epochs=cnfg["epochs"],
            validation_data=(data["tune"]["images"], data["tune"]["labels"]),
            callbacks=[cbk, csv_logger],
            max_queue_size=3,
            workers=10,
            use_multiprocessing=True
        )

        first_run = cnfg["run"] + 1
