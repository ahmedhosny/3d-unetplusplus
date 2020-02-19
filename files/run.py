import os
import glob
from keras import backend as K
import tensorflow as tf
# from unet3d.data import write_data_to_file, open_data_file
# from generator import get_training_and_validation_generators
from model_3d.model import isensee2017_model
from custom_funcs import weighted_dice_coefficient_loss, tversky_loss, focal_tversky, focal_tversky_inverse, bbox_distance_loss
import numpy as np
from keras.optimizers import Adam
from data import get_data
from keras.utils import multi_gpu_model
from model_callbacks import Cbk
from keras.callbacks import CSVLogger
K.set_image_dim_ordering('th')

# 45 - 0 - ctv
# 46 - 1 - heart
# 47 - 2 - lung
# 48 - 3 - esophagus
# 49 - 4 - cord



RUN = 54
AUG_SEED = 1
BATCH_SIZE_PER_GPU = 3
EPOCHS = 40
GPUS = 8
print("train run # {}".format(RUN))

# get data
data = get_data(mode="train")

config = dict()
# original 80, 100, 108 - adjusted to fit 80, 96, 96
config["image_shape"] = (80, 96, 96)
config["labels"] = [1]  # the label numbers on the input image
config["n_labels"] = len(config["labels"])
config["n_base_filters"] = 16
config["input_shape"] = tuple([1] + list(config["image_shape"]))
config["initial_learning_rate"] = 5e-4

# model
# with tf.device('/cpu:0'):
original_model = isensee2017_model(input_shape=config["input_shape"], n_labels=config["n_labels"],
initial_learning_rate=config["initial_learning_rate"],
n_base_filters=config["n_base_filters"])
# print(original_model.summary(line_length=150))

# parallel model
parallel_model = multi_gpu_model(original_model, gpus=GPUS)
parallel_model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=bbox_distance_loss)

# callbacks
cbk = Cbk(original_model, RUN)
csv_logger = CSVLogger('/output/{}.csv'.format(RUN), append=True, separator=',')

# fit
parallel_model.fit(
x=data["train"]["images"],
y=data["train"]["labels"],
batch_size=BATCH_SIZE_PER_GPU*GPUS,
 # steps_per_epoch=None,
 epochs=EPOCHS,
 shuffle=True,
 validation_data=(data["tune"]["images"], data["tune"]["labels"]),
 callbacks=[cbk, csv_logger]
 )










#
#
#
#
#
# import json
# import math
# from model_2d.model import Xnet
# from keras.models import model_from_json

# import tensorflow as tf
# from keras.callbacks import ModelCheckpoint
# from custom_funcs import dice_coefficient, dice_coefficient_loss
#

# from keras.preprocessing.image import ImageDataGenerator

#

#
# # data
# data = get_data(mode="train")
#
# # data augmentation
# datagen_args = dict(rotation_range=8,
#     width_shift_range=20,
#     height_shift_range=20,
#     fill_mode='constant',
#     cval=0.0)
# #
# image_datagen = ImageDataGenerator(**datagen_args)
# image_generator = image_datagen.flow(
#             x=data["train"]["images"],
#             batch_size=BATCH_SIZE_PER_GPU*GPUS,
#             shuffle=True,
#             seed=AUG_SEED
#             )
# label_datagen = ImageDataGenerator(**datagen_args)
# label_generator = label_datagen.flow(
#             x=data["train"]["labels"],
#             batch_size=BATCH_SIZE_PER_GPU*GPUS,
#             shuffle=True,
#             seed=AUG_SEED
#             )
# #
# train_generator = zip(image_generator, label_generator)
#
# # model
# original_model = Xnet(input_shape=(288, 320, 3), decoder_block_type='upsampling', classes=4)
# parallel_model = multi_gpu_model(original_model, gpus=GPUS)
# parallel_model.compile(optimizer='Adam',
#     loss=dice_coefficient_loss,
#     metrics=[dice_coefficient])
#

# parallel_model.fit_generator(
#     generator=train_generator,
#     steps_per_epoch=len(image_generator),
#     epochs=EPOCHS,
#     shuffle=True,
#     validation_data=(data["tune"]["images"], data["tune"]["labels"]),
#     callbacks=[cbk, csv_logger]
#     )
