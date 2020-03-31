import os
import glob
import numpy as np
import math
from time import time
#
from keras import backend as K
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ReduceLROnPlateau
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
K.set_image_dim_ordering('th')
#
from model_3d.model import isensee2017_model
from data import get_data
from generator import Generator
from model_callbacks import model_callbacks
from losses import precision_loss, dice_loss, tversky_loss, focal_tversky_loss, bce_loss, bce_dice_loss, wce_dice_loss
from utils import get_lr_metric

# LOCALIZATION or SEGMENTATION
TASK = "SEGMENTATION"

if TASK == "LOCALIZATION":
    IMAGE_SHAPE = (80, 96, 96)
    BATCH_SIZE_PER_GPU = 18
    BLUR_LABEL = True
elif TASK == "SEGMENTATION":
    IMAGE_SHAPE = (64, 160, 160)
    BATCH_SIZE_PER_GPU = 8
    BLUR_LABEL = False

N_GPUS = 4
BATCH_SIZE = BATCH_SIZE_PER_GPU * N_GPUS
EPOCHS = 300
LABELS = [1]
N_LABELS = len(LABELS)
N_BASE_FILTERS = 16
INPUT_SHAPE = tuple([1] + list(IMAGE_SHAPE))
# AUGMENTATION
AUGMENT = True
ELASTIC = True # only matters if AUGMENT == True
SHUFFLE = True

# get data
data = get_data("train_tune", IMAGE_SHAPE, TASK)


RUN = 112
NAME = "focal-tversky-loss-0.0005-augment-rt-maastro"# loss.__name__.replace("_", "-") + "-0.0005"
MODEL = "/output/111_focal-tversky-loss-0.0005-augment-maastro/111.h5"
# original = 5e-4
INITIAL_LR = 0.0005

print("train run # {}".format(RUN))

# create folder
dir_name = "/output/{}_{}".format(RUN, NAME)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
    print("directory {} created".format(dir_name))

STEPS_PER_EPOCH = math.floor(
len(data["train"]["images"]) / BATCH_SIZE)
print("steps_per_epoch :: ", STEPS_PER_EPOCH)

# model
if MODEL == "":
    original_model = isensee2017_model(input_shape=INPUT_SHAPE, n_labels=N_LABELS, n_base_filters=N_BASE_FILTERS)
else:
    original_model = load_model(MODEL, custom_objects={'InstanceNormalization': InstanceNormalization})
    print ("using pretrained model")
# print(original_model.summary(line_length=150))

# parallel model
parallel_model = multi_gpu_model(original_model, gpus=N_GPUS)
optimizer = Adam(lr=INITIAL_LR)
lr_metric = get_lr_metric(optimizer)
parallel_model.compile(optimizer=optimizer, loss=focal_tversky_loss, metrics=[lr_metric])

# callbacks
cbk = model_callbacks(original_model, RUN, dir_name)
csv_logger = CSVLogger(dir_name + '/{}.csv'.format(RUN), append=True, separator=',')
# https://stackoverflow.com/questions/39517431/should-we-do-learning-rate-decay-for-adam-optimizer
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
#                               patience=12, min_lr=0)

gen = Generator(data["train"]["images"],
                data["train"]["labels"],
                BATCH_SIZE,
                IMAGE_SHAPE,
                BLUR_LABEL,
                AUGMENT,
                ELASTIC,
                SHUFFLE)

# fit
parallel_model.fit_generator(
    generator=gen,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=(data["tune"]["images"], data["tune"]["labels"]),
    callbacks=[cbk, csv_logger], #reduce_lr
    shuffle = True, # shuffles order of batches, but not batch content (already shuffled)
    max_queue_size=STEPS_PER_EPOCH*3,
    workers=STEPS_PER_EPOCH*2,
    use_multiprocessing=True
)
