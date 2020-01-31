import json
import math
from model_2d.model import Xnet
from keras.models import model_from_json
from data import get_data
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from custom_funcs import dice_coefficient, dice_coefficient_loss
from keras.utils import multi_gpu_model
from model_callbacks import Cbk
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger

RUN = 27
AUG_SEED = 1
BATCH_SIZE_PER_GPU = 4
EPOCHS = 3
GPUS = 8
LABELS = ["heart", "cord", "esophagus", "ctv"]
print("train run # {}".format(RUN))

# data
data = get_data(LABELS, mode="train")

# data augmentation
datagen_args = dict(rotation_range=8,
    width_shift_range=20,
    height_shift_range=20,
    fill_mode='constant',
    cval=0.0)
image_datagen = ImageDataGenerator(**datagen_args)
image_generator = image_datagen.flow(
            x=data["train"]["images"],
            batch_size=BATCH_SIZE_PER_GPU*GPUS,
            shuffle=True,
            seed=AUG_SEED
            )
label_datagen = ImageDataGenerator(**datagen_args)

# model
original_model = Xnet(input_shape=(288, 320, 3), decoder_block_type='upsampling')
parallel_model = multi_gpu_model(original_model, gpus=GPUS)
parallel_model.compile(optimizer='Adam',
    loss=dice_coefficient_loss,
    metrics=[dice_coefficient])

for label in LABELS:
    print (label)
    # callbacks
    cbk = Cbk(original_model, label, RUN)
    csv_logger = CSVLogger('/output/{}_{}.csv'.format(RUN, label), append=True, separator=',')
    # label generator
    label_generator = label_datagen.flow(
                x=data["train"]["labels"][label],
                batch_size=BATCH_SIZE_PER_GPU*GPUS,
                shuffle=True,
                seed=AUG_SEED
                )
    train_generator = zip(image_generator, label_generator)

    parallel_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(image_generator),
        epochs=EPOCHS,
        shuffle=True,
        validation_data=(data["tune"]["images"], data["tune"]["labels"][label]),
        callbacks=[cbk, csv_logger]
        )
