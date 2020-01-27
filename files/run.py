import json
from model import Xnet
from keras.models import model_from_json
from data import get_data
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from custom_funcs import dice_coefficient, dice_coefficient_loss
from keras.utils import multi_gpu_model
from model_callbacks import Cbk
from keras.preprocessing.image import ImageDataGenerator


# from keras import backend as K
# tf.logging.set_verbosity(tf.logging.ERROR)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

RUN = 16

BATCH_SIZE_PER_GPU = 4
EPOCHS = 5
GPUS = 8
LABELS = ["lung", "heart", "cord", "esophagus", "ctv"]
print("run # {}".format(RUN))
# data
data = get_data(LABELS)

datagen = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=20,
    height_shift_range=20,
    fill_mode='constant',
    cval=0.0)

datagen.fit(data["train"]["images"])

# model
original_model = Xnet(input_shape=(288, 320, 3), decoder_block_type='upsampling')
parallel_model = multi_gpu_model(original_model, gpus=GPUS)
# model.compile(optimizer='Adam',
#     loss='binary_crossentropy',
#     metrics=['binary_accuracy'])
parallel_model.compile(optimizer='Adam',
    loss=dice_coefficient_loss,
    metrics=[dice_coefficient])

# checkpointer = ModelCheckpoint(filepath='/output/lung_2d.hdf5',
#     monitor='val_loss',
#     verbose=1,
#     save_best_only=True)

for label in LABELS:
    print (label)
    cbk = Cbk(original_model, label, RUN)

    parallel_model.fit_generator(
        generator=datagen.flow(
            x=data["train"]["images"],
            y=data["train"]["labels"][label],
            batch_size=BATCH_SIZE_PER_GPU*GPUS,
            shuffle=True
            # save_to_dir='/output',
            # save_prefix='',
            # save_format='png'
        ),
        # If none, will use len(generator) as number of steps
        # len(data["train"]["images"]) / BATCH_SIZE_PER_GPU*GPUS
        steps_per_epoch=None,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=(data["tune"]["images"], data["tune"]["labels"][label]),
        callbacks=[cbk]
        )


#
#
#
#
#
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# y_train = np_utils.to_categorical(y_train, num_classes)
# y_test = np_utils.to_categorical(y_test, num_classes)
#
# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
#
# # compute quantities required for featurewise normalization
# # (std, mean, and principal components if ZCA whitening is applied)
# datagen.fit(x_train)
#
# # fits the model on batches with real-time data augmentation:
# model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
#                     steps_per_epoch=len(x_train) / 32, epochs=epochs)
#
# # here's a more "manual" example
# for e in range(epochs):
#     print('Epoch', e)
#     batches = 0
#     for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
#         model.fit(x_batch, y_batch)
#         batches += 1
#         if batches >= len(x_train) / 32:
#             # we need to break the loop by hand because
#             # the generator loops indefinitely
#             break
#








    # parallel_model.fit(x=data["train"]["images"],
    #         y=data["train"]["labels"][label],
    #         batch_size=BATCH_SIZE_PER_GPU*GPUS,
    #         epochs=EPOCHS,
    #         shuffle=True,
    #         validation_data=(data["tune"]["images"], data["tune"]["labels"][label]),
    #         callbacks=[cbk])



#
# predictions = model.predict_on_batch(data["test"]["images"])
# print(predictions.shape)
# print(predictions[0])
#
# predictions = model.predict(data["test"]["images"])
# print(predictions.shape)
# print(predictions[0])


# model_json = model.to_json()
# with open("original_model_upsample.json", "w") as json_file:
#     json_file.write(model_json)


# json_file = open('original_model_upsample.json', 'r')
# original_model_json = json_file.read()
# json_file.close()
# original_model = model_from_json(original_model_json)
#
# print (model.get_config() == original_model.get_config())
