import json
from model import Xnet
from keras.models import model_from_json
from data import get_data
import tensorflow as tf


# from keras import backend as K
# # tf.logging.set_verbosity(tf.logging.ERROR)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True


# data
data = get_data()

# model
model = Xnet(input_shape=(256, 256, 3), decoder_block_type='upsampling')
# model.summary()
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# with tf.device('/gpu:0'):
model.fit(x=data["train"]["images"], y=data["train"]["labels"], batch_size=2, epochs=10)




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
