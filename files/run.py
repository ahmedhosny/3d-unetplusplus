import json
from model import Xnet
from keras.models import model_from_json

model = Xnet(decoder_block_type='upsampling')
model.summary()

model_json = model.to_json()
with open("original_model_upsample.json", "w") as json_file:
    json_file.write(model_json)

# json_file = open('original_model_upsample.json', 'r')
# original_model_json = json_file.read()
# json_file.close()
# original_model = model_from_json(original_model_json)
#
# print (model.get_config() == original_model.get_config())
