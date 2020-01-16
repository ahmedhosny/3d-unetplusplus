import numpy as np

def print_shape(obj, mode):
    print ("{} image shape :: {} \n{} label shape :: {}".format(
        mode, obj["images"].shape, mode, obj["labels"].shape))

def generate_data(size):
    _IMAGE_CHANNELS = 3
    _IMAGE_SIZE_X = 256
    _IMAGE_SIZE_Y = 256
    return {
        "images": np.random.rand(size,_IMAGE_SIZE_X, _IMAGE_SIZE_Y, _IMAGE_CHANNELS), # 0 to 1
        "labels": np.random.randint(2, size=(size,_IMAGE_SIZE_X, _IMAGE_SIZE_Y, 1))
    }

def get_data():
    data = {
        "train": generate_data(10),
        "tune": generate_data(5),
        "test": generate_data(6)
    }
    print_shape(data["train"], "train")
    print_shape(data["tune"], "tune")
    print_shape(data["test"], "test")
    return data

data = get_data()
print (data["train"]["images"])
