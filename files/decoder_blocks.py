from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Concatenate


def handle_block_names(stage, cols):
    conv_name = 'decoder_stage{}-{}_conv'.format(stage, cols)
    bn_name = 'decoder_stage{}-{}_bn'.format(stage, cols)
    relu_name = 'decoder_stage{}-{}_relu'.format(stage, cols)
    up_name = 'decoder_stage{}-{}_upsample'.format(stage, cols)
    merge_name = 'merge_{}-{}'.format(stage, cols)
    return conv_name, bn_name, relu_name, up_name, merge_name

def ConvRelu(x, filters, kernel_size, use_batchnorm=False, conv_name='conv', bn_name='bn', relu_name='relu'):
    x = Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not(use_batchnorm))(x)
    if use_batchnorm:
        x = BatchNormalization(name=bn_name)(x)
    x = Activation('relu', name=relu_name)(x)
    return x

def Upsample2D_block(x, filters, stage, cols, kernel_size=(3,3), upsample_rate=(2,2),use_batchnorm=False, skip=None):
    # name cleanup
    conv_name, bn_name, relu_name, up_name, merge_name = handle_block_names(stage, cols)
    # upsample
    x = UpSampling2D(size=upsample_rate, name=up_name)(x)
    # concatenate
    if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
        if type(skip) is list:
            x = Concatenate(name=merge_name)([x] + skip)
        else:
            x = Concatenate(name=merge_name)([x, skip])
    # convolution x 2
    x = ConvRelu(x, filters, kernel_size, use_batchnorm=use_batchnorm,
                 conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')
    x = ConvRelu(x, filters, kernel_size, use_batchnorm=use_batchnorm,
                 conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')
    return x

def Transpose2D_block(x, filters, stage, cols, kernel_size=(3,3), upsample_rate=(2,2), transpose_kernel_size=(4,4), use_batchnorm=False, skip=None):
    # name cleanup
    conv_name, bn_name, relu_name, up_name, merge_name = handle_block_names(stage, cols)
    # deconvolution
    x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,padding='same', name=up_name, use_bias=not(use_batchnorm))(x)
    if use_batchnorm:
        x = BatchNormalization(name=bn_name+'1')(x)
    x = Activation('relu', name=relu_name+'1')(x)
    # concatenate
    if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
        if type(skip) is list:
            merge_list = []
            merge_list.append(x)
            for l in skip:
                merge_list.append(l)
            x = Concatenate(name=merge_name)(merge_list)
        else:
            x = Concatenate(name=merge_name)([x, skip])
    # convolution
    x = ConvRelu(x, filters, kernel_size, use_batchnorm=use_batchnorm,
                 conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')
    return x
