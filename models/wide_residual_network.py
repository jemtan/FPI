#using code from https://github.com/asmith26/wide_resnets_keras.git
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import os

import logging
logging.basicConfig(level=logging.DEBUG)

import sys
#sys.stdout = sys.stderr
# Prevent reaching to maximum recursion depth in `theano.tensor.grad`
#sys.setrecursionlimit(2 ** 20)

import numpy as np
np.random.seed(2 ** 10)

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, BatchNormalization, Dropout, Input, Activation, Add, Dense, Flatten, UpSampling2D, Lambda, Concatenate
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import losses
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import tensorflow as tf
from functools import partial

USE_BIAS = False # no bias in conv
WEIGHT_INIT = "he_normal" 
WEIGHT_DECAY = 0.0005 
CHANNEL_AXIS = -1


# Wide residual network http://arxiv.org/abs/1605.07146
def _wide_basic(n_input_plane, n_output_plane, stride, dropout_probability=0.0, direction='down'):
    def f(net):
        # format of conv_params:
        #               [ [nb_col="kernel width", nb_row="kernel height",
        #               subsample="(stride_vertical,stride_horizontal)",
        #               border_mode="same" or "valid"] ]
        # B(3,3): orignal <<basic>> block
        if direction == 'up':
            conv_params = [ [3,3,(1,1),"same"],
                            [3,3,(1,1),"same"] ]        
        else:
            conv_params = [ [3,3,stride,"same"],
                            [3,3,(1,1),"same"] ]
 
        n_bottleneck_plane = n_output_plane

        # Residual block
        for i, v in enumerate(conv_params):
            if i == 0:
                if n_input_plane != n_output_plane:
                    net = BatchNormalization(axis=CHANNEL_AXIS)(net)
                    net = Activation("relu")(net)
                    convs = net
                else:
                    convs = BatchNormalization(axis=CHANNEL_AXIS)(net)
                    convs = Activation("relu")(convs)
                convs = Conv2D(n_bottleneck_plane, 
                               (v[0],v[1]),
                                strides=v[2],
                                padding=v[3],
                                kernel_initializer=WEIGHT_INIT,
                                kernel_regularizer=l2(WEIGHT_DECAY),
                                use_bias=USE_BIAS)(convs)
                if direction == 'up':
                    convs = UpSampling2D(stride)(convs)
            else:
                convs = BatchNormalization(axis=CHANNEL_AXIS)(convs)
                convs = Activation("relu")(convs)
                if dropout_probability > 0:
                   convs = Dropout(dropout_probability)(convs)
                convs = Conv2D(n_bottleneck_plane, 
                               (v[0],v[1]),
                                strides=v[2],
                                padding=v[3],
                                kernel_initializer=WEIGHT_INIT,
                                kernel_regularizer=l2(WEIGHT_DECAY),
                                use_bias=USE_BIAS)(convs)

        # Shortcut Conntection: identity function or 1x1 convolutional
        #  (depends on difference between input & output shape - this
        #   corresponds to whether we are using the first block in each
        #   group; see _layer() ).
        if n_input_plane != n_output_plane:
            shortcut_stride = 1 if direction == 'up' else stride
            shortcut = Conv2D(n_output_plane, 
                              (1,1),
                              strides=shortcut_stride,
                              padding="same",
                              kernel_initializer=WEIGHT_INIT,
                              kernel_regularizer=l2(WEIGHT_DECAY),
                              use_bias=USE_BIAS)(net)
            if direction == 'up':
                shortcut = UpSampling2D(stride)(shortcut)
        else:
            if stride == 1:
                shortcut = net
            elif direction == 'up':
                shortcut = UpSampling2D(stride)(net)
            else:
                shortcut = AveragePooling2D(stride)(net) 
            

        return Add()([convs, shortcut])
    
    return f


# "Stacking Residual Units on the same stage"
def _layer(block, n_input_plane, n_output_plane, count, stride, **kwargs):
    def f(net):
        net = block(n_input_plane, n_output_plane, stride, **kwargs)(net)
        for i in range(2,int(count+1)):
            net = block(n_output_plane, n_output_plane, stride=(1,1), **kwargs)(net)
        return net
    
    return f


def create_model():
    logging.debug("Creating model...")
    
    assert((depth - 4) % 6 == 0)
    n = (depth - 4) / 6
    
    inputs = Input(shape=input_shape)

    n_stages=[16, 16*k, 32*k, 64*k]


    conv1 = Conv2D(n_stages[0], 
                    (3, 3), 
                    strides=1,
                    padding="same",
                    kernel_initializer=weight_init,
                    kernel_regularizer=l2(weight_decay),
                    use_bias=use_bias)(inputs) # "One conv at the beginning (spatial size: 32x32)"

    # Add wide residual blocks
    block_fn = _wide_basic
    conv2 = _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1,1))(conv1)# "Stage 1 (spatial size: 32x32)"
    conv3 = _layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2,2))(conv2)# "Stage 2 (spatial size: 16x16)"
    conv4 = _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2,2))(conv3)# "Stage 3 (spatial size: 8x8)"

    batch_norm = BatchNormalization(axis=CHANNEL_AXIS)(conv4)
    relu = Activation("relu")(batch_norm)
                                            
    # Classifier block
    pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(relu)
    flatten = Flatten()(pool)
    predictions = Dense(units=nb_classes, kernel_initializer=weight_init, use_bias=use_bias,
                        kernel_regularizer=l2(weight_decay), activation="softmax")(flatten)

    model = Model(inputs=inputs, outputs=predictions)
    return model


def create_wide_residual_network_dec(input_shape,num_classes,depth,k=4,dropout_probability=0.0,final_activation=None):
    if final_activation is None:#unspecified
        final_activation = 'softmax' if num_classes > 1 \
            else 'sigmoid'
   
    assert((depth - 6) % 10 == 0), 'depth should be 10n+6'
    n = (depth - 6) // 10
    
    inputs = Input(shape=input_shape)

    n_stages=[16, 16*k, 32*k, 64*k, 64*k, 64*k]


    conv1 = Conv2D(n_stages[0], 
                    (3, 3), 
                    strides=1,
                    padding="same",
                    kernel_initializer=WEIGHT_INIT,
                    kernel_regularizer=l2(WEIGHT_DECAY),
                    use_bias=USE_BIAS)(inputs) # "One conv at the beginning (spatial size: 32x32)"

    # Add wide residual blocks
    block_fn = _wide_basic
    conv2 = _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1,1))(conv1)# "Stage 1 (spatial size: 32x32)"
    conv3 = _layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2,2))(conv2)# "Stage 2 (spatial size: 16x16)"
    conv4 = _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2,2))(conv3)# "Stage 3 (spatial size: 8x8)"
    conv5 = _layer(block_fn, n_input_plane=n_stages[3], n_output_plane=n_stages[4], count=n, stride=(2,2))(conv4)# "Stage 4 (spatial size: 4x4)"
    conv6 = _layer(block_fn, n_input_plane=n_stages[4], n_output_plane=n_stages[5], count=n, stride=(2,2))(conv5)# "Stage 5 (spatial size: 2x2)"


    block_fn = partial(_wide_basic,direction='up')#decoder blocks,keep n=1 
    upconv1 = _layer(block_fn, n_input_plane=n_stages[5], n_output_plane=n_stages[2], count=1, stride=(2,2))(conv6)# "Stage 1up (spatial size: 4x4)"
    upconv2 = _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[1], count=1, stride=(2,2))(upconv1)# "Stage 2up (spatial size: 8x8)"
    upconv3 = _layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[0], count=1, stride=(2,2))(upconv2)# "Stage 3up (spatial size: 16x16)"
    upconv4 = _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=num_classes, count=1, stride=(2,2))(upconv3)# "Stage 4up (spatial size: 32x32)"

    logit = Lambda(lambda x:x,name='logit')(upconv4)
    if final_activation == 'linear':
        outputs = logit
    else:
        outputs = Activation(final_activation)(logit)

    loss_f = 'categorical_crossentropy' if num_classes > 1 \
            else 'binary_crossentropy'

    return Model(inputs, outputs), loss_f

def create_wide_residual_network_decdeeper(input_shape,num_classes,depth,k=4,dropout_probability=0.0,final_activation=None):
    if final_activation is None:#unspecified
        final_activation = 'softmax' if num_classes > 1 \
            else 'sigmoid'
   
    assert((depth - 6) % 10 == 0), 'depth should be 10n+6'
    n = (depth - 6) // 10
    
    inputs = Input(shape=input_shape)

    n_stages=[16, 16*k, 32*k, 64*k, 64*k, 64*k, 64*k]


    conv1 = Conv2D(n_stages[0], 
                    (3, 3), 
                    strides=1,
                    padding="same",
                    kernel_initializer=WEIGHT_INIT,
                    kernel_regularizer=l2(WEIGHT_DECAY),
                    use_bias=USE_BIAS)(inputs) # "One conv at the beginning (spatial size: 32x32)"

    # Add wide residual blocks
    block_fn = _wide_basic
    conv2 = _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1,1))(conv1)# "Stage 1 (spatial size: 32x32)"
    conv3 = _layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2,2))(conv2)# "Stage 2 (spatial size: 16x16)"
    conv4 = _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2,2))(conv3)# "Stage 3 (spatial size: 8x8)"
    conv5 = _layer(block_fn, n_input_plane=n_stages[3], n_output_plane=n_stages[4], count=n, stride=(2,2))(conv4)# "Stage 4 (spatial size: 4x4)"
    conv6 = _layer(block_fn, n_input_plane=n_stages[4], n_output_plane=n_stages[5], count=n, stride=(2,2))(conv5)# "Stage 5 (spatial size: 2x2)"
    conv7 = _layer(block_fn, n_input_plane=n_stages[5], n_output_plane=n_stages[6], count=n, stride=(2,2))(conv5)# "Stage 6 (spatial size: 1x1)"


    block_fn = partial(_wide_basic,direction='up')#decoder blocks,keep n=1 
    upconv1 = _layer(block_fn, n_input_plane=n_stages[6], n_output_plane=n_stages[2], count=1, stride=(2,2))(conv7)# "Stage 1up (spatial size: 2x2)"
    upconv2 = _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[2], count=1, stride=(2,2))(upconv1)# "Stage 1up (spatial size: 4x4)"
    upconv3 = _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[1], count=1, stride=(2,2))(upconv2)# "Stage 2up (spatial size: 8x8)"
    upconv4 = _layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[0], count=1, stride=(2,2))(upconv3)# "Stage 3up (spatial size: 16x16)"
    upconv5 = _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=num_classes, count=1, stride=(2,2))(upconv4)# "Stage 4up (spatial size: 32x32)"

    logit = Lambda(lambda x:x,name='logit')(upconv5)
    if final_activation == 'linear':
        outputs = logit
    else:
        outputs = Activation(final_activation)(logit)

    loss_f = 'categorical_crossentropy' if num_classes > 1 \
            else 'binary_crossentropy'

    return Model(inputs, outputs), loss_f


def create_wide_residual_network_selfsup(input_shape,*args,**kwargs):
    if 'net_f' in kwargs:
        net_f = globals()[kwargs['net_f']]
        del kwargs['net_f']
    else:
        net_f = create_wide_residual_network_dec
    print('Building with network: ' + net_f.__name__+ '\n')

    net_ss,loss_f = net_f(input_shape,*args,**kwargs)
    
    optim = Adam(lr=0.001)    
    #optim = SGD(lr=0.001)    
    #optim = SGD(lr=0.1, momentum=0.9, nesterov=True)

    net_ss.compile(optim,[loss_f],['acc'])

    return net_ss

       
