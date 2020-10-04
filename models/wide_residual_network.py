from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras.initializers import _compute_fans
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import to_categorical

import tensorflow as tf

WEIGHT_DECAY = 0.5 * 0.0005
LARGE_NUM = 1e9

class SGDTorch(SGD):
    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m + g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p - lr * (self.momentum * v + g)
            else:
                new_p = p - lr * v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates


def _get_channels_axis():
    return -1 if K.image_data_format() == 'channels_last' else 1


def _conv_kernel_initializer(shape, dtype=None):
    fan_in, fan_out = _compute_fans(shape)
    stddev = np.sqrt(2. / fan_in)
    return K.random_normal(shape, 0., stddev, dtype)


def _dense_kernel_initializer(shape, dtype=None):
    fan_in, fan_out = _compute_fans(shape)
    stddev = 1. / np.sqrt(fan_in)
    return K.random_uniform(shape, -stddev, stddev, dtype)


def batch_norm():
    return BatchNormalization(axis=_get_channels_axis(), momentum=0.9, epsilon=1e-5,
                              beta_regularizer=l2(WEIGHT_DECAY), gamma_regularizer=l2(WEIGHT_DECAY))


def conv2d(output_channels, kernel_size, strides=1):
    return Convolution2D(output_channels, kernel_size, strides=strides, padding='same', use_bias=False,
                         kernel_initializer=_conv_kernel_initializer, kernel_regularizer=l2(WEIGHT_DECAY))


def dense(output_units):
    return Dense(output_units, kernel_initializer=_dense_kernel_initializer, kernel_regularizer=l2(WEIGHT_DECAY),
                 bias_regularizer=l2(WEIGHT_DECAY))


def _add_upsamp_block(x_in, out_channels, strides, dropout_rate=0.0):
    is_channels_equal = K.int_shape(x_in)[_get_channels_axis()] == out_channels

    bn1 = batch_norm()(x_in)
    bn1 = Activation('relu')(bn1)
    out = conv2d(out_channels, 3, 1)(bn1)
    out = UpSampling2D(strides)(out)#interpolation='bilinear'
    out = batch_norm()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout_rate)(out)
    out = conv2d(out_channels, 3, 1)(out)#non-strided conv
    if is_channels_equal and strides == 1:
        shortcut = x_in
    elif is_channels_equal and strides == 2:
        shortcut = UpSampling2D(strides)(x_in)#x_in or bn1?
    else:
        shortcut = conv2d(out_channels, 1, 1)(bn1)
        shortcut = UpSampling2D(strides)(shortcut)
    #shortcut = x_in if is_channels_equal else conv2d(out_channels, 1, strides)(bn1)
    out = add([out, shortcut])
    return out


def _add_basic_block(x_in, out_channels, strides, dropout_rate=0.0):
    is_channels_equal = K.int_shape(x_in)[_get_channels_axis()] == out_channels

    bn1 = batch_norm()(x_in)
    bn1 = Activation('relu')(bn1)
    out = conv2d(out_channels, 3, strides)(bn1)
    out = batch_norm()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout_rate)(out)
    out = conv2d(out_channels, 3, 1)(out)
    if is_channels_equal and strides == 1:
        shortcut = x_in
    elif is_channels_equal and strides == 2:
        shortcut = AveragePooling2D(strides)(x_in)#x_in or bn1?
    else:
        shortcut = conv2d(out_channels, 1, strides)(bn1)
    #shortcut = x_in if is_channels_equal else conv2d(out_channels, 1, strides)(bn1)
    out = add([out, shortcut])
    return out


def _add_conv_group(x_in, out_channels, n, strides, dropout_rate=0.0):
    out = _add_basic_block(x_in, out_channels, strides, dropout_rate)
    for _ in range(1, n):
        out = _add_basic_block(out, out_channels, 1, dropout_rate)
    return out


def _add_upsamp_group(x_in, out_channels, n, strides, dropout_rate=0.0):
    out = _add_upsamp_block(x_in, out_channels, strides, dropout_rate)
    for _ in range(1, n):
        out = _add_upsamp_block(out, out_channels, 1, dropout_rate)
    return out


def create_wide_residual_network_dec(input_shape, num_classes, depth, widen_factor=1, dropout_rate=0.0,
                                 final_activation='sigmoid',ret_intermediate=False):
    n_channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor, 64*widen_factor, 64*widen_factor]
    assert ((depth - 6) % 10 == 0), 'depth should be 10n+6'
    n = (depth - 6) // 10

    #2xn+1 conv per conv group
    #1 + 1*numConvGroups + 2n*numConvGroups
    inp = Input(shape=input_shape)
    conv1 = conv2d(n_channels[0], 3)(inp)  # one conv at the beginning (spatial size: 32x32) - 256
    conv2 = _add_conv_group(conv1, n_channels[1], n, 1, dropout_rate)  # Stage 1 (spatial size: 32x32) - 256
    conv3 = _add_conv_group(conv2, n_channels[2], n, 2, dropout_rate)  # Stage 2 (spatial size: 16x16) - 128
    conv4 = _add_conv_group(conv3, n_channels[3], n, 2, dropout_rate)  # Stage 3 (spatial size: 8x8) - 64
    conv5 = _add_conv_group(conv4, n_channels[4], n, 2, dropout_rate)  # Stage 3 (spatial size: 4x4) - 32
    conv6 = _add_conv_group(conv5, n_channels[5], n, 2, dropout_rate)  # Stage 3 (spatial size: 2x2) - 16


    #resNet decoder - for now, keep n = 1
    #change to nearest upsampling if need speedup
    upsamp1 = _add_upsamp_group(conv6, n_channels[2], 1, 2, dropout_rate)  # Stage 3 (spatial size: 4x4) - 32
    upsamp2 = _add_upsamp_group(upsamp1, n_channels[1], 1, 2, dropout_rate)  # Stage 3 (spatial size: 8x8) - 64
    upsamp3 = _add_upsamp_group(upsamp2, n_channels[0], 1, 2, dropout_rate)  # Stage 3 (spatial size: 16x16) - 128
    upsamp4 = _add_upsamp_group(upsamp3, num_classes, 1, 2, dropout_rate)  # Stage 3 (spatial size: 32x32) - 256

    #add another non-linear layer?
    logit = Lambda(lambda x:x,name='logit')(upsamp4) 
    out = Activation(final_activation)(logit)
    #f_x = Lambda(lambda x:x,name='f_x')(out)

    if ret_intermediate:
        return Model(inp,[out,logit])

    else:
        return Model(inp, out)


def create_wide_residual_network_decdeeper(input_shape, num_classes, depth, widen_factor=1, dropout_rate=0.0,
                                 final_activation='sigmoid',ret_intermediate=False):
    n_channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor, 64*widen_factor, 64*widen_factor, 64*widen_factor]
    assert ((depth - 7) % 12 == 0), 'depth should be 12n+7'
    n = (depth - 7) // 12

    #2xn+1 conv per conv group
    #1 + 1*numConvGroups + 2n*numConvGroups
    inp = Input(shape=input_shape)
    conv1 = conv2d(n_channels[0], 3)(inp)  # one conv at the beginning (spatial size: 32x32) - 256
    conv2 = _add_conv_group(conv1, n_channels[1], n, 1, dropout_rate)  # Stage 1 (spatial size: 32x32) - 256
    conv3 = _add_conv_group(conv2, n_channels[2], n, 2, dropout_rate)  # Stage 2 (spatial size: 16x16) - 128
    conv4 = _add_conv_group(conv3, n_channels[3], n, 2, dropout_rate)  # Stage 3 (spatial size: 8x8) - 64
    conv5 = _add_conv_group(conv4, n_channels[4], n, 2, dropout_rate)  # Stage 3 (spatial size: 4x4) - 32
    conv6 = _add_conv_group(conv5, n_channels[5], n, 2, dropout_rate)  # Stage 3 (spatial size: 2x2) - 16
    conv7 = _add_conv_group(conv6, n_channels[6], n, 2, dropout_rate)  # Stage 3 (spatial size: 1x1) - 8


    #resNet decoder - for now, keep n = 1
    #change to nearest upsampling if need speedup
    upsamp1 = _add_upsamp_group(conv7, n_channels[2], 1, 2, dropout_rate)  # Stage 3 (spatial size: 4x4) - 32
    upsamp2 = _add_upsamp_group(upsamp1, n_channels[2], 1, 2, dropout_rate)  # Stage 3 (spatial size: 4x4) - 32
    upsamp3 = _add_upsamp_group(upsamp2, n_channels[1], 1, 2, dropout_rate)  # Stage 3 (spatial size: 8x8) - 64
    upsamp4 = _add_upsamp_group(upsamp3, n_channels[0], 1, 2, dropout_rate)  # Stage 3 (spatial size: 16x16) - 128
    upsamp5 = _add_upsamp_group(upsamp4, num_classes, 1, 2, dropout_rate)  # Stage 3 (spatial size: 32x32) - 256

    #add another non-linear layer?
    logit = Lambda(lambda x:x,name='logit')(upsamp5) 
    out = Activation(final_activation)(logit)
    #f_x = Lambda(lambda x:x,name='f_x')(out)

    if ret_intermediate:
        return Model(inp,[out,logit])

    else:
        return Model(inp, out)



def create_wide_residual_network_selfsup(input_shape, *args, **kwargs):
    if 'net_f' in kwargs:
        net_f = globals()[kwargs['net_f']]#get func for network based on kwarg name
        del kwargs['net_f']#get rid of it before passing on kwargs
    else:
        net_f = create_wide_residual_network_dec
    print('Building with network: ' +net_f.__name__+ '\n')#for verification

    net = net_f(input_shape,*args,**kwargs)

    inp = Input(shape=input_shape)

    f_x = net(inp)#regular inference
   

    net_ss = Model(inp,f_x)
    
    return net_ss
 

 
 
