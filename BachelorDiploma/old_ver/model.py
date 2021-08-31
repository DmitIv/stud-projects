import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras


def encoder1(inputs):
    # resnet encoder
    resn = resnet50.ResNet50(include_top=False, input_tensor=inputs, input_shape=(256, 256, 3))
    #     print(resn.summary())
    pool1 = resn.get_layer('activation_1').output
    pool2 = resn.get_layer('activation_10').output
    #     pool2 = ZeroPadding2D(((1, 0), (1, 0)),name='zero_padding2d_c2')(pool2)
    pool3 = resn.get_layer('activation_22').output
    pool4 = resn.get_layer('activation_40').output
    pool5 = resn.get_layer('activation_49').output

    return pool1, pool2, pool3, pool4, pool5


def unet_modify(pretrained_weights=None, input_size=(256, 256, 3)):
    inputs = Input(input_size)
    # inputs = Conv2D(3, 1, activation=None)(inputs)
    conv, conv1, conv2, conv3, conv4 = encoder1(inputs)
    # conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    # conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    # conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    # conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    # conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # conv5 = Conv2D(1024*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    # conv5 = Conv2D(1024*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    # drop5 = Dropout(0.5)(conv5)
    #
    # up6 = Conv2D(512*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    # merge6 = concatenate([conv3,up6], axis = 3)
    conv6 = Conv2D(512 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv6 = Conv2D(512 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256 * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv2, up7], axis=3)
    conv7 = Conv2D(256 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128 * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv1, up8], axis=3)
    conv8 = Conv2D(128 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64 * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv, up9], axis=3)
    conv9 = Conv2D(64 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    up10 = Conv2D(32 * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv9))
    merge10 = concatenate([inputs, up10], axis=3)
    conv10 = Conv2D(32 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
    conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv10)
    #     conv10 = ThresholdedReLU(theta=0.7)(conv10)
    #     conv9 = Conv2D(2, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv9)

    model = Model(input=inputs, output=conv10)
    #     iou_metric = adMetrics.MeanIoU(num_classes=2)

    # SGD_nesterov = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer = SGD_nesterov, loss = 'binary_crossentropy', metrics = ['accuracy'])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer = SGD_nesterov, loss = 'binary_crossentropy', metrics = ['accuracy'])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def segnet(input_shape=(256, 256, 1), n_classes=1):
    input = Input(shape=input_shape)

    # segnet中编码器结构
    zeropad1 = ZeroPadding2D((1, 1))(input)
    conv1 = Conv2D(64, 3, padding='valid', kernel_initializer='he_normal')(zeropad1)
    bn1 = BatchNormalization()(conv1)
    ac1 = Activation('relu')(bn1)
    pool1 = MaxPooling2D((2, 2))(ac1)

    zeropad2 = ZeroPadding2D((1, 1))(pool1)
    conv2 = Conv2D(128, 3, padding='valid', kernel_initializer='he_normal')(zeropad2)
    bn2 = BatchNormalization()(conv2)
    ac2 = Activation('relu')(bn2)
    pool2 = MaxPooling2D((2, 2))(ac2)

    zeropad3 = ZeroPadding2D((1, 1))(pool2)
    conv3 = Conv2D(256, 3, padding='valid', kernel_initializer='he_normal')(zeropad3)
    bn3 = BatchNormalization()(conv3)
    ac3 = Activation('relu')(bn3)
    pool3 = MaxPooling2D((2, 2))(ac3)

    zeropad4 = ZeroPadding2D((1, 1))(pool3)
    conv4 = Conv2D(512, 3, padding='valid', kernel_initializer='he_normal')(zeropad4)
    bn4 = BatchNormalization()(conv4)
    ac4 = Activation('relu')(bn4)

    # segnet中解码器结构
    zeropad5 = ZeroPadding2D((1, 1))(ac4)
    conv5 = Conv2D(512, 3, padding='valid', kernel_initializer='he_normal')(zeropad5)
    bn5 = BatchNormalization()(conv5)

    unpool6 = UpSampling2D(size=(2, 2))(bn5)
    zeropad6 = ZeroPadding2D((1, 1))(unpool6)
    conv6 = Conv2D(256, 3, padding='valid', kernel_initializer='he_normal')(zeropad6)
    bn6 = BatchNormalization()(conv6)

    unpool7 = UpSampling2D(size=(2, 2))(bn6)
    zeropad7 = ZeroPadding2D((1, 1))(unpool7)
    conv7 = Conv2D(128, 3, padding='valid', kernel_initializer='he_normal')(zeropad7)
    bn7 = BatchNormalization()(conv7)

    unpool8 = UpSampling2D(size=(2, 2))(bn7)
    zeropad8 = ZeroPadding2D((1, 1))(unpool8)
    conv8 = Conv2D(64, 3, padding='valid', kernel_initializer='he_normal')(zeropad8)
    bn8 = BatchNormalization()(conv8)

    conv9 = Conv2D(n_classes, 1, padding='valid', kernel_initializer='he_normal')(bn8)
    ac9 = Activation('sigmoid')(conv9)

    model = Model(input=input, output=ac9)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model
