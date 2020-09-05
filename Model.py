import random
import cv2
import numpy as np
from keras import Model, Sequential
from keras.applications import VGG16
from keras.layers import Activation, BatchNormalization, Convolution2D, GlobalAveragePooling2D, Dense, Dropout


class buildModel(object):
    def __init__(self,
                 ImageSize=96,
                 FinetuneVggLayerIndex=0,
                 PreTrainDir=''):
        self.ImageSize = ImageSize
        self.FinetuneVggLayerIndex = FinetuneVggLayerIndex
        self.PreTrainDir = PreTrainDir
        self.Model= self.buildVggNiNModel(self.ImageSize, self.FinetuneVggLayerIndex, self.PreTrainDir)
        self.Model.summary()

    def buildVggNiNModel(self,inputSize,FinetuneVggLayerIndex,preTrainDir):
        vgg = VGG16(weights='imagenet',include_top=False, input_shape=(inputSize, inputSize, 3))
        model_vgg = Sequential()
        for layer in vgg.layers[:11]:
            layer.trainable = False
            model_vgg.add(layer)

        if FinetuneVggLayerIndex != 0:
            for layer in model_vgg.layers[FinetuneVggLayerIndex:]:
                layer.trainable = True
        # -----------------------------------------------------------------------
        model = BatchNormalization()(model_vgg.output)
        model = Convolution2D(512, (3, 3))(model)
        model = BatchNormalization()(model)
        model = Activation(activation='relu')(model)
        model = Convolution2D(256, (1, 1), activation='relu')(model)
        model = Dropout(0.4)(model)
        model = Convolution2D(128, (1, 1), activation='relu')(model)
        model = Dropout(0.4)(model)
        model = GlobalAveragePooling2D()(model)
        model = Dense(7, activation='softmax')(model)
        # -----------------------------------------------------------
        # combine the 2 parts as one model
        model_vgg_nin = Model(inputs=model_vgg.input, outputs=model)
        if preTrainDir:
            # load a pre-trained weights file
            model_vgg_nin.load_weights(preTrainDir, by_name=True)
        else:
            print('no pretrained model')
        return model_vgg_nin