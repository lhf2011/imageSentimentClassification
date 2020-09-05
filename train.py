import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import argparse
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from Model import buildModel
from GetImage import getImage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('-s', '--ImageSize', type=int, default=96)
    parser.add_argument('-f', '--FinetuneVggLayerIndex', type=int, default=0)
    parser.add_argument('-pre','--PreTrainDir', type=str, default='')
    parser.add_argument('-d', '--SaveModelDir', type=bool,
                        default='model_vgg_3x3conv256_BN_1x1conv128_1x1conv64_GAP_softmax_1.h5')
    parser.add_argument('-lr', '--LearningRate', type=float, default=0.1)
    parser.add_argument('-r', '--RotationRange', type=float, default=0.)
    parser.add_argument('-z', '--ZoomRange', type=float, default=0.)
    parser.add_argument('-ws', '--WidthShiftRange', type=float, default=0.)
    parser.add_argument('-hs', '--HeightShiftRange', type=float, default=0.)
    parser.add_argument('-sh', '--ShearRange', type=float, default=0.)
    parser.add_argument('-b', '--BatchSize', type=np.uint8, default=0.)
    parser.add_argument('-e', '--Epochs', type=np.uint8, default=0.)
    parser.add_argument('-st', '--StepsPerEpoch', type=np.long, default=0.)
    args = parser.parse_args()

    preTrainModel=args.PreTrainDir
    finetuneVggLayerIndex=args.FinetuneVggLayerIndex
    saveModel= args.SaveModelDir
    learningRate= args.LearningRate
    rotationRange= args.RotationRange
    zoomRange= args.ZoomRange
    widthShiftRange = args.WidthShiftRange
    heightShiftRange = args.HeightShiftRange
    shearRange = args.ShearRange
    batchSize = args.BatchSize
    epochs = args.Epochs
    stepsPerEpoch = args.StepsPerEpoch

    # -----------------------------------------------------------------------------------------
    # you can set configs here, or comment them out and just get suggested parameters from here
    # -----------------------------------------------------------------------------------------
    # for model load and save
    preTrainModel='./preTrainedModel.h5'
    finetuneVggLayerIndex=-2
    saveModel='trainWellModel.h5'
    # for data augement
    rotationRange=40
    zoomRange=0.4
    widthShiftRange=0.2
    heightShiftRange=0.2
    shearRange=0.1
    batchSize=240
    # for train
    learningRate = 0.1
    epochs=2
    stepsPerEpoch=50
    # -----------------------------------------------------------------------------------------

    # build a model to train
    trainModel=buildModel(ImageSize=args.ImageSize,FinetuneVggLayerIndex=finetuneVggLayerIndex,PreTrainDir=preTrainModel)

    # get train data set, from the 1~18000 images in train.csv
    trainData= getImage(imageDir='./train.csv',imageSize=args.ImageSize,startIndex=1,endIndex=18000,histEqual=True)
    y_train, x_train = trainData.label,trainData.colorImageInTargetSize
    y_train = np_utils.to_categorical(y_train)

    # get validation data set, from the 18000~20000 images in train.csv
    validData = getImage(imageDir='./train.csv', imageSize=args.ImageSize, startIndex=18000, histEqual=True)
    y_validate, x_validate = validData.label,validData.colorImageInTargetSize
    y_validate = np_utils.to_categorical(y_validate)

    # data augement, to generate more training data
    datagen = ImageDataGenerator(rotation_range=rotationRange, zoom_range=zoomRange,
                                 width_shift_range=widthShiftRange, height_shift_range=heightShiftRange,
                                 shear_range=shearRange, horizontal_flip=True,
                                 fill_mode='nearest', dtype='uint8')
    train_generator = datagen.flow(x_train, y_train, batch_size=batchSize)

    # set the SGD optimizer
    sgd = SGD(lr=learningRate, decay=1e-6, momentum=0.9, nesterov=True)
    trainModel.Model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set the train data and validation data, start train
    history = trainModel.Model.fit_generator(train_generator, validation_data=(x_validate, y_validate),
                                       epochs=epochs,steps_per_epoch=stepsPerEpoch, verbose=2, shuffle=True)

    # save the new trained model
    trainModel.Model.save(saveModel)

    # plot accuracy and loss
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.title('Training and Validation accurarcy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()
