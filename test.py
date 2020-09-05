import os
import random
from imageio import imwrite
from keras import backend as K
import tensorflow as tf
from keras import activations, Sequential
from vis.visualization import visualize_cam, overlay
from vis.utils import utils

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Model import buildModel
from GetImage import getImage

# calculate the predict accuracy if you have ground truth
def calcValidationAccuracy(y_GT, y_pred):
    num=0
    for i in range(len(y_pred)):
        if y_pred[i]==y_GT[i]:
            num=num+1
    print(num / len(y_pred))

# statistic the confusion value in a matrix form
def calcConfusionMatrix(y_train, y_pred):
    labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    tick_marks = np.array(range(len(labels))) + 0.5
    cm = confusion_matrix(y_train, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print (cm_normalized)
    plt.figure(figsize=(12, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plotConfusionMatrix(cm_normalized,labels, title='Normalized confusion matrix')
    plt.savefig('confusion_matrix.png', format='png')
    plt.show()

# plot the matrix
def plotConfusionMatrix(cm, labels, title='Confusion Matrix'):
    cmap = plt.cm.binary
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# write the predict results into a txt file
def geneTestPredictFile(predict,fileName):
    # remove the exist file in the same name, be careful!!!
    if os.access(fileName, os.R_OK):
        os.remove(fileName)
    writehardElem = open(fileName, 'a')
    writehardElem.write('id,label\n')
    for i in range(len(predict)):
        writehardElem.write(str(i))
        writehardElem.write(',')
        writehardElem.write(str(predict[i]))
        writehardElem.write('\n')

# show saliency map, heat map
def showSaliencyMap(showModel,layerName,showImage,imageSize):
    layer_idx = utils.find_layer_idx(showModel, layerName)
    showModel.layers[layer_idx].activation = activations.linear
    showModel = utils.apply_modifications(showModel)
    plt.rcParams['figure.figsize'] = (18, 6)
    for modifier in [None, 'guided', 'relu']:
        plt.figure()
        showImage=np.array(showImage)
        f, ax = plt.subplots(1, len(showImage))
        plt.suptitle("vanilla" if modifier is None else modifier)
        for i, img in enumerate(showImage):
            img = np.array(img)
            grads = visualize_cam(showModel, layer_idx, filter_indices=3,seed_input=img, backprop_modifier=modifier)
            # overlay the heatmap onto original image.
            jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
            jet_heatmap = np.array(np.reshape(jet_heatmap, (imageSize, imageSize, 3)), dtype=np.uint8)
            if len(showImage)==1:
                # for single image
                ax.imshow(overlay(jet_heatmap, img))
            else:
                # for multi images
                ax[i].imshow(overlay(jet_heatmap, img))
        plt.show()

# decode the array into a image form
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def filterVisualization(showModel, layerName, imageSize,showRows,filterNum,showImage,overlapOnOriginImage):
    # the placeholder for the input images
    input_img = showModel.input

    # get the symbolic outputs of each "key" layer
    layer_dict = dict([(layer.name, layer) for layer in showModel.layers[1:]])
    kept_filters = []

    # only scan through the first 200 filters, even though we have more
    for filter_index in range(filterNum):
        # build a loss function that maximizes the activation of the nth filter of the layer considered
        layer_output = layer_dict[layerName].output
        loss = K.mean(layer_output[:, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]
        # normalize the gradient
        grads = normalize(grads)
        # returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.
        if overlapOnOriginImage:
            # lower the gray level of origin image, makes show better
            input_img_data = np.array(showImage)
            input_img_data = (input_img_data / 255 - 0.5) * 100 + 138
        else:
            # generate a gray image with some random noise
            input_img_data = np.random.random((1, imageSize, imageSize, 3))
            input_img_data = (input_img_data - 0.5) * 20 + 128

        # run gradient ascent for 60 steps
        for i in range(60):
            loss_value, grads_value = iterate([input_img_data])
            if overlapOnOriginImage:
                input_img_data = np.array(input_img_data, dtype=np.float32)
            input_img_data += grads_value * step
            if loss_value <= 0.:  # some filters get stuck to 0, we can skip them
                break
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
    
    # choose the largest filters, the number is showRows*showRows
    # it may give an error when the suitable kept_filters are not enough,
    # when it happens, you can increase the 'filterNum', or decrease the 'showRows'
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:showRows * showRows]
    # set a margin size on canvas, between images
    margin = 5
    width = showRows * imageSize + (showRows - 1) * margin
    height = showRows * imageSize + (showRows - 1) * margin
    stitched_filters = np.zeros((width, height, 3))
    # draw images one by one, arrange the location on canvas
    for i in range(showRows):
        for j in range(showRows):
            img, loss = kept_filters[i * showRows + j]
            stitched_filters[(imageSize + margin) * i: (imageSize + margin) * i + imageSize,
            (imageSize + margin) * j: (imageSize + margin) * j + imageSize, :] = img
    imwrite('stitched_filters_%dx%d.png' % (showRows, showRows), stitched_filters)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='testing')
    parser.add_argument('-s', '--ImageSize', type=int, default=96)
    parser.add_argument('-f', '--FinetuneVggLayerIndex', type=int, default=0,
                        help='-2,-3,-4 are conv layers in last block of Vgg16')
    parser.add_argument('-pre','--PreTrainDir', type=str,
                        default='trainWellModel.h5')
    parser.add_argument('-v', '--CheckValidationStatus', type=bool,default=False)
    parser.add_argument('-t', '--CheckTestSetStatus', type=bool, default=False)
    parser.add_argument('-map', '--SaliencyMap', type=bool, default=False) #True
    parser.add_argument('-fv', '--FilterVis', type=bool, default=False)
    args = parser.parse_args()

    # build a model for test
    testModel=buildModel(ImageSize=args.ImageSize,FinetuneVggLayerIndex=args.FinetuneVggLayerIndex,PreTrainDir=args.PreTrainDir)

    if args.CheckValidationStatus:
        validateImage= getImage(imageDir='./train.csv',imageSize=args.ImageSize,startIndex=18000,histEqual=True)
        validateGtLabel= validateImage.label
        validatePredLabel = np.argmax(testModel.Model.predict(validateImage.colorImageInTargetSize), axis=1)
        # check validation accuracy here if you have another validation set
        calcValidationAccuracy(validateGtLabel, validatePredLabel)
        # generate a confusion matrix
        calcConfusionMatrix(validateGtLabel, validatePredLabel)

    # predict the test.csv and generate the submission file
    if args.CheckTestSetStatus:
        testImage = getImage(imageDir='./test.csv', imageSize=args.ImageSize, startIndex=1, histEqual=True)
        testPredLabel = np.argmax(testModel.Model.predict(testImage.colorImageInTargetSize), axis=1)
        geneTestPredictFile(testPredLabel,'predSubmission.csv')

    # specify a conv layer and a image, draw a saliency map
    if args.SaliencyMap:
        showImage= getImage(imageDir='./train.csv',imageSize=args.ImageSize,startIndex=15,endIndex=17,histEqual=True)
        showSaliencyMap(testModel.Model, 'conv2d_3', showImage.colorImageInTargetSize,args.ImageSize)

    # draw filter visualization
    if args.FilterVis:
        # False: draw on random noise, you can find out what those filters are extracting
        # True: draw on origin image, you can find out what those filters do on our image
        filterVisOverOriginImage=False
        showImage = getImage(imageDir='./train.csv', imageSize=args.ImageSize, startIndex=15, endIndex=16,histEqual=True)
        filterVisualization(testModel.Model, 'block3_conv3', args.ImageSize,4,100,showImage.colorImageInTargetSize,filterVisOverOriginImage)
