import random
import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
class getImage(object):
    def __init__(self,
                 imageDir='./train.csv',
                 imageSize=224,
                 startIndex=1,
                 endIndex=0,
                 histEqual= True):
        self.imageDir = imageDir
        self.imageSize = imageSize
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.histEqual = histEqual
        self.label, self.rawImage = self.getRawImage(self.imageDir, self.startIndex, self.endIndex,self.histEqual)
        self.colorImageInOriginSize = self.convGray2RGB(self.rawImage)
        self.colorImageInTargetSize = np.array(self.upscaleImage(self.colorImageInOriginSize, imageSize))

    # statistical gray scale distribution
    def calcHistogram(self,image):
        grayStatistic = [0] * 256  # how many pixels in each gray scale
        w = image.shape[0]
        h = image.shape[1]
        for i in range(w):
            for j in range(h):
                gray = image[i, j]
                grayStatistic[gray] += 1
        return grayStatistic

    # according to the gray scale distribution of a image, make equalization
    def histEqualization(self,grayStatistic, image):
        b = [0] * 256  # save gray scale ratio
        c = [0] * 256  # save accumulated counts
        w = image.shape[0]
        h = image.shape[1]
        mn = w * h * 1.0
        img = np.zeros([w, h], np.uint8)  # save img after equalization
        for i in range(len(grayStatistic)):
            b[i] = grayStatistic[i] / mn

        # accumulate
        for i in range(len(c)):
            if i == 1:
                c[i] = b[i]
            else:
                c[i] = c[i - 1] + b[i]
                grayStatistic[i] = int(255 * c[i])

        # equalization
        for i in range(w):
            for j in range(h):
                img[i, j] = grayStatistic[image[i, j]]
        return img

    # get the raw image(48x48 size) and lable in kaggle data set
    def getRawImage(self,imageDir,startLine,endLine,histEqual):
        lines = open(imageDir).readlines()
        label=[]
        rawImage=[]
        if endLine==0:
            endLine=lines.__len__()
        for line in lines[startLine:endLine]:
            line = line.strip('\n')
            line = line.split(',')
            label.append(int(line[0]))
            line = np.array(line[1].split(" "), dtype=np.uint8)
            if histEqual:
                # do histogram Equalization
                line = np.array(np.reshape(line, (48, 48)), dtype=np.uint8)
                a = self.calcHistogram(line)
                line = self.histEqualization(a, line)
                line = np.array(np.reshape(line, (48, 48)), dtype=np.uint8)
                rawImage.append(line)
        label = np.array(label)
        rawImage = np.array(rawImage)
        return label,rawImage

    # convert the gray image to RGB image, because VGG has a 3 channel input
    def convGray2RGB(self,grayImage):
        rgbImage = []
        for image in grayImage:
            image = np.array(np.reshape(image, (48, 48)), dtype=np.uint8)
            image = cv2.cvtColor(cv2.resize(image, (48, 48)), cv2.COLOR_GRAY2RGB)  # (48, 48, 3)
            rgbImage.append(image)
        rgbImage = np.array(rgbImage)
        return rgbImage

    # adjust the image size to model required input size
    def upscaleImage(self,image, largeScall):
        largeImage = [cv2.resize(i, (largeScall, largeScall)) for i in image]
        largeImage = np.array(largeImage)
        return largeImage

    # reshape the image set to model required input shape
    def concatenateImage4Model(self,image):
        concImage = np.concatenate([arr[np.newaxis] for arr in image])
        return concImage
