# Arda Mavi

import keras
import numpy as np
from scipy.misc import imread, imresize
from keras.datasets import mnist

def get_img(data_path):
    img_size = 28
    img = imread(data_path)
    img = imresize(img, (img_size, img_size, 1))
    img = img.reshape(1, 28, 28, 1)
    return img

def get_dataset():
    (X, Y), (X_test, Y_test) = mnist.load_data()
    X = X.reshape(X.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    Y = keras.utils.to_categorical(Y, 10)
    Y_test = keras.utils.to_categorical(Y_test, 10)
    return X, X_test, Y, Y_test
