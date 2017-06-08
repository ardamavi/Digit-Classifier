# Arda Mavi

import sys
import numpy as np
from get_dataset import get_img
from keras.models import Sequential
from keras.models import model_from_json

def predict(model, X):
    Y = model.predict(X)
    Y = np.argmax(Y, axis=1)
    return Y

if __name__ == '__main__':
    img_dir = sys.argv[1]
    img = get_img(img_dir)
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    print(predict(model, img))
