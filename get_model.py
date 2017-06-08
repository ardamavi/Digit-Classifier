# Arda Mavi

import os
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout

def save_model(model):
    if not os.path.exists('Data/Model/'):
        os.makedirs('Data/Model/')
    model_json = model.to_json()
    with open("Data/Model/model.json", "w") as model_file:
        model_file.write(model_json)
    model.save_weights("Data/Model/weights.h5")
    print('Model and weights saved')
    return

def get_model():

    inputs = Input(shape=(28, 28, 1))

    conv_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    conv_1 = Conv2D(64, (3,3), strides=(1,1))(conv_1)
    conv_1 = Activation('relu')(conv_1)

    conv_2 = Conv2D(64, (3,3), strides=(1,1))(conv_1)
    conv_2 = Activation('relu')(conv_2)
    conv_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_2)

    flat_1 = Flatten()(conv_2)

    fc = Dense(128)(flat_1)
    fc = Activation('relu')(fc)
    fc = Dropout(0.5)(fc)
    fc = Dense(10)(fc)

    outputs = Activation('sigmoid')(fc)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    print(model.summary())

    return model
