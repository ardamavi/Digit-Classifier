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


# Other Models:

# LENET-5
def get_lenet_5():
    inputs = Input(shape=(28, 28, 1))
    
    conv_1 = Conv2D(6, (5, 5), strides=(1,1), padding='SAME')(inputs)
    act_1 = Activation('relu')(conv_1)
    
    pooling_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act_1)
    
    conv_2 = Conv2D(16, (5, 5), strides=(1,1), padding='SAME')(pooling_1)
    act_2 = Activation('relu')(conv_2)
    
    pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act_2)
    
    conv_3 = Conv2D(120, (5, 5), strides=(1,1), padding='SAME')(pooling_2)
    act_3 = Activation('relu')(conv_3)
    
    flat_1 = Flatten()(act_3)
    
    fc = Dense(84)(flat_1)
    fc = Activation('relu')(fc)
    fc = Dropout(0.5)(fc)
    
    fc = Dense(10)(fc)
    outputs = Activation('softmax')(fc)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

# MVGG-5
def get_mvgg_5():
    inputs = Input(shape=(28, 28, 1))
    
    conv_1 = Conv2D(16, (3, 3), strides=(1,1), padding='SAME')(inputs)
    act_1 = Activation('relu')(conv_1)
    
    conv_2 = Conv2D(16, (3, 3), strides=(1,1), padding='SAME')(act_1)
    act_2 = Activation('relu')(conv_2)
    
    pooling_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(act_2)
    
    conv_3 = Conv2D(48, (3, 3), strides=(1,1), padding='SAME')(pooling_1)
    act_3 = Activation('relu')(conv_3)
    
    pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(act_3)
    
    flat_1 = Flatten()(pooling_2)
    
    fc = Dense(128)(flat_1)
    fc = Activation('relu')(fc)
    fc = Dropout(0.5)(fc)
    
    fc = Dense(10)(fc)
    outputs = Activation('softmax')(fc)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

# MVGG-6
def get_mvgg_6():
    inputs = Input(shape=(28, 28, 1))
    
    conv_1 = Conv2D(16, (3, 3), strides=(1,1), padding='SAME')(inputs)
    act_1 = Activation('relu')(conv_1)
    
    conv_2 = Conv2D(16, (3, 3), strides=(1,1), padding='SAME')(act_1)
    act_2 = Activation('relu')(conv_2)
    
    pooling_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(act_2)
    
    conv_3 = Conv2D(32, (3, 3), strides=(1,1), padding='SAME')(pooling_1)
    act_3 = Activation('relu')(conv_3)
    
    conv_4 = Conv2D(48, (3, 3), strides=(1,1), padding='SAME')(act_3)
    act_4 = Activation('relu')(conv_4)
    
    pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(act_4)
    
    flat_1 = Flatten()(pooling_2)
    
    fc = Dense(128)(flat_1)
    fc = Activation('relu')(fc)
    fc = Dropout(0.5)(fc)
    
    fc = Dense(10)(fc)
    outputs = Activation('softmax')(fc)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

# MVGG-7
def get_mvgg_7():
    inputs = Input(shape=(28, 28, 1))
    
    conv_1 = Conv2D(16, (3, 3), strides=(1,1), padding='SAME')(inputs)
    act_1 = Activation('relu')(conv_1)
    
    conv_2 = Conv2D(16, (3, 3), strides=(1,1), padding='SAME')(act_1)
    act_2 = Activation('relu')(conv_2)
    
    pooling_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(act_2)
    
    conv_3 = Conv2D(32, (3, 3), strides=(1,1), padding='SAME')(pooling_1)
    act_3 = Activation('relu')(conv_3)
    
    conv_4 = Conv2D(32, (3, 3), strides=(1,1), padding='SAME')(act_3)
    act_4 = Activation('relu')(conv_4)
    
    pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(act_4)
    
    conv_5 = Conv2D(48, (3, 3), strides=(1,1), padding='SAME')(pooling_2)
    act_5 = Activation('relu')(conv_5)
    
    pooling_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(act_5)
    
    flat_1 = Flatten()(pooling_3)
    
    fc = Dense(128)(flat_1)
    fc = Activation('relu')(fc)
    fc = Dropout(0.5)(fc)
    
    fc = Dense(10)(fc)
    outputs = Activation('softmax')(fc)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

# MVGG-8
def get_mvgg_8():
    inputs = Input(shape=(28, 28, 1))
    
    conv_1 = Conv2D(16, (3, 3), strides=(1,1), padding='SAME')(inputs)
    act_1 = Activation('relu')(conv_1)
    
    conv_2 = Conv2D(16, (3, 3), strides=(1,1), padding='SAME')(act_1)
    act_2 = Activation('relu')(conv_2)
    
    pooling_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(act_2)
    
    conv_3 = Conv2D(32, (3, 3), strides=(1,1), padding='SAME')(pooling_1)
    act_3 = Activation('relu')(conv_3)
    
    conv_4 = Conv2D(32, (3, 3), strides=(1,1), padding='SAME')(act_3)
    act_4 = Activation('relu')(conv_4)
    
    pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(act_4)
    
    conv_5 = Conv2D(48, (3, 3), strides=(1,1), padding='SAME')(pooling_2)
    act_5 = Activation('relu')(conv_5)
    
    conv_6 = Conv2D(48, (3, 3), strides=(1,1), padding='SAME')(act_5)
    act_6 = Activation('relu')(conv_6)
    
    pooling_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(act_6)
    
    flat_1 = Flatten()(pooling_3)
    
    fc = Dense(128)(flat_1)
    fc = Activation('relu')(fc)
    fc = Dropout(0.5)(fc)
    
    fc = Dense(10)(fc)
    outputs = Activation('softmax')(fc)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

# MVGG-9
def get_mvgg_9():
    inputs = Input(shape=(28, 28, 1))
    
    conv_1 = Conv2D(16, (3, 3), strides=(1,1), padding='SAME')(inputs)
    act_1 = Activation('relu')(conv_1)
    
    conv_2 = Conv2D(16, (3, 3), strides=(1,1), padding='SAME')(act_1)
    act_2 = Activation('relu')(conv_2)
    
    pooling_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(act_2)
    
    conv_3 = Conv2D(32, (3, 3), strides=(1,1), padding='SAME')(pooling_1)
    act_3 = Activation('relu')(conv_3)
    
    conv_4 = Conv2D(32, (3, 3), strides=(1,1), padding='SAME')(act_3)
    act_4 = Activation('relu')(conv_4)
    
    pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(act_4)
    
    conv_5 = Conv2D(48, (3, 3), strides=(1,1), padding='SAME')(pooling_2)
    act_5 = Activation('relu')(conv_5)
    
    conv_6 = Conv2D(48, (3, 3), strides=(1,1), padding='SAME')(act_5)
    act_6 = Activation('relu')(conv_6)
    
    pooling_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(act_6)
    
    conv_7 = Conv2D(64, (3, 3), strides=(1,1), padding='SAME')(pooling_3)
    act_7 = Activation('relu')(conv_7)
    
    pooling_4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='SAME')(act_7)
    
    flat_1 = Flatten()(pooling_2)
    
    fc = Dense(128)(flat_1)
    fc = Activation('relu')(fc)
    fc = Dropout(0.5)(fc)
    
    fc = Dense(10)(fc)
    outputs = Activation('softmax')(fc)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model
