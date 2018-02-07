import numpy as np
import os
import cv2
import pandas as pd

from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop, Adam, Adadelta

from rdkit import Chem
from rdkit.Chem import Draw
image_size = 300

df_train = pd.read_csv("train.csv", nrows = 50)

def load_train():
    X_train = []
    y_train = []
    print('Read train images')
    for i in range(len(df_train)):
        #print i
        smile = df_train.smiles[i]
        pil_image = Draw.MolToImage(Chem.MolFromSmiles(smile))
        #pil_image.show()
        open_cv_image = np.array(pil_image)
        #open_cv_image = open_cv_image[:, :, ::-1].copy() 
        #img = cv2.resize(open_cv_image,(image_size, image_size)).astype(np.float32)
        #img = img.transpose((2,0,1))
        X_train.append(open_cv_image)
        y_train.append(df_train.gap[i])
    return X_train, y_train

def read_and_normalize_train_data():
    train_data, train_target = load_train()
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.float32)
    m = train_data.mean()
    s = train_data.std()

    print ('Train mean, sd:', m, s )
    train_data -= m
    train_data /= s
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target


def create_model():
    nb_filters = 8
    nb_conv = 5

    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(image_size, image_size, 3) ) )
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer=Adadelta())
    return model

def train_model(batch_size = 50, nb_epoch = 20):
    num_samples = 50
    cv_size = 10

    train_data, train_target = read_and_normalize_train_data()
    train_data = train_data[0:num_samples,:,:,:]
    train_target = train_target[0:num_samples]

    X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target, test_size=cv_size, random_state=56741)

    model = create_model()
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_valid, y_valid) )

    predictions_valid = model.predict(X_valid, batch_size=50, verbose=1)
    compare = pd.DataFrame(data={'original':y_valid.reshape((cv_size,)),
             'prediction':predictions_valid.reshape((cv_size,))})
    compare.to_csv('compare.csv')

    return model

train_model(nb_epoch = 50)