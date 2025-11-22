import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam

CSV_PATH = "fer2013.csv"
IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 40

def load_data():
    df = pd.read_csv(CSV_PATH)
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    for _, row in df.iterrows():
        pixels = np.fromstring(row['pixels'], sep=' ')
        if pixels.size != IMG_SIZE * IMG_SIZE:
            continue
        img = pixels.reshape((IMG_SIZE, IMG_SIZE))
        if row['Usage'] == 'Training':
            X_train.append(img)
            y_train.append(row['emotion'])
        elif row['Usage'] == 'PublicTest':
            X_val.append(img)
            y_val.append(row['emotion'])
        else:
            X_test.append(img)
            y_test.append(row['emotion'])
    def prep(X):
        X = np.array(X, dtype='float32')/255.0
        return np.expand_dims(X, -1)
    return prep(X_train), to_categorical(y_train, NUM_CLASSES), prep(X_val), to_categorical(y_val, NUM_CLASSES), prep(X_test), to_categorical(y_test, NUM_CLASSES)

def model_build():
    m = Sequential()
    m.add(Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(IMG_SIZE,IMG_SIZE,1)))
    m.add(BatchNormalization())
    m.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    m.add(BatchNormalization())
    m.add(MaxPooling2D((2,2)))
    m.add(Dropout(0.25))
    m.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    m.add(BatchNormalization())
    m.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    m.add(BatchNormalization())
    m.add(MaxPooling2D((2,2)))
    m.add(Dropout(0.25))
    m.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    m.add(BatchNormalization())
    m.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    m.add(BatchNormalization())
    m.add(MaxPooling2D((2,2)))
    m.add(Dropout(0.25))
    m.add(Flatten())
    m.add(Dense(256,activation='relu'))
    m.add(BatchNormalization())
    m.add(Dropout(0.5))
    m.add(Dense(NUM_CLASSES,activation='softmax'))
    return m

X_train, y_train, X_val, y_val, X_test, y_test = load_data()
model = model_build()
model.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)
model.save("emotion_model.h5")
