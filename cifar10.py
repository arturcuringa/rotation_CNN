import numpy as np
import pandas as pd
import os, glob
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

 
PATH = os.getcwd()

X = [] 

y = pd.read_csv(PATH + '/train.truth.csv')

train_path = y.drop(y.columns[1] ,axis=1)
y = y.drop(y.columns[0] ,axis=1)
y = y.values.tolist()

encoder = LabelBinarizer().fit(["upright", "rotated_left", "rotated_right", "upside_down"])
y = encoder.transform(y)


for idx, sample in train_path.iterrows():
	x = image.load_img(PATH + '/train/' + sample[0], target_size=(32,32,3))
	X.append(image.img_to_array(x))


X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.3, random_state=42)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

# Train the model
model.fit(np.array(X_train), np.array(y_train),
          batch_size=128,
          shuffle=True,
          epochs=250,
          validation_data=(np.array(X_val), np.array(y_val)),
          callbacks=[EarlyStopping(min_delta=0.001, patience=3)])


if not os.path.isdir('model'):
    os.makedirs('model')
model_path = os.path.join('model', 'cifar10')
model.save(model_path)

