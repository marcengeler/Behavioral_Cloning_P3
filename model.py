import csv
import cv2
import numpy as np

path_to_data = 'C:/Users/Marc Engeler/PycharmProjects/simulator_learning_data/'
lines = []
with open(path_to_data + 'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[i]
		image = cv2.imread(source_path)
		images.append(image)
		images.append(np.fliplr(image))
		measurement = float(line[3])
		if i==1:
			measurement = measurement + 0.25
		if i==2:
			measurement = measurement -0.25
		measurements.append(measurement)
		measurements.append(-measurement)

X_train = np.asarray(images)
y_train = measurements

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
## Add An Inception Module
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3,3, activation='relu'))
model.add(Convolution2D(64, 3,3, activation='relu'))
model.add(Flatten())
model.add(Activation('relu'))
model.add(Dropout(0.65))
model.add(Dense(64))
model.add(Dropout(0.65))
model.add(Dense(32))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

model.save('model.h5')

import matplotlib.pyplot as plt
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()