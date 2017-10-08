import csv
import cv2
import numpy as np


### Read in the CSV Data ###
path_to_data = 'C:/Users/Marc Engeler/PycharmProjects/simulator_learning_data/'
lines = []
with open(path_to_data + 'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

### Read in the Image and Measurement Data ###
images = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[i]
		image = cv2.imread(source_path)
		# Read in the image
		images.append(image)
		# Also mirror the image
		images.append(np.fliplr(image))
		measurement = float(line[3])
		# For the left and right camera, change the measurement
		# signal a little bit, that the vehicle steers back to the
		# center of the road
		if i==1:
			measurement = measurement + 0.25
		if i==2:
			measurement = measurement -0.25
		# Append measurements for the image and the inverted image
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
## Build the Sequential Model
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5,5, subsample=(2,2), border_mode='same', activation='relu'))
model.add(Convolution2D(36, 5,5, subsample=(2,2), border_mode='same', activation='relu'))
model.add(Convolution2D(48, 5,5, subsample=(2,2), border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3,3, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3,3, border_mode='same', activation='relu'))
model.add(Flatten())
# model.add(Activation('relu'))
# model.add(Dropout(0.50))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.50))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(1))

# Train Model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

# Save the Model
model.save('model.h5')

import matplotlib.pyplot as plt
### Plot Validation and Training Loss
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()