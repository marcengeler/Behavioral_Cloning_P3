import csv
import numpy as np
from PIL import Image


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
		image = np.array(Image.open(source_path))
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

X_train = (X_train-255)-0.5
plt.imshow(X_train[0])
plt.plot()

img = X_train[0]
img_new = X_train[:, 70:135]
plt.imshow(img_new)
plt.plot()

hist, bins = np.histogram(measurements, bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()