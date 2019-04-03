import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers import Conv2D
import glob
from PIL import Image
from matplotlib import pyplot as plt
from random import shuffle
from random import randint

def model():
	model = Sequential()
	model.add(Conv2D(64, kernel_size=3, kernel_initializer='normal', activation='relu', input_shape=(96,64,3)))
	model.add(Conv2D(32, kernel_size=3, kernel_initializer='normal', activation='relu'))
	model.add(Flatten())
	model.add(Dense(32,  kernel_initializer='normal', activation='relu'))
	model.add(Dense(1,  kernel_initializer='normal', activation='sigmoid'))
	model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
	return model
#Initialize model
model = model()

#Read data
faces = []
image_names = [img for img in glob.glob('/home/kev/Desktop/FaceDetection/Faces/*.jpg')]
image_names.sort()
for img in image_names:
	image = Image.open(img)
	faces.append(np.asarray(image, dtype ="int32"))


not_faces = []
image_names = [img for img in glob.glob('/home/kev/Desktop/FaceDetection/NotFaces/*.jpg')]
for img in sorted(image_names):
	image = Image.open(img)
	not_faces.append(np.asarray(image, dtype ="int32"))


all_images = faces + not_faces
for i in range(len(all_images)):
	all_images[i] = all_images[i] / 255
image_array = np.array(all_images)

labels = np.zeros( (len(all_images), 1) )
for i in range(len(faces)):
	labels[i] = 1
for i in range(len(faces), len(all_images)):
	labels[i] = 0


#Train
model.fit(image_array, labels, shuffle = True, epochs = 10, batch_size = 32, validation_split = 0.2)

#Save model
print("Do you want to save the model? (y for yes, any key for no)")
n = input()
if n == 'y':
	model.save("FaceDetect.h5")
	print("Model saved!")
else:
	print("Model not saved!")



