from keras.models import load_model
import glob
from PIL import Image
import os
import numpy as np
if os.path.isfile("FaceDetect.h5"):
	model = load_model("FaceDetect.h5")
	image_names = [img for img in glob.glob('/home/kev/Desktop/*.jpg')]
	image = Image.open(image_names[0])
	image.show()
	resized = image.resize((64, 96))
	train_image = np.asarray(resized, dtype ="int32")
	train_image = train_image / 255
	print(train_image.shape)
	print(model.predict(train_image.reshape(1,96,64,3)))
else:
	print("No model found!")