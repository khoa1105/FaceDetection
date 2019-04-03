from keras.models import load_model

if os.path.isfile("FaceDetect.h5"):
	model = load_model("FaceDetect.h5")