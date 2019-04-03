from PIL import Image
import glob
from random import randint

images = []

image_names = [img for img in glob.glob('/home/kev/Desktop/FaceDetection/project_train/train/*.jpg')]
for img in sorted(image_names):
	image = Image.open(img)
	images.append(image)


with open("/home/kev/Desktop/FaceDetection/project_train/label.txt", "r") as file:
	data = file.readlines()

data = [x.strip() for x in data]

labels = []
for d in data:
	numbers = d.split()
	labels.append([int(numbers[0])-1,int(numbers[1]), int(numbers[2]), int(numbers[3]), int(numbers[4])])

count = 1
for label in labels:
	new_image = images[label[0]].copy()
	cropped = new_image.crop( (label[2], label[1], label[2] + label[4], label[1] + label[3]) )
	resized = cropped.resize((64, 96))
	#Save path
	path = "/home/kev/Desktop/FaceDetection/Faces/" + str(count).zfill(4) + ".jpg"
	count += 1
	resized.save(path, "JPEG")

count = 1
for image in images:
	for i in range(2):
		new_image = image.copy()
		image_width, image_height = new_image.size
		x = randint(1, image_width-101)
		y = randint(1, image_height-101)
		crop_width = randint(50, 101)
		crop_height = randint(50, 101)
		cropped = new_image.crop( (x,y,x+crop_width, y+crop_height))
		resized = cropped.resize((64,96))
		#Save path
		path = "/home/kev/Desktop/FaceDetection/NotFaces/" + str(count).zfill(4) + ".jpg"
		count += 1
		resized.save(path, "JPEG")




