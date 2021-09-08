#github.com/msiddhu/Emotion-detection
import numpy as np
import argparse
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import csv
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

def get_face(path):
	pixels = pyplot.imread(path)
	h,w = pixels.shape[0],pixels.shape[1]
	detector = MTCNN()
	faces = detector.detect_faces(pixels)

	max_conf = 0
	max_conf_roi = [0,0,w,h]

	#find the face with max conf
	#if no face, default is whole image
	for face in faces:
		if face["confidence"]>max_conf:
			max_conf = face["confidence"]
			max_conf_roi = face["box"]
	if max_conf_roi[0]<0:
		max_conf_roi[0]=0
	if max_conf_roi[1]<0:
		max_conf_roi[1]=0

	return max_conf_roi

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'





# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
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
model.add(Dense(7, activation='softmax'))



# emotions will be displayed on your face from the webcam feed
model.load_weights('model.h5')
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}



#testing on cleaned insta data
labels = ["happy","neutral","happy","happy","sad","neutral","sad","surprised","angry","happy","sad","neutral","surprised","happy","sad","surprised","happy","surprised","happy","sad","neutral","neutral","sad","sad","neutral","sad","surprised","neutral","happy","sad","happy","sad","sad","neutral","neutral","sad","happy","happy","sad","sad","neutral","neutral","neutral","sad","neutral","sad","sad","neutral","angry","sad","neutral","neutral","neutral","angry","neutral","surprised","happy","neutral","neutral","sad","neutral","happy","happy","happy","happy","happy","happy","sad","neutral","happy","angry","neutral","surprised","neutral","neutral","neutral","happy","happy","sad","happy","neutral","surprised","neutral","sad","happy","sad","angry","neutral","neutral","sad","neutral","surprised","happy","happy","sad","sad","surprised","happy","angry","neutral","sad","sad","neutral","sad","happy","happy","neutral","happy","neutral","sad","neutral","happy","neutral","neutral","happy","happy","neutral","happy","neutral","happy","neutral","sad","neutral","neutral","sad","happy","happy","sad","happy","angry","surprised","happy","neutral","sad","sad","sad","sad","happy","happy","happy","neutral","happy","neutral","neutral","neutral","happy","sad","sad","happy","angry","sad","neutral","happy","neutral","happy","happy","sad","sad","neutral","neutral","happy","happy","sad","neutral","neutral","neutral","neutral","neutral","neutral","neutral","sad","happy","sad","neutral","sad","sad","neutral","happy","angry","happy","neutral","happy","neutral","happy","happy","happy","neutral","happy","happy","neutral","neutral","happy","neutral","neutral","happy","sad","sad","neutral","sad","happy"]
insta_test_path = "..\\insta_data\\"
total = 200
good  = 0
bad = []
result_rows = []
for i in range(1,total+1):
	path = insta_test_path+str("{:03d}".format(i))+".jpg"
	print(path)
	[x,y,w,h] = get_face(path)
	print("face",x,y,w,h)
	frame = cv2.imread(path)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = gray[y:y + h, x:x + w]
	expanded = np.expand_dims(np.expand_dims(cv2.resize(gray, (48, 48)), -1), 0)
	prediction = model.predict(expanded)
	maxindex = int(np.argmax(prediction))
	result_rows.append([labels[i-1],emotion_dict[maxindex]])
	if emotion_dict[maxindex]==labels[i-1]:
		good+=1
	else:
		bad.append(i)
print("Test on instagram dataset ",good/total)
print("These images where wrong:")
print(bad)


with open('result1.csv', 'w') as f: 
	csvwriter = csv.writer(f) 
    # writing the fields 
	csvwriter.writerow(['actual','predicted'])
	csvwriter.writerows(result_rows)

'''
output was:
Test on instagram dataset  0.425
These images where wrong:
[2, 3, 5, 6, 7, 8, 9, 15, 16, 18, 20, 21, 22, 23, 24, 26, 27, 28, 30, 32, 33, 35, 36, 39, 41, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 58, 60, 61, 62, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 86, 87, 88, 89, 90, 92, 95, 96, 97, 98, 100, 102, 103, 104, 107, 109, 110, 113, 115, 117, 119, 122, 123, 125, 128, 130, 131, 133, 134, 135, 137, 141, 142, 144, 147, 148, 150, 151, 152, 156, 158, 159, 163, 164, 165, 166, 167, 171, 173, 175, 176, 177, 179, 183, 187, 191, 196, 197, 199]
'''