#github.com/omar178/Emotion-recognition
import os
from keras.preprocessing.image import img_to_array
import cv2
from keras.models import load_model
import numpy as np
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

emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgusted","fearful", "happy", "sad", "surprised", "neutral"]

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
	roi = cv2.resize(gray, (64, 64))
	roi = roi.astype("float") / 255.0
	roi = img_to_array(roi)
	roi = np.expand_dims(roi, axis=0)
	preds = emotion_classifier.predict(roi)[0]
	emotion_probability = np.max(preds)
	label = EMOTIONS[preds.argmax()]
	result_rows.append([labels[i-1],label])
	if label==labels[i-1]:
		good+=1
	else:
		bad.append(i)
	print(good)
print("Test on instagram dataset ",good/total)
print("These images where wrong:")
print(bad)

with open('result2.csv', 'w') as f: 
	csvwriter = csv.writer(f) 
    # writing the fields 
	csvwriter.writerow(['actual','predicted'])
	csvwriter.writerows(result_rows)

'''
output was:
Test on instagram dataset  0.505
These images where wrong:
[7, 8, 9, 11, 12, 15, 16, 18, 20, 21, 22, 23, 24, 26, 27, 32, 33, 36, 39, 40, 41, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 56, 60, 62, 68, 69, 71, 73, 78, 79, 80, 81, 82, 84, 86, 87, 91, 92, 95, 96, 97, 98, 99, 100, 101, 102, 103, 106, 107, 109, 110, 111, 114, 115, 119, 123, 128, 130, 131, 132, 134, 135, 137, 142, 143, 144, 147, 148, 150, 151, 153, 154, 159, 160, 161, 162, 163, 164, 166, 168, 170, 173, 175, 177, 179, 191, 195, 196, 197]
'''