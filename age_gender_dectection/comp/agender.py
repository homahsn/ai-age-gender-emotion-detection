import cv2
from comp.pyagender import PyAgender

agender = PyAgender() 

def detect(i):
    faces = agender.detect_genders_ages(cv2.imread(i))
    res = []
    if not faces:
        res = [('No face detected', '')]
    for f in faces:
        gender = 'Male  ' if f.get('gender') < .5 else 'Female'
        age = int(f.get('age'))
        res.append((gender, age))
    return res
