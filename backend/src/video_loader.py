import os
import cv2
import tensorflow as tf
from typing import List
import cv2
from matplotlib import pyplot as plt
import dlib
import imageio

pwd = os.path.dirname(__file__)

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor(pwd + "/../data/shape_predictor_68_face_landmarks.dat")

def load_video(path:str) -> List[float]: 

    cap = cv2.VideoCapture(path)
    face_landmarks = dlib_facelandmark(gray, face[0])
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)

        start_point = ((face_landmarks.part(50).x) - 100, face_landmarks.part(50).y - 30)
        end_point = ((face_landmarks.part(56).x) + 100, face_landmarks.part(56).y + 30)

        frames.append(frame[190:236,80:220,:])

    cv2.imwrite("frame%d.jpg" % 0, frames[0][190:236,80:220,:]) 
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

frames = load_video('../data/test_video.mpg')
plt.imshow(frames[0])