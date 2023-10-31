import os
import cv2
import dlib

pwd = os.path.dirname(__file__)

cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor(pwd + "/../data/shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)

        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 10)

        start_point = ((face_landmarks.part(50).x) - 100, face_landmarks.part(50).y - 30)
        end_point = ((face_landmarks.part(56).x) + 100, face_landmarks.part(56).y + 30)
        cv2.rectangle(frame, start_point, end_point, (0, 0, 255), 5)
        cv2.imwrite("frame%d.jpg" % 0, frame[190:236,80:220,:]) 

    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()