import cv2
import dlib

# Load the video file
cap = cv2.VideoCapture("./data/testvideo.mpg")

# Get the video's width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load the dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")

# Create a VideoWriter object to save the cropped video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# print(int(cap.get(cv2.CAP_PROP_FPS)))
out = cv2.VideoWriter("cropped_video.mp4", fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (width, height))

# Iterate over the video frames
while True:
    # Capture the next frame
    ret, frame = cap.read()

    # If the frame is empty, break the loop
    if not ret:
        break

    # Detect faces in the frame
    rects = detector(frame)

    # For each detected face
    for rect in rects:
        # Extract the mouth landmarks
        # mouth_landmarks = predictor(frame, rect)[48:68]
        mouth_landmarks = predictor(frame, rect)[48:68]
        print(f'mouth_landmarks: {mouth_landmarks}')
        # Compute the convex hull of the mouth landmarks
        hull = cv2.convexHull(mouth_landmarks)

        # Crop the frame to the mouth region
        cropped_frame = frame[hull[:, 1].min():hull[:, 1].max(), hull[:, 0].min():hull[:, 0].max()]

        # Write the cropped frame to the output video
        out.write(cropped_frame)

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()
