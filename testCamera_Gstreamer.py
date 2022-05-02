
import numpy as np
import argparse
import cv2

def gstreamer_pipeline():
    return "libcamerasrc ! video/x-raw,  width=(int)1280, height=(int)1280, framerate=(fraction)25/1 ! videoconvert ! videoscale ! video/x-raw, width=(int)300, height=(int)300 ! appsink"
vs = cv2.VideoCapture(gstreamer_pipeline(),1800)

# grab the next frame from the video file
(grabbed, frame) = vs.read()
# check to see if we have reached the end of the video file

cv2.imshow("Frame", frame)
cv2.imwrite("test.jpg", frame)
print("end")