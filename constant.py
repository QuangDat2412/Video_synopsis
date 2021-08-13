# -*- coding: utf-8 -*-
import cv2
VIDEO_PATH = "./dataset/virat.avi"
vid = cv2.VideoCapture(VIDEO_PATH)
print(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
print(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_height = 1080
video_width = 1920
