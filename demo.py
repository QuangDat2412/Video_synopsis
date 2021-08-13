#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from constant import VIDEO_PATH

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync
from constant import *
warnings.filterwarnings('ignore')


def main(yolo):

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    tracking = True
    writeVideo_flag = True
    asyncVideo_flag = False

    if asyncVideo_flag :
        video_capture = VideoCaptureAsync(VIDEO_PATH)
    else:
        video_capture = cv2.VideoCapture(VIDEO_PATH)

    if asyncVideo_flag:
        video_capture.start()

    if not video_capture.isOpened():
        print("open false!")

    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
        else:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        print(w, h)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('./temp_file/output.avi', fourcc, 30, (w, h))
        list_file = open('./temp_file/detection.txt', 'w')
        tracking_file = open('./temp_file/tracking.txt', 'w')
        frame_index = -1 

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
             break

        t1 = time.time()

        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxes, confidence, classes = yolo.detect_image(image)

        if tracking:
            features = encoder(frame, boxes)

            detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                          zip(boxes, confidence, classes, features)]
        else:
            detections = [Detection_YOLO(bbox, confidence, cls) for bbox, confidence, cls in
                          zip(boxes, confidence, classes)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        print(scores)
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker_IDs = []
        if tracking:
            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
                tracker_IDs.append(track.track_id)

        for det, id in zip(detections, tracker_IDs):
            bbox = det.to_tlbr()
            score = "%.2f" % round(det.confidence * 100, 2) + "%"
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            if len(classes) > 0:
                cls = det.cls
                cv2.putText(frame, str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 200, (0, 0, 225), 2)

        if writeVideo_flag: # and not asyncVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

            list_file.write(str(frame_index) + ' @' + str(len(boxes)) + ' ')
            if len(boxes) != 0:
                for i in range(0,len(boxes)):
                    list_file.write('$' + str(boxes[i][0]) + ' '+str(boxes[i][1]) + ' '+str(boxes[i][2]) + ' '+str(boxes[i][3]) + ' ')
            list_file.write('\n')

            # 写入tracking
            tracking_file.write(str(frame_index) + ' @' + str(len(tracker.tracks)) + ' ')
            for track in tracker.tracks:
                if (not track.is_confirmed() or track.time_since_update > 1) and frame_index >= 2 :
                    continue
                bbox = track.to_tlbr()
                tracking_file.write('$ ' + str(track.track_id) + ' ' + str(int(bbox[0])) + ' ' + str(int(bbox[1])) + ' ' + str(int(bbox[2])) + ' ' + str(int(bbox[3])) + ' ')

            tracking_file.write('\n')

        fps_imutils.update()

        if not asyncVideo_flag:
            fps = (fps + (1./(time.time()-t1))) / 2
            print("FPS = %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
