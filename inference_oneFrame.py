import sys
from pathlib import Path
import csv
import cv2
import datetime
import depthai
# import blobconverter
import numpy as np
import time
import os
import argparse
import copy
import pickle

"""
Script takes a video stream with the camera and keeps one frame.

INPUT:
Many arguments passed (organized below with parser) can affect the detection success, 
they must be tweaked for the particular situation.

Don't pass "-sho" to show output with odroid, it won't work

OUTPUT: 
- One raw frame (...raw.jpg)
and possibly 
- One frame with detection box(es) (...boxed.jpg)
- Location/size of the box(es) (...detections.csv)

"""

# ==============================================================================
# Parameters and functions
# ==============================================================================

HOME_PATH = str(Path.home())
parser = argparse.ArgumentParser()
parser.add_argument(
    '-iq', '--image_quality',  type=float, default = 100,
    help="Number in [0,100] setting image quality"
    )
parser.add_argument(
    '-ct', '--confidence_threshold',  type=float, default = 0.5,
    help="Sets a minimal confidence score. Detections scoring below are ignored"
    )
parser.add_argument(
    '-bt', '--blur_threshold', type=float, default = 1000,
    help="Sets a minimal score used to filter out blurry pictures (must be determined empirically)"
    )
parser.add_argument(
    '-sho', '--show_output', action="store_true", 
    help="Show captured frames during the stream (show video)"
    )
parser.add_argument(
    '-flip', '--flip', action="store_true", 
    help="Horizontal flip"
    )
parser.add_argument(
    '-out', '--output_path', type=str, default = HOME_PATH + '/projets/Camera-PMR-2/output/tests/',
    help="Path where .jpg and .csv output are saved"
    )
parser.add_argument(
    '-sd', '--stream_duration', type=float, default = 8,
    help="Number of seconds allowed to capture the stream"
    )
parser.add_argument(
    '-fps', '--frames_per_second', type = float, default = 18,
    help="Set FPS"
    )
parser.add_argument(
    '-res', '--resolution', type = int, default = 1080,
    help="Set Camera resolution"
    )
parser.add_argument(
    '-foc', '--lens_focus', type = int, default = -1,
    help="Set lens focus (-1 is AutoFocusMode)"
    )

args = parser.parse_args()

FPS = args.frames_per_second

STREAM_DURATION = args.stream_duration
EARLY_STOP = 0.75
LENS_FOCUS = args.lens_focus
OUTPUT_PATH = args.output_path
IMG_QUALITY = args.image_quality
CONF = args.confidence_threshold
BLUR_THRESHOLD = args.blur_threshold
RES = args.resolution 

BLOB_NAME, WI, HE = 'pedestrian-detection-adas-0002', 672, 384
# BLOB_NAME, WI, HE, FPS = 'mobilenet-ssd', 300, 300, 30

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
        

def box_the_frame(frame, detections_list):
    for det in detections_list:
        bbox = frameNorm(frame, (det[0],det[1],det[2],det[3]))
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.putText(frame, f"{int(det[4] * 100)}%", (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
    return frame 

def store_data(frame, detections_list, stream_h, output_path, img_quality):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.imwrite(output_path + timestamp + '_raw.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), img_quality])
    if detections_list != []:
        box_the_frame(frame, detections_list)
        cv2.imwrite(output_path + timestamp + '_boxed.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), img_quality])
        with open(output_path + timestamp + '_detections.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in range(0, len(detections_list)):
                spamwriter.writerow(detections_list[i])
    with open(output_path + timestamp + '_stream_h.pkl', 'wb') as f:
        pickle.dump(stream_h, f)
    return timestamp

def get_blurScore(frame):
    return cv2.Laplacian(frame, cv2.CV_64F).var() 

def is_clear(frame, threshold):
    return cv2.Laplacian(frame, cv2.CV_64F).var() > threshold

class stream_history:
    def __init__(self, detection_number = [], blur_score = []):
        self.detection_number = detection_number
        self.blur_score = blur_score
    
    def update(self, frame, detections_list):
        self.detection_number.append(len(detections_list))
        self.blur_score.append(get_blurScore(frame))
        


# ==============================================================================
# Setting color camera and neural network
# ==============================================================================

# Pipeline tells DepthAI what operations to perform when running
pipeline = depthai.Pipeline()

# Initialize color cam
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(WI, HE) # C'est ce que le NN me demande
cam_rgb.setInterleaved(False)

cam_rgb.setFps(FPS)

if RES == 1080:
    cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
    N_SHAVES = 6
elif RES == 2160:
    cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_4_K)
    N_SHAVES = 5
elif RES == 3040:
    cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_12_MP)
    N_SHAVES = 5
else :
    print('Input resolution is invalid: Setting 1080P')
    cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
    N_SHAVES = 6

if args.flip :
    cam_rgb.setImageOrientation(depthai.CameraImageOrientation.HORIZONTAL_MIRROR)


# Initialize Neural network (which model to choose, confidence_threshold, etc.)
detection_nn = pipeline.createMobileNetDetectionNetwork()
#detection_nn.setBlobPath(str(blobconverter.from_zoo(name=BLOB_NAME, shaves=N_SHAVES)))

detection_nn.setBlobPath(HOME_PATH + '/projets/Camera-PMR-2/resources/blobs/'+ BLOB_NAME + '_openvino_2021.4_' + str(N_SHAVES) + 'shave.blob')


# nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setConfidenceThreshold(CONF)

# Outputs + Linking
cam_rgb.preview.link(detection_nn.input)
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Control the camera
controlIn = pipeline.createXLinkIn()
controlIn.setStreamName('control')
controlIn.out.link(cam_rgb.inputControl)


# ==============================================================================
# Pedestrian detection (device is running, sending data via XLink)
# ==============================================================================
print('=== Starting pipeline =================================================')
with depthai.Device(pipeline) as device:
    # To consume device results, output two queues from the device
    controlQueue = device.getInputQueue('control')
    ctrl = depthai.CameraControl()
    if LENS_FOCUS == -1:
        ctrl.setAutoFocusMode(depthai.RawCameraControl.AutoFocusMode.MACRO)
    else: 
        ctrl.setManualFocus(LENS_FOCUS)
    controlQueue.send(ctrl)

    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)

    # Init the loop
    frame, frame_backup = None, None
    time_init = time.time()
    stream_h = stream_history() 
    while True:
        detections_list = []
        frame = q_rgb.get().getCvFrame()
        in_nn = q_nn.get()
        frame_is_clear = is_clear(frame, BLUR_THRESHOLD)
        
        if frame_backup is None and frame_is_clear:
            frame_backup = copy.deepcopy(frame)

        if in_nn is not None:
            detections = in_nn.detections
            detections_list = [[det.xmin, det.ymin, det.xmax, det.ymax, det.confidence] for det in detections]
        
        # Stream history
        stream_h.update(frame, detections_list)
       
       # Show the frames lives (by passing -sho)
        if args.show_output:
            if detections_list != []:
                boxed_frame = copy.deepcopy(frame)
                box_the_frame(boxed_frame, detections_list)
                cv2.imshow('Ouput', boxed_frame)
            else: 
                print('ye')
                cv2.imshow('Output', frame)
        
        # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
        if cv2.waitKey(1) == ord('q'):
            break

        # Early break if a detection on a clear frame is found
        if time.time() - time_init > EARLY_STOP * STREAM_DURATION: 
            if frame_is_clear and detections_list != []:
                break
        
        # Break if maximal stream duration is reached
        if time.time() - time_init > STREAM_DURATION:
            if not frame_is_clear and frame_backup is not None:
                frame = frame_backup
                frame_is_clear = True
            else:
                print('Warning: No clear frame was obtained')
            break
        

# ==============================================================================
# Post-pipeline treatment
# ==============================================================================
# save files
timestamp = store_data(frame, detections_list, stream_h, OUTPUT_PATH, IMG_QUALITY)

# log
print('=======================================================================')
print('=== Over: ' + timestamp + ' =========================================')
print('=======================================================================')
print('Resolution: ' + str(RES))
print('Confidence threshold: ' + str(CONF) + ' & blur threshold: ' + str(BLUR_THRESHOLD))
print('Blur score: ' + str(get_blurScore(frame)))
print('Frame is clear: ' + str(frame_is_clear))
print('Number of detections: ' + str(len(detections_list)))
print('=======================================================================')

# Show output
if args.show_output:
    import matplotlib.pyplot as plt
    y1, y2 = stream_h.detection_number, [b_s/BLUR_THRESHOLD for b_s in stream_h.blur_score]
    x = range(0,len(y1))
    plt.scatter(x, y1)
    plt.scatter(x, y2)
    plt.show()
    
    if detections_list != []:
        cv2.imshow('Boxed frame kept', boxed_frame)
    else: 
        cv2.imshow('Frame kept', frame)
    

