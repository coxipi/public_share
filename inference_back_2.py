# first, import all necessary modules
import sys
from pathlib import Path

import blobconverter
# import copy  
import csv
import cv2
import datetime
import depthai
import numpy as np
import time
import os
import argparse
import copy



# ==============================================================================
# Functions and parameters
# ==============================================================================

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

def store_data(frame, detections_list, output_path, img_quality):
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
    return timestamp

def blur_score(frame):
    return cv2.Laplacian(frame, cv2.CV_64F).var() 

def is_blurry(frame, threshold):
    return cv2.Laplacian(frame, cv2.CV_64F).var() < threshold

BLOB_NAME, WI, HE = 'pedestrian-detection-adas-0002', 672, 384
#BLOB_NAME, WI, HE = 'mobilenet-ssd', 300, 300
#BLOB_NAME, WI, HE = 'person-detection-retail-0013', 544, 320


#OUTPUT_PATH = os.path.abspath( os.path.dirname( __file__ ) )  + '/output/tests/'
#OUTPUT_PATH = 'output/'
SHOW_OUTPUT = True
KEEP_MAX_DET = False
WAIT_TIME = 1000

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









# ==============================================================================
# Setting color camera, neural network, camera control
# ==============================================================================

# Pipeline tells DepthAI what operations to perform when running
pipeline = depthai.Pipeline()

# Initialize color cam
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(WI, HE) # C'est ce que le NN me demande
# cam_rgb.setVideoSize(WI, HE)
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
# cam_rgb.setImageOrientation(depthai.CameraImageOrientation.VERTICAL_FLIP)


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
    if LENS_FOCUS != -1:
        controlQueue = device.getInputQueue('control')
        ctrl = depthai.CameraControl()
        ctrl.setManualFocus(LENS_FOCUS)
        controlQueue.send(ctrl)

    # To consume device results, output two queues from the device
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)


    # Init the loop
    frame_raw = None
    detections, detections_keep = [], []
    time_init = time.time()    
    # Dump first seconds

    frame_backup = None
    while True:
        detections_list = []
        frame_raw = q_rgb.get().getCvFrame()
        in_nn = q_nn.get()
        #in_nn = q_nn.get()
        # in_rgb = q_rgb.tryGet()
        # if in_rgb is not None:
        #     # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
        #     frame = in_rgb.getCvFrame()
        
        if is_blurry(frame_raw, BLUR_THRESHOLD) == False:
            
            # Keep at least one clear frame as a backup
            if frame_backup is None:
                frame_backup = frame_raw
        
            if in_nn is not None:
                # when data from nn is received, we take the detections array that contains the blob results
                detections = in_nn.detections
                detections_list = [[det.xmin, det.ymin, det.xmax, det.ymax, det.confidence] for det in detections]
                if args.show_output and detections_list != []:
                    frame_boxed = copy.deepcopy(frame_raw)
                    box_the_frame(frame_boxed, detections_list)
        
        if args.show_output:
            if detections_list != []:
                cv2.imshow('test', frame_boxed)
            else: 
                cv2.imshow('test', frame_raw)
        
        # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
        if cv2.waitKey(1) == ord('q'):
            break
        
        # A detection is obtained and the frame is clear ; BREAK
        if detections != []:
                break
        
        # No detection obtained, but stream is over ; BREAK
        if time.time() - time_init > STREAM_DURATION :
            if frame_backup is not None:
                frame_raw = frame_backup
        else:
            print("Warning: No clear frame obtained during stream!" )
        break
        
        

# ==============================================================================
# Post-treatment, after pipeline
# ==============================================================================

timestamp = store_data(frame_raw, detections_list, OUTPUT_PATH, IMG_QUALITY)


if args.show_output:
    if detections_list != []:
        cv2.imshow('test', frame_boxed)
    else: 
        cv2.imshow('test', frame_raw)
    
print('=======================================================================')
print('=== Over: ' + timestamp + ' =========================================')
print('=======================================================================')
print('Confidence threshold: ' + str(CONF) + ' and blur threshold: ' + str(BLUR_THRESHOLD))
print('Blur score: ' + str(blur_score(frame_raw)))
print('Frame is clear:' + str(not is_blurry(frame_raw, BLUR_THRESHOLD)))
print('Number of detections : ' + str(len(detections_list)))
print('=======================================================================')



