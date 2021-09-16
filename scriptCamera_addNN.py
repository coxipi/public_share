#!/usr/bin/env python3

import cv2
import depthai as dai
import datetime
import blobconverter
import numpy as np

# confidence treshold 
CONF = 0.01


# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam = pipeline.createColorCamera()
cam.setPreviewSize(300, 300)  # 300x300 will be the preview frame size, available as 'preview' output of the node
cam.setInterleaved(False)
# cam.setStillSize(300, 300)         # 1st MOD
# cam.setVideoSize(300, 300)        # 2nd MOD


# Script node
script = pipeline.create(dai.node.Script)
script.setScript("""
    ctrl = CameraControl()
    ctrl.setCaptureStill(True)
    node.io['out'].send(ctrl)
""")

# XLinkOut
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName('still')

# Connections
script.outputs['out'].link(cam.inputControl)
cam.still.link(xout.input)


# NN
detection_nn = pipeline.createMobileNetDetectionNetwork()
# Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
# We're using a blobconverter tool to retreive the MobileNetSSD blob automatically from OpenVINO Model Zoo
detection_nn.setBlobPath(str(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6)))
# Next, we filter out the detections that are below a confidence threshold. Confidence can be anywhere between <0..1>
detection_nn.setConfidenceThreshold(CONF)
# Next, we link the camera 'preview' output to the neural network detection input, so that it can produce detections
cam.still.link(detection_nn.input)



# The same XLinkOut mechanism will be used to receive nn results
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Connect to device with pipeline
with dai.Device(pipeline) as device:
    # with the 2nd MOD, the script doesn't get past the next line
    img = device.getOutputQueue("still").get()
    img_nn = device.getOutputQueue("nn").get()
    detections = img_nn.detections
    frame = img.getCvFrame()

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
        
    for detection in detections:
         # for each bounding box, we first normalize it to match the frame size
         bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
         # and then draw a rectangle on the frame to show the actual result
         cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    #After all the drawing is finished, we show the frame on the screen    
    cv2.imshow("preview", frame)
    cv2.waitKey()
