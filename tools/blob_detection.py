"""Code to test blob detection parameters with depthAI device"""

import cv2
import numpy as np
import depthai as dai

pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

xoutBw = pipeline.createXLinkOut()
xoutBw.setStreamName("bw")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
stereo.setConfidenceThreshold(250)

stereo.setLeftRightCheck(True)
stereo.setSubpixel(False)
stereo.setExtendedDisparity(False)

# linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.rectifiedRight.link(xoutBw.input)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 100
params.maxThreshold = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 800
# params.maxArea = 10000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8
# params.maxCircularity = 0.785

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.2
params.maxInertiaRatio = 1

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

with dai.Device(pipeline) as device:

    qBw = device.getOutputQueue(name="bw", maxSize=4, blocking=False)

    while(1):
        inBw = qBw.get()
        bwFrame = inBw.getCvFrame()

        keypoints = detector.detect(bwFrame)

        im_with_keypoints = cv2.drawKeypoints(bwFrame, keypoints, np.array([]),
                                              (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("Keypoints", im_with_keypoints)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
