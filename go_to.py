import cv2
import numpy as np
import depthai as dai
import threading
import time

class BlobDetection (threading.Thread):
    def __init__(self, min_thresh=0, max_thresh=255, area=False, min_area=None, max_area=None,
                 circularity=False, min_circ=None, max_circ=None, convexity=False, min_conv=None,
                 max_conv=None, Inertia=False, min_in=None, max_in=None ):
        threading.Thread.__init__(self)
        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = min_thresh
        params.maxThreshold = max_thresh

        if area:
            params.filterByArea = True
            if min_area is not None: params.minArea = min_area
            if max_area is not None: params.maxArea = max_area
        if circularity:
            params.filterByCircularity = True
            if min_circ is not None: params.minCircularity = min_circ
            if max_circ is not None: params.maxCircularity = max_circ
        if convexity:
            params.filterByConvexity = True
            if min_conv is not None: params.minConvexity = min_conv
            if max_conv is not None: params.maxConvexity = max_conv
        if Inertia:
            params.filterByInertia = True
            if min_in is not None: params.minInertiaRatio = min_in
            if max_in is not None: params.maxInertiaRatio = max_in

        self.detector = cv2.SimpleBlobDetector_create(params)
        self.pos = {'im': {'x': None, 'y': None, 'x': None}, 
                    'robot': {'x': None, 'y': None, 'x': None}}
        self.keypoints = None
        
    def run(self):
        global bwFrame
        while(1):
            if bwFrame is not None:
                self.keypoints = self.detector.detect(bwFrame)
            time.sleep(0.001)
pipeline = dai.Pipeline()# Define sources and outputs

monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

xoutDepth = pipeline.createXLinkOut()
xoutSpatialData = pipeline.createXLinkOut()
xinSpatialCalcConfig = pipeline.createXLinkIn()
xoutBw = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
xoutBw.setStreamName("bw")

# Properties

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
stereo.setConfidenceThreshold(250)

stereo.setLeftRightCheck(True)
# stereo.setRectifyMirrorFrame(True)
# stereo.setDepthAlign(dai.StereoDepthProperties.DepthAlign.RECTIFIED_RIGHT)
# stereo.setDepthAlign(dai.CameraBoardSocket.RIGHT)
stereo.setSubpixel(False)
stereo.setExtendedDisparity(False)

# Config
topLeft = dai.Point2f(0.45, 0.45)
bottomRight = dai.Point2f(0.55, 0.55)
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(topLeft, bottomRight)
spatialLocationCalculator.setWaitForConfigInput(False)

spatialLocationCalculator.initialConfig.addROI(config)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)
spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)
# monoRight.out.link(xoutBw.input)
stereo.rectifiedRight.link(xoutBw.input)

bwFrame = None

with dai.Device(pipeline) as device:    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")    
    qBw = device.getOutputQueue(name="bw", maxSize=4, blocking=False)

    gripper_detection = BlobDetection(min_thresh=20, max_thresh=150, area=True, min_area=120, max_area=400,
                 circularity=True, min_circ=0.5, convexity=True, min_conv=0.87,
                 Inertia=True, min_in=0.2)
    gripper_detection.setDaemon(True)
    gripper_detection.start()

    target_detection = BlobDetection(min_thresh=20, max_thresh=150, area=True, min_area=800, max_area=3000,
                 circularity=True, min_circ=0.65, max_circ=0.785)
    target_detection.setDaemon(True)
    target_detection.start()
    
    while True:
        inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived        
        inBw = qBw.get()
        depthFrame = inDepth.getFrame()
        bwFrame = inBw.getCvFrame()

        if gripper_detection.keypoints is not None and target_detection.keypoints is not None:
            im_with_keypoints = cv2.drawKeypoints(bwFrame, gripper_detection.keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, target_detection.keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            cv2.imshow("Keypoints", im_with_keypoints)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


