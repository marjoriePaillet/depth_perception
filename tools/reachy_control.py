import cv2
import depthai as dai
import numpy as np
import math
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
import pyzbar.pyzbar as pyzbar

reachy = ReachySDK(host='localhost')
stepSize = 0.005
newConfig = False

f_estimation = open("estime.txt", "a")
f_real = open("real.txt", "a")

pipeline = dai.Pipeline()

# Define sources and outputs
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
stereo.rectifiedRight.link(xoutBw.input)

monoHFOV = 83

depthWidth = 1280
Start = False
rec = False


def calc_angle(offset, fov, depth):
    return math.atan(math.tan(fov / 2.0) * offset / (depth / 2.0))


def decode(im):
    # Find barcodes and QR codes
    decodedObjects = pyzbar.decode(im)

    # Print results
    for obj in decodedObjects:
        if obj.data == b'gripper':
            points = obj.polygon
            # If the points do not form a quad, find convex hull
            if len(points) > 4:
                hull = cv2.convexHull(np.array([point for point in points],
                                      dtype=np.float32))
                hull = list(map(tuple, np.squeeze(hull)))
            else:
                hull = points

            return hull
        else:
            return None


def compute_z(im, center, r):

    xmin = int(center[0]-r)-10
    ymin = int(center[1]-r)-10
    xmax = int(center[0]+r)+10
    ymax = int(center[1]+r)+10

    roi = im[int(ymin):int(ymax), int(xmin):int(xmax)]

    flatRoi = roi.flatten()
    result = list(filter(lambda val: val !=  0, list(flatRoi)))

    z = np.median(np.array(result))
    return z, [xmin, ymin, xmax, ymax]


# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 20
params.maxThreshold = 150

# Filter by Area.
params.filterByArea = True
params.minArea = 120
params.maxArea = 400

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.2

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)


with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
    qBw = device.getOutputQueue(name="bw", maxSize=4, blocking=False)

    color = (255, 255, 255)
    print("Use WASD keys to move ROI!")
    i = 0
    V = 0
    desired_pos = None
    real_pos = None
    Z = None
    p = False
    while True:
        inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived
        inBw = qBw.get()
        depthFrame = inDepth.getFrame()
        bwFrame = inBw.getCvFrame()

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        # Detect blobs
        keypoints = detector.detect(bwFrame)

        if len(keypoints) > 0:
            z, point = compute_z(depthFrame, (keypoints[0].pt[0], keypoints[0].pt[1]), keypoints[0].size/2)
            real = reachy.r_arm.forward_kinematics()
            xmin = point[0]
            ymin = point[1]
            xmax = point[2]
            ymax = point[3]

            deltaX = int((xmax - xmin) * 0.1)
            deltaY = int((ymax - ymin) * 0.1)
            bbox = np.zeros(4)
            bbox[0] = max(xmin + deltaX, 0)
            bbox[1] = max(ymin + deltaY, 0)
            bbox[2] = min(xmax - deltaX, depthFrame.shape[1])
            bbox[3] = min(ymax - deltaY, depthFrame.shape[0])

            centroidX = int((bbox[2] - bbox[0]) / 2) + bbox[0] # -40
            # centroidX = centroidX - centroidX*0.0390625
            centroidY = int((bbox[3] - bbox[1]) / 2) + bbox[1]

            midy = int(depthFrame.shape[0] / 2)
            midx = int((depthFrame.shape[1]) / 2)

            bb_x_pos = centroidX - midx
            bb_y_pos = centroidY - midy

            angle_x = calc_angle(bb_x_pos, np.deg2rad(79.31), 1280)
            angle_y = calc_angle(bb_y_pos, np.deg2rad(55.12), 800)

            x = z*math.tan(angle_x)
            y = -z*math.tan(angle_y)

            cv2.rectangle(depthFrameColor, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            cv2.rectangle(bwFrame, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

            if p:
                print(f'x = {x}, y = {y}, z = {z}')

            if Start is True:
                R = np.array([[0, np.sin(np.pi/4), np.cos(np.pi/4), 0.0825],
                             [-1, 0, 0, 0.105],
                             [0, np.cos(np.pi/4), -np.sin(np.pi/4), -0.045],
                             [0, 0, 0, 1]])
                v = np.array([x/1000,
                             y/1000,
                             z/1000,
                             1])[:, np.newaxis]
                V = np.dot(R, v)
                f_estimation.write(f'{round(V[0][0],4)}, {round(V[1][0],4)}, {round(V[2][0],4)}\n')
                f_real.write(f'{round(real[0][3], 4)}, {round(real[1][3], 4)}, {round(real[2][3], 4)}\n')
                Start = False

        cv2.imshow("depth", depthFrameColor)
        cv2.imshow("bw", bwFrame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            reachy.turn_off_smoothly('r_arm')
            f_estimation.close()
            f_real.close()
            break

        if key == ord('s'):
            if Start is True:
                Start = False
            else:
                Start = True
                # t = move()
                # t.start()
        if key == ord('r'):
            print(reachy.joints)

        if key == ord('p'):
            if p is True:
                p = False
            else:
                p = True
        # elif key == ord('w'):
        #     if topLeft.y - stepSize >= 0:
        #         topLeft.y -= stepSize
        #         bottomRight.y -= stepSize
        #         newConfig = True
        # elif key == ord('a'):
        #     if topLeft.x - stepSize >= 0:
        #         topLeft.x -= stepSize
        #         bottomRight.x -= stepSize
        #         newConfig = True
        # elif key == ord('s'):
        #     if bottomRight.y + stepSize <= 1:
        #         topLeft.y += stepSize
        #         bottomRight.y += stepSize
        #         newConfig = True
        # elif key == ord('d'):
        #     if bottomRight.x + stepSize <= 1:
        #         topLeft.x += stepSize
        #         bottomRight.x += stepSize
        #         newConfig = True
        # elif key == ord('r'):
        #     f_estimation = open("data.txt", "a")
        #     f_estimation.write(str(int(depthData.spatialCoordinates.z))+"\n")
        #     f_estimation.close()

        # if newConfig:
        #     config.roi = dai.Rect(topLeft, bottomRight)
        #     cfg = dai.SpatialLocationCalculatorConfig()
        #     cfg.addROI(config)
        #     spatialCalcConfigInQueue.send(cfg)
        #     newConfig = False