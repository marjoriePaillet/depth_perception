import cv2
import depthai as dai
import numpy as np
import math
import matplotlib.pyplot as plt
import threading
import matplotlib.animation as animation
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
import time
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
from scipy.spatial.transform import Rotation as Rot
import pyzbar.pyzbar as pyzbar

reachy = ReachySDK(host='localhost')
stepSize = 0.005
newConfig = False

f_estimation = open("estime.txt", "a")
f_real = open("real.txt", "a")

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
stereo.setExtendedDisparity(False)# Config

topLeft = dai.Point2f(0.45, 0.45)
bottomRight = dai.Point2f(0.55, 0.55)
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(topLeft, bottomRight)
spatialLocationCalculator.setWaitForConfigInput(False)

spatialLocationCalculator.initialConfig.addROI(config)# Linking

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)
spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)
# monoRight.out.link(xoutBw.input)
stereo.rectifiedRight.link(xoutBw.input)

monoHFOV = 83

depthWidth = 1280
Start = False
rec = False

class move (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        print('start')

    def run(self):
        global reachy, rec
        print('run')
        time.sleep(1)
        # recorded_joints = [
        #     reachy.r_arm.r_shoulder_pitch,
        #     reachy.r_arm.r_shoulder_roll,
        #     reachy.r_arm.r_arm_yaw,
        #     reachy.r_arm.r_elbow_pitch,
        #     reachy.r_arm.r_forearm_yaw,
        #     reachy.r_arm.r_wrist_pitch,
        #     reachy.r_arm.r_wrist_roll,
        # ]
        # trajectories = [[-8.86, -2.70, 33.10, -76.09, -3.37, -16.31, 49.12, -9.46], [-14.75, 16.90, 48.66, -69.67, -4.55, -0.66, 38.27, -1.03]]
        # A_point = dict(zip(recorded_joints, trajectories[0]))
        # B_point = dict(zip(recorded_joints, trajectories[1]))

        # # Goes to the start of the trajectory in 3s
        # reachy.turn_on('r_arm')

        # goto(A_point, duration=1.0,  interpolation_mode=InterpolationMode.LINEAR)
        # rec = True
        # time.sleep(2)
        # goto(B_point, duration=1.0, interpolation_mode=InterpolationMode.LINEAR)
        # time.sleep(2)
        # rec = False

        # reachy.turn_off_smoothly('r_arm')   

def calc_angle(offset, fov, depth):
    return math.atan(math.tan( fov/ 2.0) * offset / (depth / 2.0))


# def decode(im) :
#   # Find barcodes and QR codes
#   decodedObjects = pyzbar.decode(im)

#   # Print results
# #   for obj in decodedObjects:
#     # print('Type : ', obj.type)
#     # print('Data : ', obj.data,'\n')

#   return decodedObjects

def compute_z (im, center, r):

    xmin = int(center[0]-r)-10 
    ymin = int(center[1]-r)-10
    xmax = int(center[0]+r)+10
    ymax = int(center[1]+r)+10

    roi = im[int(ymin):int(ymax), int(xmin):int(xmax)]
    
    # print(list(masked))
    flatRoi = roi.flatten()
    result = list(filter(lambda val: val !=  0, list(flatRoi)))

    z = np.median(np.array(result))
    # z = np.median(roi)
    # print(z)

    # cv2.imshow("masked", roi)

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


with dai.Device(pipeline) as device:    # Output queue will be used to get the depth frames from the outputs defined above
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
    # Start = False
    p = False
    while True:
        inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived        
        inBw = qBw.get()
        depthFrame = inDepth.getFrame()
        bwFrame = inBw.getCvFrame()
        
        # ret, depthFrame = cv2.threshold(depthFrame, 470, 0, cv2.THRESH_TOZERO_INV)
        
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)        
        # spatialData = spatialCalcQueue.get().getSpatialLocations()

        # Detect blobs
        keypoints = detector.detect(bwFrame)
        
        # bwFrame = cv2.drawKeypoints(bwFrame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # depthFrameColor = cv2.drawKeypoints(depthFrameColor, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        if len(keypoints) > 0:
            z, point = compute_z(depthFrame, (keypoints[0].pt[0], keypoints[0].pt[1]), keypoints[0].size/2)
            # real = reachy.r_arm.forward_kinematics()
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

            centroidX = int((bbox[2] - bbox[0]) / 2) + bbox[0] 
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

            cv2.rectangle(depthFrameColor, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            cv2.rectangle(bwFrame, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            
            if p:
                print(f'x = {x}, y = {y}, z = {z}')
            
            if Start == True: #and rec == True:
                R = np.array([[0, np.sin(np.pi/4), np.cos(np.pi/4), 0.0825],
                                [-1, 0, 0, 0.105],
                                [0, np.cos(np.pi/4), -np.sin(np.pi/4), -0.045],
                                [0, 0, 0, 1]])
                v = np.array([x/1000,
                                y/1000,
                                z/1000,
                                1])[:,np.newaxis]
                V = np.dot(R, v)
                # real = reachy.r_arm.forward_kinematics()
                f_estimation.write(f'{round(V[0][0],4)}, {round(V[1][0],4)}, {round(V[2][0],4)}\n')
                # print(real)
                f_real.write(f'{round(real[0][3], 4)}, {round(real[1][3], 4)}, {round(real[2][3], 4)}\n')
                Start = False
                # print(V)
                # print(real)

        
        cv2.imshow("depth", depthFrameColor)
        cv2.imshow("bw", bwFrame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            reachy.turn_off_smoothly('r_arm')
            f_estimation.close()
            f_real.close()
            break
        
        if key == ord('s'):
            if Start == True:
                Start = False
            else :
                Start = True
                # t = move()
                # t.start()
        if key == ord('r'):
            print(reachy.joints)

        if key == ord('p'):
            if p == True:
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
        
        # elif key == ord('z'):
        #     Z = z
        #     print(f'Z saved as: {Z}mm')
        elif key == ord('c'):
            R = np.array([[0, np.sin(np.pi/4), np.cos(np.pi/4), 0.0825],
                            [-1, 0, 0, 0.105],
                            [0, np.cos(np.pi/4), -np.sin(np.pi/4), -0.045],
                            [0, 0, 0, 1]])
            v = np.array([x/1000,
                            y/1000,
                            z/1000,
                            1])[:,np.newaxis]
            print(f"x:{x}mm, y:{y}mm, z:{z}mm")  
            V = np.dot(R, v)
            print(f'coordinates in Reachy frame: {V}')
            desired_pos = reachy.r_arm.forward_kinematics()
            print(f'desired pos: {desired_pos}')

        elif key == ord('g'):
            A = desired_pos[:4,:3]
            M = np.concatenate((A,V),axis=1)
            print(f'4x4 target pose: {M}')
            joint_pos = reachy.r_arm.inverse_kinematics(M)
            print(f'joint pos: {joint_pos}')
            print(f'theoric forward kinematics: {reachy.r_arm.forward_kinematics(joint_pos)} ')
            reachy.turn_on('r_arm')
            goto({reachy.r_arm.l_gripper: 50}, duration=0.5)
            goto({joint: pos for joint,pos in zip(reachy.r_arm.joints.values(), joint_pos)}, duration=1.0)
            time.sleep(1.1)
            goto({reachy.r_arm.l_gripper: 30}, duration=0.5)
            print(f'real pos: {reachy.r_arm.forward_kinematics()}')

        # elif key == ord('o'):
        #     reachy.turn_off_smoothly('r_arm')

        # elif key == ord('p'):
        #     R = np.array([[0, np.sin(np.pi/4), np.cos(np.pi/4), 0.0825],
        #                     [-1, 0, 0, -0.14],
        #                     [0, np.cos(np.pi/4), -np.sin(np.pi/4), -0.045],
        #                     [0, 0, 0, 1]])
        #     v = np.array([x/1000,
        #                     y/1000,
        #                     z/1000,
        #                     1])[:,np.newaxis]
        #     print(f"x:{x}mm, y:{y}mm, z:{z}mm")  
        #     a = reachy.r_arm.forward_kinematics()[:4,:3]
        #     v = np.concatenate((a,v),axis=1)
        #     v = np.dot(R, v)
        #     print(f'coordinates in Reachy frame:\n {v}')
        #     v = reachy.r_arm.inverse_kinematics(v)
        #     print(f'inverse_kinematics of coordinates in Reachy frame:\n {v}')
        #     v = reachy.r_arm.forward_kinematics(v)
        #     print(f'forward_kinematics of coordinates in Reachy frame:\n {v}')



        # if newConfig:
        #     config.roi = dai.Rect(topLeft, bottomRight)
        #     cfg = dai.SpatialLocationCalculatorConfig()
        #     cfg.addROI(config)
        #     spatialCalcConfigInQueue.send(cfg)
        #     newConfig = False