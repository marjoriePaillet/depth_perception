import cv2
import depthai as dai
import numpy as np
import math
import threading
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
import time


# reachy = ReachySDK(host='localhost')
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

stereo.setLeftRightCheck(False)
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
monoRight.out.link(xoutBw.input)

monoHFOV = 83

depthWidth = 1280
Start = False

class move (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        reachy = ReachySDK(host='localhost')
        time.sleep(0.5)
        recorded_joints = [
            reachy.l_arm.l_shoulder_pitch,
            reachy.l_arm.l_shoulder_roll,
            reachy.l_arm.l_arm_yaw,
            reachy.l_arm.l_elbow_pitch,
            reachy.l_arm.l_forearm_yaw,
            reachy.l_arm.l_wrist_pitch,
            reachy.l_arm.l_wrist_roll,
        ]
        trajectories = [[-2, 2.53, -34.86, -70.02, -1.91, -13.23, -10.12, 54.33], [-23.98, 1.3, -36.7, -52.97, -13.93, -16.57, -28.89, 54.33], [-28.72, -12.77, -53.23, -50.13, -4.55, -16.31, -19.21, 54.91], [-18.35, -4.24, -61.32, -65.27, 11.88, -6.64, -30.94, 54.33]]
        A_point = dict(zip(recorded_joints, trajectories[0]))
        B_point = dict(zip(recorded_joints, trajectories[1]))
        C_point = dict(zip(recorded_joints, trajectories[2]))
        D_point = dict(zip(recorded_joints, trajectories[3]))
        # Goes to the start of the trajectory in 3s
        reachy.turn_on('l_arm')

        goto(A_point, duration=2.0)
        time.sleep(0.5)
        goto(B_point, duration=2.0)
        time.sleep(0.5)
        goto(C_point, duration=2.0)
        time.sleep(0.5)
        goto(D_point, duration=2.0)
        time.sleep(0.5)

        reachy.turn_off_smoothly('l_arm')   

def calc_angle(offset):
    return math.atan(math.tan( monoHFOV/ 2.0) * offset / (depthWidth / 2.0))


# def decode(im) :
#   # Find barcodes and QR codes
#   decodedObjects = pyzbar.decode(im)

#   # Print results
# #   for obj in decodedObjects:
#     # print('Type : ', obj.type)
#     # print('Data : ', obj.data,'\n')

#   return decodedObjects

def compute_z (im, decodedObject):
    points = decodedObject.polygon
    if len(points) > 4 :
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
    else :
        hull = points

    n = len(hull)
        
    hull = np.array(hull)

    xmin = min(hull[:, 0])
    ymin = min(hull[:, 1])
    xmax = max(hull[:, 0])
    ymax = max(hull[:, 1])

    roi = im[int(ymin)-1:int(ymax)+1, int(xmin)-1:int(xmax)+1]

    mask = np.zeros(roi.shape[:2], dtype="uint8")
    pts = np.array([[hull[0][0]-xmin,hull[0][1]-ymin], [hull[1][0]-xmin,hull[1][1]-ymin], [hull[2][0]-xmin,hull[2][1]-ymin], [hull[3][0]-xmin,hull[3][1]-ymin]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(mask, [pts],(255, 255, 255))

    masked = cv2.bitwise_and(roi, roi, mask=mask)
    # print(list(masked))
    flatMask = masked.flatten()
    result = list(filter(lambda val: val !=  0, list(flatMask)))

    z = np.mean(np.array(result))
    # print(z)

    # cv2.imshow("masked", masked)

    return z, hull


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
        # print(type(depthFrame))
        
        ret, depthFrame = cv2.threshold(depthFrame, 470, 0, cv2.THRESH_TOZERO_INV)
        # print(depthFrame)
        # print(type(depthFrame))
        # depthFrameTemp = depthFrame.copy()
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)        
        # spatialData = spatialCalcQueue.get().getSpatialLocations()


        decodedObjects = pyzbar.decode(bwFrame)
        if len(decodedObjects) > 0:
            for dObject in decodedObjects:
                z, hull = compute_z(depthFrame, dObject)
                xmin = int(hull[0][0])
                ymin = int(hull[0][1])
                xmax = int(hull[2][0])
                ymax = int(hull[2][1])

                deltaX = int((xmax - xmin) * 0.1)
                deltaY = int((ymax - ymin) * 0.1)
                bbox = np.zeros(4)
                bbox[0] = max(xmin + deltaX, 0)
                bbox[1] = max(ymin + deltaY, 0)
                bbox[2] = min(xmax - deltaX, depthFrame.shape[1])
                bbox[3] = min(ymax - deltaY, depthFrame.shape[0])

                centroidX = int((bbox[2] - bbox[0]) / 2) + bbox[0]
                centroidY = int((bbox[3] - bbox[1]) / 2) + bbox[1] 

                midy = int(depthFrame.shape[0] / 2) 
                midx = int(depthFrame.shape[1] / 2) 

                bb_x_pos = centroidX - midx
                bb_y_pos = centroidY - midy  

                angle_x = calc_angle(bb_x_pos)
                angle_y = calc_angle(bb_y_pos)  

                x = z*math.tan(angle_x)
                y = -z*math.tan(angle_y)

                cv2.rectangle(depthFrameColor, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                fontType = cv2.FONT_HERSHEY_TRIPLEX
                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
                
                if p:
                    print(f'x = {x}, y = {y}, z = {z}')

                n = len(hull)
                for j in range(0,n):
                    cv2.line(depthFrameColor, tuple(hull[j]), tuple(hull[ (j+1) % n]), (0,255,0), 3)
            
            if Start == True:
                R = np.array([[0, np.sin(np.pi/4), np.cos(np.pi/4), 0.0825],
                                [-1, 0, 0, -0.14],
                                [0, np.cos(np.pi/4), -np.sin(np.pi/4), -0.045],
                                [0, 0, 0, 1]])
                v = np.array([x/1000,
                                y/1000,
                                z/1000,
                                1])[:,np.newaxis]
                V = np.dot(R, v)
                real = reachy.l_arm.forward_kinematics()
                f_estimation.write(f'{round(V[0][0],4)}, {round(V[1][0],4)}, {round(V[2][0],4)}\n')
                print(real)
                f_real.write(f'{round(real[0][3], 4)}, {round(real[1][3], 4)}, {round(real[2][3], 4)}\n')
                print(V)
                print(real)

        cv2.imshow("depth", depthFrameColor)
        cv2.imshow("bw", bwFrame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            reachy.turn_off_smoothly('l_arm')
            f_estimation.close()
            f_real.close()
            break
        
        if key == ord('s'):
            if Start == True:
                Start = False
            else :
                Start = True
                # t = move()
                # t.Start()
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
        # elif key == ord('c'):
        #     R = np.array([[0, np.sin(np.pi/4), np.cos(np.pi/4), 0.0825],
        #                     [-1, 0, 0, -0.14],
        #                     [0, np.cos(np.pi/4), -np.sin(np.pi/4), -0.045],
        #                     [0, 0, 0, 1]])
        #     v = np.array([x/1000,
        #                     y/1000,
        #                     Z/1000,
        #                     1])[:,np.newaxis]
        #     print(f"x:{x}mm, y:{y}mm, z:{Z}mm")  
        #     V = np.dot(R, v)
        #     print(f'coordinates in Reachy frame: {V}')
        #     desired_pos = reachy.l_arm.forward_kinematics()
        #     print(f'desired pos: {desired_pos}')

        # elif key == ord('g'):
        #     A = desired_pos[:4,:3]
        #     M = np.concatenate((A,V),axis=1)
        #     print(f'4x4 target pose: {M}')
        #     joint_pos = reachy.l_arm.inverse_kinematics(M)
        #     print(f'joint pos: {joint_pos}')
        #     print(f'theoric forward kinematics: {reachy.l_arm.forward_kinematics(joint_pos)} ')
        #     reachy.turn_on('l_arm')
        #     goto({reachy.l_arm.l_gripper: 50}, duration=0.5)
        #     goto({joint: pos for joint,pos in zip(reachy.l_arm.joints.values(), joint_pos)}, duration=1.0)
        #     time.sleep(1.1)
        #     goto({reachy.l_arm.l_gripper: 30}, duration=0.5)
        #     print(f'real pos: {reachy.l_arm.forward_kinematics()}')

        # elif key == ord('o'):
        #     reachy.turn_off_smoothly('l_arm')

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
        #     a = reachy.l_arm.forward_kinematics()[:4,:3]
        #     v = np.concatenate((a,v),axis=1)
        #     v = np.dot(R, v)
        #     print(f'coordinates in Reachy frame:\n {v}')
        #     v = reachy.l_arm.inverse_kinematics(v)
        #     print(f'inverse_kinematics of coordinates in Reachy frame:\n {v}')
        #     v = reachy.l_arm.forward_kinematics(v)
        #     print(f'forward_kinematics of coordinates in Reachy frame:\n {v}')



        # if newConfig:
        #     config.roi = dai.Rect(topLeft, bottomRight)
        #     cfg = dai.SpatialLocationCalculatorConfig()
        #     cfg.addROI(config)
        #     spatialCalcConfigInQueue.send(cfg)
        #     newConfig = False