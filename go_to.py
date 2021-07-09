import cv2
import numpy as np
import depthai as dai
import threading
import time
import math
import copy
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
from scipy.spatial.transform import Rotation as R

class BlobDetection (threading.Thread):
    def __init__(self, name = None, depth=None, min_thresh=0, max_thresh=255, area=False, min_area=None, max_area=None,
                 circularity=False, min_circ=None, max_circ=None, convexity=False, min_conv=None,
                 max_conv=None, Inertia=False, min_in=None, max_in=None, hfov=79.31, vfov=55.12):
        threading.Thread.__init__(self)

        if name is not None: self.name = name

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
        
        self.depth = None
        if depth is not None: self.depth = depth

        self.detector = cv2.SimpleBlobDetector_create(params)
        self.pos = {'im': {'x': None, 'y': None, 'x': None}, 
                    'robot': {'x': None, 'y': None, 'x': None}}
        self.roi = {'xmin': None, 'ymin': None, 'xmax': None, 'ymax': None}
        self.keypoints = None
        self.Terminated = False

        self.hfov = hfov
        self.vfov = vfov
        
    def run(self):
        # global bwFrame
        while not self.Terminated:
            if self.depth.bwFrame is not None:
                self.keypoints = self.detector.detect(self.depth.bwFrame)
                if len(self.keypoints) > 0:
                    self.define_roi((self.keypoints[0].pt[0], self.keypoints[0].pt[1]), self.keypoints[0].size/2)
                    self.compute_pos_in_frame()
                    self.compute_pos_in_robot_frame()
            else:
                self.keypoints = None
            time.sleep(0.01)
    
    def stop(self):
        self.Terminated = True

    def define_roi(self, center, r):
        self.roi['xmin'] = int(center[0] - r - 10)
        self.roi['ymin'] = int(center[1] - r - 10)
        self.roi['xmax'] = int(center[0] + r + 10)
        self.roi['ymax'] = int(center[1] + r + 10)

    def calc_angle(self, offset, fov, size):
        return math.atan(math.tan( fov/ 2.0) * offset / (size / 2.0))
    
    def compute_pos_in_frame(self):
        depthFrame = self.depth.depthFrame
        # Compute Z
        roi = depthFrame[self.roi['ymin']:self.roi['ymax'], self.roi['xmin']:self.roi['xmax']]
        flatRoi = roi.flatten()
        result = list(filter(lambda val: val !=  0, list(flatRoi)))
        z = np.median(np.array(result))

        # Compute X and Y
        deltaX = int((self.roi['xmax'] - self.roi['xmin'] ) * 0.1)
        deltaY = int((self.roi['ymax'] - self.roi['ymin'] ) * 0.1)
        bbox = np.zeros(4)
        bbox[0] = max(self.roi['xmin'] + deltaX, 0)
        bbox[1] = max(self.roi['ymin'] + deltaY, 0)
        bbox[2] = min(self.roi['xmax'] - deltaX, depthFrame.shape[1])
        bbox[3] = min(self.roi['ymax'] - deltaY, depthFrame.shape[0])

        centroidX = int((bbox[2] - bbox[0]) / 2) + bbox[0] #- 40
        centroidY = int((bbox[3] - bbox[1]) / 2) + bbox[1]

        midx = int(depthFrame.shape[1] / 2)
        midy = int(depthFrame.shape[0] / 2) 

        bb_x_pos = centroidX - midx
        bb_y_pos = centroidY - midy  

        angle_x = self.calc_angle(bb_x_pos, np.deg2rad(self.hfov), depthFrame.shape[1])
        angle_y = self.calc_angle(bb_y_pos, np.deg2rad(self.vfov), depthFrame.shape[0])

        self.pos['im']['x'] = z*math.tan(angle_x)
        self.pos['im']['y'] = -z*math.tan(angle_y)
        self.pos['im']['z'] = z

    def compute_pos_in_robot_frame(self):
        R = np.array([[0, np.sin(np.pi/4), np.cos(np.pi/4), 0.0825],
                                [-1, 0, 0, 0.105],
                                [0, np.cos(np.pi/4), -np.sin(np.pi/4), -0.045],
                                [0, 0, 0, 1]])
        v = np.array([self.pos['im']['x']/1000,
                        self.pos['im']['y']/1000,
                        self.pos['im']['z']/1000,
                        1])[:,np.newaxis]
        V = np.dot(R, v)                 
        self.pos['robot']['x'] = round(V[0][0],4)
        self.pos['robot']['y'] = round(V[1][0],4)
        self.pos['robot']['z'] = round(V[2][0],4)
       


class DepthComputation (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.pipeline = dai.Pipeline()# Define sources and outputs

        monoLeft = self.pipeline.createMonoCamera()
        monoRight = self.pipeline.createMonoCamera()
        stereo = self.pipeline.createStereoDepth()
        spatialLocationCalculator = self.pipeline.createSpatialLocationCalculator()

        xoutDepth = self.pipeline.createXLinkOut()
        xoutSpatialData = self.pipeline.createXLinkOut()
        xinSpatialCalcConfig = self.pipeline.createXLinkIn()
        xoutBw = self.pipeline.createXLinkOut()
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

        self.bwFrame = None
        self.depthFrame = None
        self.depthFrameColor = None

        self.Terminated = False
        
    def run(self):
        with dai.Device(self.pipeline) as device:    # Output queue will be used to get the depth frames from the outputs defined above
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
            spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")    
            qBw = device.getOutputQueue(name="bw", maxSize=4, blocking=False)
            
            while not self.Terminated:
                inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived        
                inBw = qBw.get()
                self.depthFrame = inDepth.getFrame()
                self.bwFrame = inBw.getCvFrame()
                depthFrameColor = cv2.normalize(self.depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                self.depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)   

    def stop(self):
        self.Terminated = True


class ReachyControl (threading.Thread):
    def __init__(self, g, t):
        threading.Thread.__init__(self)
        self.g = g
        self.t = t
        self.key = None
        self.reachy = ReachySDK(host='localhost')
        self.Terminated = False
  
        self.ref = None
        self.Start = False

    def run (self):
        while not self.Terminated:

            if self.Start:
                if self.ref is None:
                    self.ref = copy.deepcopy(self.g.pos['robot'])
                if self.compare_to_ref():
                    r = self.compute_rotation()
                    c = self.reachy.r_arm.forward_kinematics()
                    c[:3,:3]=r
                    joint_pos = self.reachy.r_arm.inverse_kinematics(c)
                    print(joint_pos)
                    self.reachy.turn_on('r_arm')
                    goto({joint: pos for joint,pos in zip(self.reachy.r_arm.joints.values(), joint_pos)}, duration=0.25)
                time.sleep(0.25)
         
            if self.key == ord('s'):
                if self.Start == True:
                    self.Start = False
                else:
                    self.Start = True

    def compare_to_ref(self):
        current = self.g.pos['robot']
        if abs(current['x']-self.ref['x']) < 0.05 and abs(current['y']-self.ref['y']) < 0.05 and abs(current['z']-self.ref['z']) < 0.05 :
            self.ref = copy.deepcopy(current)
            return True
        return False

    def stop(self):
        self.Terminated = True


    def compute_rotation(self):
        # print(t.pos['robot'])
        # print(g.pos['robot'])
        dx = self.t.pos['robot']['x']-self.g.pos['robot']['x']
        dy = self.t.pos['robot']['y']-self.g.pos['robot']['y']
        theta = np.arctan(dy/dx)
        if dx<0 and dy>0:
            theta = np.pi + theta
        elif dx<0 and dy<0:
            theta = theta - np.pi 
        print(np.rad2deg(theta))
        R1 = R.from_euler('y', np.deg2rad(-90))
        R2 = R.from_euler('x', theta)
        r = R1 * R2
        return r.as_matrix()



depth_comput = DepthComputation()
depth_comput.start()

gripper_detection = BlobDetection(name='gripper', depth=depth_comput, min_thresh=20, max_thresh=150, area=True, min_area=120, max_area=400,
                circularity=True, min_circ=0.5, convexity=True, min_conv=0.87,
                Inertia=True, min_in=0.2)
gripper_detection.start()

target_detection = BlobDetection(name='target', depth=depth_comput, min_thresh=20, max_thresh=150, area=True, min_area=800, max_area=3000,
                circularity=True, min_circ=0.65, max_circ=0.785)
target_detection.start()

reachy_control = ReachyControl(gripper_detection, target_detection)
reachy_control.start()

blob_threads = [gripper_detection, target_detection]


while True:

    if depth_comput.depthFrameColor is not None:
        im_with_keypoints = depth_comput.depthFrameColor
        # im_with_keypoints = cv2.drawKeypoints(depth_comput.depthFrameColor, gripper_detection.keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, target_detection.keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        for thread in blob_threads:
            if thread.roi['ymax'] is not None:
                cv2.rectangle(im_with_keypoints, (thread.roi['xmin'], thread.roi['ymin']), (thread.roi['xmax'], thread.roi['ymax']), (255,0,0), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
        
        cv2.imshow("Keypoints", im_with_keypoints)
    reachy_control.key = cv2.waitKey(1)

    if reachy_control.key == ord('q'):
        reachy_control.reachy.turn_off_smoothly('r_arm')
        reachy_control.stop()
        gripper_detection.stop()
        target_detection.stop()
        depth_comput.stop()
        break

   