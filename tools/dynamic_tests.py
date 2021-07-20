"""Move Reachy right arm randomly in a virtual box of H=30cm, W=30cm, D=15cm"""

import cv2
import depthai as dai
import numpy as np
import math
import random
import threading
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
import time


class BlobDetection(threading.Thread):
    def __init__(self, name=None, depth=None, min_thresh=0, max_thresh=255,
                 area=False, min_area=None, max_area=None,
                 circularity=False, min_circ=None, max_circ=None,
                 convexity=False, min_conv=None,
                 max_conv=None, Inertia=False, min_in=None, max_in=None,
                 hfov=79.31, vfov=55.12):

        threading.Thread.__init__(self)

        if name is not None:
            self.name = name

        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = min_thresh
        params.maxThreshold = max_thresh

        if area:
            params.filterByArea = True
            if min_area is not None:
                params.minArea = min_area
            if max_area is not None:
                params.maxArea = max_area
        if circularity:
            params.filterByCircularity = True
            if min_circ is not None:
                params.minCircularity = min_circ
            if max_circ is not None:
                params.maxCircularity = max_circ
        if convexity:
            params.filterByConvexity = True
            if min_conv is not None:
                params.minConvexity = min_conv
            if max_conv is not None:
                params.maxConvexity = max_conv
        if Inertia:
            params.filterByInertia = True
            if min_in is not None:
                params.minInertiaRatio = min_in
            if max_in is not None:
                params.maxInertiaRatio = max_in

        self.depth = None
        if depth is not None:
            self.depth = depth

        self.hfov = hfov
        self.vfov = vfov

        self.detector = cv2.SimpleBlobDetector_create(params)
        self.pos = {'im': {'x': None, 'y': None, 'x': None},
                    'robot': {'x': None, 'y': None, 'x': None}}
        self.roi = {'xmin': None, 'ymin': None, 'xmax': None, 'ymax': None}
        self.bbox = {'xmin': None, 'ymin': None, 'xmax': None, 'ymax': None}
        self.keypoints = None
        self.Terminated = False

    def run(self):
        while not self.Terminated:
            if self.depth.bwFrame is not None:
                self.keypoints = self.detector.detect(self.depth.bwFrame)
                if len(self.keypoints) > 0:
                    self.define_roi((self.keypoints[0].pt[0],
                                    self.keypoints[0].pt[1]),
                                    self.keypoints[0].size/2)
                    self.compute_pos_in_camera_frame()
                    self.compute_pos_in_robot_frame()
            else:
                self.keypoints = None
            time.sleep(0.02)

    def stop(self):
        self.Terminated = True

    def define_roi(self, center, r):
        self.roi['xmin'] = int(center[0] - r - 10)
        self.roi['ymin'] = int(center[1] - r - 10)
        self.roi['xmax'] = int(center[0] + r + 10)
        self.roi['ymax'] = int(center[1] + r + 10)

    def calc_angle(self, offset, fov, size):
        return math.atan(math.tan(fov / 2.0) * offset / (size / 2.0))

    def compute_pos_in_camera_frame(self):
        depthFrame = self.depth.depthFrame
        # Compute Z
        roi = depthFrame[self.roi['ymin']:self.roi['ymax'],
                         self.roi['xmin']:self.roi['xmax']]
        flatRoi = roi.flatten()
        result = list(filter(lambda val: val != 0, list(flatRoi)))
        z = np.median(np.array(result))

        # Compute X and Y
        deltaX = int((self.roi['xmax'] - self.roi['xmin']) * 0.3)
        deltaY = int((self.roi['ymax'] - self.roi['ymin']) * 0.3)
        self.bbox['xmin'] = max(self.roi['xmin'] + deltaX, 0)
        self.bbox['ymin'] = max(self.roi['ymin'] + deltaY, 0)
        self.bbox['xmax'] = min(self.roi['xmax'] - deltaX, depthFrame.shape[1])
        self.bbox['ymax'] = min(self.roi['ymax'] - deltaY, depthFrame.shape[0])
        centroidX = int((self.bbox['xmax'] - self.bbox['xmin']) / 2) + self.bbox['xmin']
        centroidY = int((self.bbox['ymax'] - self.bbox['ymin']) / 2) + self.bbox['ymin']

        midx = int(depthFrame.shape[1] / 2)
        midy = int(depthFrame.shape[0] / 2)

        bb_x_pos = centroidX - midx
        bb_y_pos = centroidY - midy

        angle_x = self.calc_angle(bb_x_pos, np.deg2rad(self.hfov),
                                  depthFrame.shape[1])
        angle_y = self.calc_angle(bb_y_pos, np.deg2rad(self.vfov),
                                  depthFrame.shape[0])

        self.pos['im']['x'] = z*math.tan(angle_x)
        self.pos['im']['y'] = -z*math.tan(angle_y)
        self.pos['im']['z'] = z

    def compute_pos_in_robot_frame(self):
        R = np.array([[0, np.sin((3*np.pi)/8), np.cos((3*np.pi)/8), 0.08+0.065*np.sin((np.pi/8)+0.1526)-0.02],
                      [-1, 0, 0, 0.038],
                      [0, np.cos((3*np.pi)/8), -np.sin((3*np.pi)/8), 0.13-0.065*np.cos((np.pi/8)+0.1526)],
                      [0, 0, 0, 1]])
        v = np.array([self.pos['im']['x']/1000,
                     self.pos['im']['y']/1000,
                     self.pos['im']['z']/1000,
                     1])[:, np.newaxis]
        V = np.dot(R, v)
        self.pos['robot']['x'] = round(V[0][0], 4)
        self.pos['robot']['y'] = round(V[1][0], 4)
        self.pos['robot']['z'] = round(V[2][0], 4)


class DepthComputation (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.pipeline = dai.Pipeline()

        # Define nodes
        monoLeft = self.pipeline.createMonoCamera()
        monoRight = self.pipeline.createMonoCamera()
        stereo = self.pipeline.createStereoDepth()

        # Define sources and outputs
        xoutDepth = self.pipeline.createXLinkOut()
        xoutBw = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")
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

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        stereo.depth.link(xoutDepth.input)
        stereo.rectifiedRight.link(xoutBw.input)

        self.bwFrame = None
        self.depthFrame = None
        self.depthFrameColor = None

        self.Terminated = False

    def run(self):
        with dai.Device(self.pipeline) as device:
            depthQueue = device.getOutputQueue(name="depth", maxSize=4,
                                               blocking=False)
            qBw = device.getOutputQueue(name="bw", maxSize=4, blocking=False)

            while not self.Terminated:
                # Blocking calls, will wait until a new data has arrived
                inDepth = depthQueue.get()
                inBw = qBw.get()

                self.depthFrame = inDepth.getFrame()
                self.bwFrame = inBw.getCvFrame()
                depthFrameColor = cv2.normalize(self.depthFrame, None, 255, 0,
                                                cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                self.depthFrameColor = cv2.applyColorMap(depthFrameColor,
                                                         cv2.COLORMAP_HOT)

    def stop(self):
        self.Terminated = True


class DynamicTest (threading.Thread):
    def __init__(self, g):
        threading.Thread.__init__(self)
        self.g = g
        self.reachy = ReachySDK(host='localhost')
        self.Terminated = False

    def stop(self):
        self.Terminated = True

    def run(self):
        f_estimation = open("estime.txt", "a")
        f_real = open("real.txt", "a")
        f_commande = open("commande.txt", "a")

        xmin = 16
        xmax = 44
        ymin = -19
        ymax = 9
        zmin = -34
        zmax = -19

        datas = []
        datas_id = []

        i = 0

        # load gripper orientation depending on spatial location
        with open('datas.npy', 'rb') as f:
            datas = np.load(f)
            datas = list(datas)  # list or orientation

        with open('datas_id.npy', 'rb') as f_id:
            datas_id = np.load(f_id)  # 3D matrix of list indexes refering to spatial positions

        while(i < 50 and self.Terminated is False):
            x = random.randrange(xmin, xmax)/100
            y = random.randrange(ymin, ymax)/100
            z = random.randrange(zmin, zmax)/100

            ind_x = int((x*100 - 15)/10)
            ind_y = int(-(y*100 - 10)/10)
            ind_z = int(-(z*100 + 20)/5)

            M = datas[int(datas_id[ind_z][ind_y][ind_x])]
            M[0][3] = x
            M[1][3] = y
            M[2][3] = z

            j = self.reachy.r_arm.inverse_kinematics(M)

            self.reachy.turn_on('r_arm')
            goto({joint: pos for joint, pos in zip(self.reachy.r_arm.joints.values(), j)},
                 duration=1, interpolation_mode=InterpolationMode.MINIMUM_JERK)
            time.sleep(1)
            desired_pos = self.reachy.r_arm.forward_kinematics()

            posx = self.g.pos['robot']['x']
            posy = self.g.pos['robot']['y']
            posz = self.g.pos['robot']['z']
            f_estimation.write(f'{round(posx, 4)}, {round(posy, 4)}, {round(posz, 4)}\n')
            f_real.write(f'{round(desired_pos[0][3], 4)}, {round(desired_pos[1][3], 4)}, {round(desired_pos[2][3], 4)}\n')
            f_commande.write(f'{round(M[0][3], 4)}, {round(M[1][3], 4)}, {round(M[2][3], 4)}\n')
            i += 1
            print('step = ' + str(i))


depth_comput = DepthComputation()
depth_comput.start()

gripper_detection = BlobDetection(name='gripper', depth=depth_comput,
                                  min_thresh=100, max_thresh=255,
                                  area=True, min_area=800,
                                  circularity=True, min_circ=0.8,
                                  convexity=True, min_conv=0.87,
                                  Inertia=True, min_in=0.2)
gripper_detection.start()

reachy_control = DynamicTest(gripper_detection)

blob_threads = [gripper_detection]

while True:
    if depth_comput.depthFrameColor is not None:
        im_with_keypoints = depth_comput.depthFrameColor
        if gripper_detection.bbox['ymax'] is not None:
            cv2.rectangle(im_with_keypoints, (gripper_detection.bbox['xmin'],
                          gripper_detection.bbox['ymin']),
                          (gripper_detection.bbox['xmax'],
                          gripper_detection.bbox['ymax']), (255, 0, 0),
                          cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

        cv2.imshow("Keypoints", im_with_keypoints)
    key = cv2.waitKey(1)

    if key == ord('q'):
        reachy_control.reachy.turn_off_smoothly('r_arm')
        reachy_control.stop()
        gripper_detection.stop()
        depth_comput.stop()
        break
    elif key == ord('s'):
        reachy_control.start()
