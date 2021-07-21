"""Code to perform depth coomputation and cup grabbing."""

import cv2
import time
import math
import threading
import numpy as np

import depthai as dai

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

from scipy.spatial.transform import Rotation as R


class BlobDetection(threading.Thread):
    """Handle the blob detection and 3D positions computation in
    camera frame and Reachy frame according to passed arguments."""

    def __init__(self, name=None, depth=None, min_thresh=0, max_thresh=255,
                 area=False, min_area=None, max_area=None,
                 circularity=False, min_circ=None, max_circ=None,
                 convexity=False, min_conv=None,
                 max_conv=None, Inertia=False, min_in=None, max_in=None,
                 hfov=79.31, vfov=55.12):
        """Set up blob detector algorithm parametters
        and attributes needed for 3D positions computation."""
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
        """Find blob in frame according to parameters,
        define Roi and compute 3D position in camera frame, then in Reachy frame."""

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
        """Allow to kill the main loop"""
        self.Terminated = True

    def define_roi(self, center, r):
        """Define Roi as a rectangle circumscribed by a circle.

        Args:
            center: tuple (x,y) refering to the centre of a circle
            r: radius of a circle
        """
        self.roi['xmin'] = int(center[0] - r - 10)
        self.roi['ymin'] = int(center[1] - r - 10)
        self.roi['xmax'] = int(center[0] + r + 10)
        self.roi['ymax'] = int(center[1] + r + 10)

    def calc_angle(self, offset, fov, size):
        """Compute angle needed to calculate 3D position.

        Args:
            offset: distance in pixel between ROI centroid and image center along x or y axis.
            fov: horizontal or vertical field of vue depending on axis (x or y) in radian
            size: horizontal or vertical image size depending on axis (x or y) in pixel
        """
        return math.atan(math.tan(fov / 2.0) * offset / (size / 2.0))

    def compute_pos_in_camera_frame(self):
        """Compute 3D position in camera frame of an object in ROI."""

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
        """Convert 3D position of an object in ROI from camera frame to Reachy frame."""
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
    """Handle depth map computation from stereo cameras."""

    def __init__(self):
        """Set-up depthAi pipeline"""
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
        """Save curent depth map and black and white frame"""
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
        """Allow to kill the main loop"""
        self.Terminated = True


class ReachyControl (threading.Thread):
    """Handle Reachy motions to reach an object detected by blob detection."""
    def __init__(self, g, t):
        """Set up Reachy"""
        threading.Thread.__init__(self)
        self.g = g
        self.t = t
        self.key = None
        self.reachy = ReachySDK(host='localhost')
        self.Terminated = False
        self.Start = False

    def run(self):
        """Main loop minimizing the distance between gripper and target position in reachy frame"""
        kpx = 0.5
        kpy = 0.5
        kpz = 1.5
        first = True
        self.reachy.turn_on('r_arm')
        goto({self.reachy.r_arm.r_gripper: -60}, duration=1)
        while not self.Terminated:
            if self.Start:
                if self.compare_to_ref():
                    err = [self.t.pos['robot']['x'] - self.g.pos['robot']['x'],
                           self.t.pos['robot']['y'] - self.g.pos['robot']['y'],
                           self.t.pos['robot']['z'] + 0.03 - self.g.pos['robot']['z']]
                    ro = np.sqrt(err[0]**2 + err[1]**2)
                    if ro > 0.06:
                        rot = self.compute_rotation()
                        c = self.reachy.r_arm.forward_kinematics()
                        orientation = np.zeros((4, 3))
                        orientation[:3, :3] = rot

                        M = np.concatenate((orientation, [[c[0][3]+kpx*err[0]],
                                            [c[1][3]+kpy*err[1]],
                                            [c[2][3]+kpz*err[2]],
                                            [1]]), axis=1)

                        joints_pos = self.reachy.r_arm.inverse_kinematics(M)
                        self.reachy.turn_on('r_arm')
                        goto({joint: pos for joint, pos in zip(self.reachy.r_arm.joints.values(), joints_pos)},
                             duration=1,
                             interpolation_mode=InterpolationMode.MINIMUM_JERK)
                        # time.sleep(1)
                    elif first:
                        print('first')
                        goto({self.reachy.r_arm.r_gripper: -15}, duration=0.5)
                        first = False

            if self.key == ord('s'):
                if self.Start:
                    self.Start = False
                else:
                    self.Start = True

    def compare_to_ref(self):
        """Test if the gripper position estimation is not so far from the real position."""
        current = self.g.pos['robot']
        reachy = self.reachy.r_arm.forward_kinematics()
        if abs(current['x']-reachy[0][3]) < 0.08 and abs(current['y']-reachy[1][3]) < 0.08:
            return True
        return False

    def stop(self):
        """Allow to kill the main loop."""
        self.Terminated = True

    def compute_rotation(self):
        """Compute the gripper rotation matrix depending on target position"""
        dx = self.t.pos['robot']['x']-self.g.pos['robot']['x']
        dy = self.t.pos['robot']['y']-self.g.pos['robot']['y']
        theta = np.arctan(dy/dx)
        if dx < 0 and dy > 0:
            theta = np.pi + theta
        elif dx < 0 and dy < 0:
            theta = theta - np.pi
        R1 = R.from_euler('y', np.deg2rad(-90))
        R2 = R.from_euler('x', theta)
        r = R1 * R2
        return r.as_matrix()


def main():
    """Main function launching threads and displaying the depth map with current ROI."""
    depth_comput = DepthComputation()
    depth_comput.start()

    gripper_detection = BlobDetection(name='gripper', depth=depth_comput,
                                      min_thresh=100, max_thresh=255,
                                      area=True, min_area=800,
                                      circularity=True, min_circ=0.8,
                                      convexity=True, min_conv=0.87,
                                      Inertia=True, min_in=0.2)
    gripper_detection.start()

    target_detection = BlobDetection(name='target', depth=depth_comput,
                                     min_thresh=100, max_thresh=255,
                                     area=True, min_area=3000, max_area=10000,
                                     circularity=True, min_circ=0.65,
                                     max_circ=0.785)
    target_detection.start()

    reachy_control = ReachyControl(gripper_detection, target_detection)
    reachy_control.start()

    blob_threads = [gripper_detection, target_detection]

    while True:

        if depth_comput.depthFrameColor is not None:
            im_with_keypoints = depth_comput.depthFrameColor

            for thread in blob_threads:
                if thread.bbox['ymax'] is not None:
                    cv2.rectangle(im_with_keypoints,
                                  (thread.bbox['xmin'], thread.bbox['ymin']),
                                  (thread.bbox['xmax'], thread.bbox['ymax']),
                                  (255, 0, 0), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

            cv2.imshow("Keypoints", im_with_keypoints)
        reachy_control.key = cv2.waitKey(1)

        if reachy_control.key == ord('q'):
            reachy_control.reachy.turn_off_smoothly('r_arm')
            reachy_control.stop()
            gripper_detection.stop()
            target_detection.stop()
            depth_comput.stop()
            break


if __name__ == "__main__":
    main()
