"""Save gripper orientation depending on spatial location"""

import cv2
import numpy as np
from reachy_sdk import ReachySDK
import time

reachy = ReachySDK(host='localhost')
datas_id = np.zeros((3, 3, 3))
datas = []

im = cv2.imread('../images/index.jpeg')
i = 0
while(1):

    pos = reachy.r_arm.forward_kinematics()
    key = cv2.waitKey(1)
    cv2.imshow('im', im)

    if key == ord('q'):
        with open('datas.npy', 'wb') as f:
            np.save(f, np.array(datas))
        with open('datas_id.npy', 'wb') as f_id:
            np.save(f_id, datas_id)
        break

    if key == ord('s'):
        x = int((pos[0][3]*100-15)/10)
        y = int(-(pos[1][3]*100-10)/10)
        z = int(-(pos[2][3]*100+20)/5)

        datas_id[z][y][x] = i
        datas.append(pos)
        i += 1
        print(datas_id)

    time.sleep(1)
