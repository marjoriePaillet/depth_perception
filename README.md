# depth_perception with Reachy's arm

Package allowing Reachy's arm to grab objects thanks to blob detection and Luxonis custom device.

**Python version: 3**
Dependencies: [reachy_sdk](https://github.com/pollen-robotics/reachy-sdk), [opencv](https://opencv.org/)

How to install:

```bash
cd ~/pathToYourDirectory
git clone https://github.com/marjoriePaillet/depth_perception.git
```

## How to use it

Depth percption is compute thanks to a custom electronic card from Luxonis with two monochromatics cameras.
![alt text](https://github.com/marjoriePaillet/depth_perception/blob/master/images/depthAI-card.jpg)

cameras FOV: 
  - HFOV = 79.31 deg
  - VFOV = 55.12 deg
  - DFOV = 88.81 deg
FOV values can be changed in BlobDetection() objects declaration.

In order to make possible the opencv blob detection with current parameters put a black 2cm circle of paper on Reachy's gripper, 
and a black 3.5cm square of paper on on the targeted objet as shown in the following picture.
![alt text](https://github.com/marjoriePaillet/depth_perception/blob/master/images/Reachy_setup.jpg)

Pieces of paper must stay visible by cameras.

To run the code, go to /scr directory and run go_to.py file as follow : 
```python3 go_to.py```
