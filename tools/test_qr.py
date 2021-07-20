"""Test of QR-Code detection with opencv and Zbar librairies"""

import cv2
import pyzbar.pyzbar as pyzbar
import numpy as np

"""With openCv"""

# inputIm = cv2.imread('qr.png')


# def display(im, bbox):
#     bbox = np.squeeze(bbox)
#     cv2.rectangle(im, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (0, 255, 0), 3)

#     cv2.imshow('Result', im)


# qrDecoder = cv2.QRCodeDetector()

# data, bbox, rectifiedImage = qrDecoder.detectAndDecode(inputIm)
# if len(data) > 0:
#     print(f'Decoded Data: {data}')
#     display(inputIm, bbox)
#     rectifiedImage = np.uint8(rectifiedImage)
#     cv2.imshow('Rectified image', rectifiedImage)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""With zbar"""


def decode(im):
    # Find barcodes and QR codes
    decodedObjects = pyzbar.decode(im)
    for obj in decodedObjects:
        print('Type : ', obj.type)
        print('Data : ', obj.data, '\n')

    return decodedObjects


# Display barcode and QR code location
def display(im, decodedObjects):

    # Loop over all decoded objects
    for decodedObject in decodedObjects:
        points = decodedObject.polygon

        # If the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points

        # Number of points in the convex hull
        n = len(hull)
        print(hull)

        # Draw the convext hull
        for j in range(0, n):
            cv2.line(im, hull[j], hull[(j+1) % n], (255, 0, 0), 3)

    # Display results
    cv2.imshow("Results", im)
    cv2.waitKey(0)


# Main
if __name__ == '__main__':

    # Read image
    im = cv2.imread('qr.png')

    decodedObjects = decode(im)
    display(im, decodedObjects)
