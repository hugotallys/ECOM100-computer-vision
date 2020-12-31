import numpy
import os
import cv2
import urllib
import time

import numpy as np

from urllib.request import urlopen


def main():
    cap = cv2.VideoCapture(0)

    # termination criteria

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    while(True):
        # Capture frame-by-frame
        
        _, frame = cap.read()

        # Our operations on the frame come here
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

        # If found, add object points, image points (after refining them)
        
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            
            frame = cv2.drawChessboardCorners(frame, (7,6), corners2,ret)
            
            # Display the resulting frame
            
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            # Computing rotation matrix and translation vector using the first corner (camera extrinsic parameters)
            
            rotation_matrix = numpy.zeros(shape=(3,3))
            cv2.Rodrigues(rvecs[0], rotation_matrix)

            print(50*'-' + '\nR=')
            print(rotation_matrix)
            print(50*'-' + '\nT=')
            print(tvecs[0])

            cv2.imshow('frame', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        else:
            cv2.imshow('frame', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
