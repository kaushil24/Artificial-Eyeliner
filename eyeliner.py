import numpy as np
import dlib
import cv2
import matplotlib.pyplot as plt
import skimage
from scipy.interpolate import interp1d
from imutils import face_utils
import argparse
import os
from PIL import Image



def getEyeLandnarkPts(face_landmark_points):
    '''
    Input: Coordinates of Bounding Box single face
    Returns: eye's landmark points
    '''
    face_landmark_points[36][0]-=5
    face_landmark_points[39][0]+=5
    face_landmark_points[42][0]-=5
    face_landmark_points[45][0]+=5
    
    L_eye_top = face_landmark_points[36: 40]
    L_eye_bottom = np.append(face_landmark_points[39: 42], face_landmark_points[36]).reshape(4,2)

    R_eye_top = face_landmark_points[42:  46]
    R_eye_bottom = np.append(face_landmark_points[45:48], face_landmark_points[42]).reshape(4,2)
       
    return [L_eye_top, L_eye_bottom, R_eye_top, R_eye_bottom]


def interpolateCoordinates(xy_coords, x_intrp):
    x = xy_coords[:, 0]
    y = xy_coords[:, 1]
    intrp = interp1d(x, y, kind='quadratic')
    y_intrp = intrp(x_intrp)
    y_intrp = np.floor(y_intrp).astype(int)
    return y_intrp


def getEyelinerPoints(eye_landmark_points):
    '''
    Takes an array of eye coordinates and interpolates them:
    '''
    L_eye_top, L_eye_bottom, R_eye_top, R_eye_bottom = eye_landmark_points

    L_interp_x = np.arange(L_eye_top[0][0], L_eye_top[-1][0], 1)
    R_interp_x = np.arange(R_eye_top[0][0], R_eye_top[-1][0], 1)

    L_interp_top_y = interpolateCoordinates(L_eye_top, L_interp_x)
    L_interp_bottom_y = interpolateCoordinates(L_eye_bottom, L_interp_x)

    R_interp_top_y = interpolateCoordinates(R_eye_top, R_interp_x)
    R_interp_bottom_y = interpolateCoordinates(R_eye_bottom, R_interp_x)

    return [(L_interp_x, L_interp_top_y, L_interp_bottom_y), (R_interp_x, R_interp_top_y, R_interp_bottom_y)]


def drawEyeliner(img, interp_pts, color = (85,90,92) , thickness = 1):
    L_eye_interp, R_eye_interp = interp_pts

    L_interp_x, L_interp_top_y, L_interp_bottom_y = L_eye_interp
    R_interp_x, R_interp_top_y, R_interp_bottom_y = R_eye_interp

    overlay = img.copy()
    # overlay = np.empty(img.shape)
    # overlay = np.zeros_like(img)

    for i in range(len(L_interp_x)-2):
        x1 = L_interp_x[i]
        y1_top = L_interp_top_y[i]
        x2 = L_interp_x[i+1]
        y2_top = L_interp_top_y[i+1]
        cv2.line(overlay, (x1, y1_top), (x2, y2_top), color, thickness)

        y1_bottom = L_interp_bottom_y[i]
        y2_bottom = L_interp_bottom_y[i+1]
        cv2.line(overlay, (x1, y1_bottom), (x1, y2_bottom), color, thickness)

    
    for i in range(len(R_interp_x)-2):
        x1 = R_interp_x[i]
        y1_top = R_interp_top_y[i]
        x2 = R_interp_x[i+1]
        y2_top = R_interp_top_y[i+1]
        cv2.line(overlay, (x1, y1_top), (x2, y2_top), color, thickness)

        y1_bottom = R_interp_bottom_y[i]
        y2_bottom = R_interp_bottom_y[i+1]
        cv2.line(overlay, (x1, y1_bottom), (x1, y2_bottom), color, thickness)

    # background = Image.fromarray(img) # .convert("1")
    # foreground = Image.fromarray(overlay).convert("1")

    # newImg = Image.composite(foreground, background, foreground)#, mask='1')
    
    # # img = cv2.bitwise_and(overlay, img)
    # return cv2.cvtColor(np.array(newImg), cv2.COLOR_RGB2BGR)
    return overlay




if __name__ == "__main__":
    cap = cv2.VideoCapture("Media/Sample Video 1.mp4") 
    face_detector = dlib.get_frontal_face_detector()
     

    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bounding_boxes = face_detector(gray, 0)# The 2nd argument means that we upscale the image by 'x' number of times to detect more faces.
        lndMrkDetector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        

        for i, bb in enumerate(bounding_boxes):
            face_landmark_points = lndMrkDetector(gray, bb)
            face_landmark_points = face_utils.shape_to_np(face_landmark_points)
            eye_landmark_points = getEyeLandnarkPts(face_landmark_points)
            eyeliner_points = getEyelinerPoints(eye_landmark_points)
            output_frame = drawEyeliner(frame, eyeliner_points,color=(0,0,0),thickness= 2) 
            
        
        
        cv2.imshow("SDSD", cv2.resize(output_frame, (600,600)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()