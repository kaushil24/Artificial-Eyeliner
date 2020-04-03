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


overlay_crop = 0
dataFile = "shape_predictor_68_face_landmarks.dat"

color = (0,0,0)
thickness = 2
face_detector = dlib.get_frontal_face_detector()
lndMrkDetector = dlib.shape_predictor(dataFile)

def getEyeLandmarkPts(face_landmark_points):
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


def drawEyeliner(img, interp_pts):
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

    overlay_crop = overlay[min(L_interp_bottom_y) - 50 : max(L_interp_top_y) + 50, L_interp_x[0]-50 : L_interp_x[-1] + 50 ]
    # print(max(L_interp_top_y) + 15, min(L_interp_bottom_y) - 15, L_interp_x[0]-10, L_interp_x[-1] + 10 )

    return overlay, overlay_crop


def Eyeliner(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bounding_boxes = face_detector(gray, 0)# The 2nd argument means that we upscale the image by 'x' number of times to detect more faces.
    if bounding_boxes:    
        for i, bb in enumerate(bounding_boxes):
            face_landmark_points = lndMrkDetector(gray, bb)
            face_landmark_points = face_utils.shape_to_np(face_landmark_points)
            eye_landmark_points = getEyeLandmarkPts(face_landmark_points)
            eyeliner_points = getEyelinerPoints(eye_landmark_points)
            op, op_crop = drawEyeliner(frame, eyeliner_points) 
        
        return op, op_crop
    else:
        return frame

def video(src = 0):

    cap = cv2.VideoCapture(src)

    if args['save']:
        if os.path.isfile(args['save']+'.avi'):
            os.remove(args['save']+'.avi')
        out = cv2.VideoWriter(args['save']+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30,(int(cap.get(3)),int(cap.get(4))))
	
    while(cap.isOpened):
        _ , frame = cap.read()
        output_frame, op_crop = Eyeliner(frame)

        if args['save']:
            out.write(output_frame)

        cv2.imshow("Artificial Eyeliner", cv2.resize(output_frame, (600,600)))
        cv2.imshow('Eye Region', cv2.resize(op_crop, (400, 200)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if args['save']:
        out.release()

    cap.release()
    cv2.destroyAllWindows()


def image(source):
    if os.path.isfile(source):
        img = cv2.imread(source)
        output_frame = Eyeliner(img)
        cv2.imshow("Artificial Eyeliner", cv2.resize(output_frame, (600, 600)))
        if args['save']:
            if os.path.isfile(args['save']+'.png'):
                os.remove(args['save']+'.png')
            cv2.imwrite(args['save']+'.png', output_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("File not found :( ")


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=False, help="Path to video file")
    ap.add_argument("-i", "--image", required=False, help="Path to image")
    ap.add_argument("-d", "--dat", required=False, help="Path to shape_predictor_68_face_landmarks.dat")
    ap.add_argument("-t", "--thickness", required=False, help="Enter int value of thickness (recommended 0-5)")
    ap.add_argument("-c", "--color", required=False, help='Enter R G B color value', nargs=3)
    ap.add_argument("-s", "--save", required=False, help='Enter the file name to save')
    args = vars(ap.parse_args())

    if args['dat']:
        dataFile = args['dat']

    else:
        dataFile = "shape_predictor_68_face_landmarks.dat"

    color = (0,0,0)
    thickness = 2
    face_detector = dlib.get_frontal_face_detector()
    lndMrkDetector = dlib.shape_predictor(dataFile)

    if args['color']:
        color = list(map(int, args['color']))
        color = tuple(color)

    if args['thickness']:
        thickness = int(args['thickness'])

    if args['image']:
        image(args['image'])

    if args['video'] and args['video']!='webcam':
        if os.path.isfile(args['video']):
            video(args['video'])

        else:
            print("File not found :( ")

    elif args['video']=='webcam':
        video(0)
