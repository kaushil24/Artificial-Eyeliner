import tkinter
import PIL.Image
import PIL.ImageTk
import cv2
import dlib
import numpy as np
from eyeliner import *

frame1_x = 910
frame1_y = 512
frame2_x = 400
frame2_y = 200
switch = False
dataFile = "shape_predictor_68_face_landmarks.dat"

color = (0,0,0)
thickness = 2
face_detector = dlib.get_frontal_face_detector()
lndMrkDetector = dlib.shape_predictor(dataFile)


class App:
    def __init__(self, window, video_source1, video_source2):
        self.window = window
        self.window.title("Artificial Eye Liner")
        self.video_source1 = video_source1
        self.video_source2 = video_source2
        self.photo1 = ""
        self.photo2 = ""

        # open video source
        self.vid1 = MyVideoCapture(self.video_source1, self.video_source2)

        # Create a canvas that can fit the above video source size
        self.canvas1 = tkinter.Canvas(window, width=frame1_x, height=frame1_y)
        self.canvas2 = tkinter.Canvas(window, width=frame2_x, height=frame2_y)
        self.canvas1.pack(padx=0, pady=2, side="top") # padx, pady = 5, 10
        self.canvas2.pack(padx=0, pady=5, side="left") # '' , '' = 5, 60
		
        def counting():
        	global switch
        	switch = not switch
        	return switch

        self.button = tkinter.Button(window, text='Eyeliner on/off', width=25, command=counting)
        self.button.pack(side = 'right')
        
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        ret1, frame1, ret2, frame2 = self.vid1.get_frame

        if ret1 and ret2:
                self.photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame1))
                self.photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame2))
                self.canvas1.create_image(0, 0, image=self.photo1, anchor=tkinter.NW)
                self.canvas2.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source1, video_source2):
        # Open the video source
        self.vid1 = cv2.VideoCapture(video_source1)
        # self.vid2 = cv2.VideoCapture(video_source2)

        if not self.vid1.isOpened():
            raise ValueError("Unable to open video source", video_source1)

    @property
    def get_frame(self):
        ret1 = ""
        ret2 = ""
        global switch
        if self.vid1.isOpened(): # and self.vid2.isOpened():
            ret1, frame1 = self.vid1.read()
            ret2 = ret1
            output_frame, op_crop = Eyeliner(frame1)
            if not switch:
            	output_frame = frame1
            	op_crop = np.zeros_like(frame1)#,frame2_y)
            # ret2, frame2 = self.vid2.read()
            frame1 = cv2.resize(output_frame, (frame1_x, frame1_y))
            frame2 = cv2.resize(op_crop, (frame2_x, frame2_y))
            if ret1 and ret2:
                # Return a boolean success flag and the current frame converted to BGR
                return ret1, cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), ret2, cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            else:
                return ret1, None, ret2, None
        else:
            return ret1, None, ret2, None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid1.isOpened():
            self.vid1.release()
        # if self.vid2.isOpened():
        #     self.vid2.release()

def callback():
    # global v1,v2
    # v1=E1.get()
    # v2=E2.get()
    # if v1 == "" or v2 == "":
    #     L3.pack()
    #     return
    initial.destroy()


v1 = "vid1.mp4"
v2 = "/home/pyf/Videos/finish_3.mp4"

initial = tkinter.Tk()
initial.minsize(600,600)
initial.title("Artificial Eye Liner")

# L0 = tkinter.Label(initial, text="Enter the full path")
# L0.pack()
# L1 = tkinter.Label(initial, text="Video 1")
# L1.pack()
# E1 = tkinter.Entry(initial, bd =5)
# E1.pack()
# L2 = tkinter.Label(initial, text="Video 2")
# L2.pack()
# E2 = tkinter.Entry(initial, bd =5)
# E2.pack()
B = tkinter.Button(initial, text ="Open Web-Cam", command = callback)
B.pack(padx = 250, pady = 250)
# L3 = tkinter.Label(initial, text="Enter both the names")

initial.mainloop()


# Create a window and pass it to the Application object
App(tkinter.Tk(),v1, v2)
