'''Real Time Driver Drowsiness Detection System'''

import cv2
import dlib
import pygame
from imutils import face_utils
from scipy.spatial import distance



def EyeAspectRatio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    EAR = (A+B) / (2*C)
    return EAR

def HistogramEqualization(image):
    return cv2.equalizeHist(image) 


SUM = 0
BLINK = 0
COUNTER = 0
EYE_ASPECT_RATIO_THRESHOLD = 0.30
EYE_ASPECT_RATIO_CONSECUTIVE_FRAMES = 15



#Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')

#Load face cascade to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

#Load face detector and predictor using dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_facial_landmarks.dat')

#Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#Start webcam video capture
video_capture = cv2.VideoCapture(0)



while(True):
    #Read each frame and flip it, and convert to grayscale
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Performing Histogram Equalization
    gray = HistogramEqualization(gray)
    
    #Detect facial points through detector function
    faces = detector(gray, 0)

    #Detect faces through haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Draw rectangle around each face detected
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
        
        
    #Detect facial points
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        #Get array of coordinates of leftEye and rightEye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        #Calculate aspect ratio of both eyes
        EAR = (EyeAspectRatio(leftEye) + EyeAspectRatio(rightEye)) / 2

        #Use hull to remove convex contour discrepencies and draw eye shape around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        

        #Detect if EAR is less than threshold
        if(EAR < EYE_ASPECT_RATIO_THRESHOLD):
            COUNTER += 1
            SUM += 1
            
            #If no. of frames is greater than threshold frames,
            if(COUNTER >= EYE_ASPECT_RATIO_CONSECUTIVE_FRAMES):
                pygame.mixer.music.play(0)
                cv2.putText(frame, "Driver Is Drowsy", (140,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 3)
            
            if(SUM > 10):
                BLINK += 1
                SUM = 0
            
        else:
             pygame.mixer.music.stop()
             COUNTER = 0
        
        #Draw the total number of blinks & current EAR
        cv2.putText(frame, "BLINKS: {}".format(BLINK), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(EAR), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
    #Show each frame
    cv2.imshow('Real Time Driver Drowsiness Detection System', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

#If video capture is over
video_capture.release()
cv2.destroyAllWindows()
