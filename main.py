import face_recognition
import cv2
from playsound import playsound
from threading import Thread
import numpy as np
from scipy.spatial import distance as dist

min_EAR=0.3
eye_Ar_cosec_frames1=12   #2.40sec
eye_Ar_cosec_frames2=24 #4.00sec

counter=0
alarmOn=False

def soundAlerm(soundFile):
    while alarmOn==True:
        playsound(soundFile)
    
def eye_aspect_ratio(eye):
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    ear = (a + b) / (2 *c)
    return ear

def main():
    global alarmOn, counter
    cap=cv2.VideoCapture(0)
    # cap.set(3,320)
    # cap.set(4,240)
    
    while True:
        ref,frame=cap.read()
        faceLandmarkLists=face_recognition.face_landmarks(frame)
        for faceLandmark in faceLandmarkLists:
            leftEye=faceLandmark['left_eye']
            rightEye=faceLandmark['right_eye']

            leftEAR=eye_aspect_ratio(leftEye)
            rightEAR=eye_aspect_ratio(rightEye)
            ear=(leftEAR+rightEAR)/2

            # lPoint=np.array(leftEye)
            # rPoint=np.array(rightEye)
            # cv2.polylines(frame,[lPoint],True,(255,255,0),1)
            # cv2.polylines(frame,[rPoint],True,(255,255,0),1)

            if(ear<min_EAR):
                counter+=1
                if counter>=eye_Ar_cosec_frames2:
                    cv2.putText(frame,'Warning...',(200,235),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
                    cv2.putText(frame,'You Are Sleeping !!!!',(175,270),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),1)  
                    cv2.rectangle(frame,(10,10),(630,470),	(0,0,255),3)
                    if not alarmOn:
                        alarmOn=True  
                        t=Thread(target=soundAlerm, args=("C:/Users/Arpit Maurya/Desktop/Programs/car/2nd/alarm.wav",))
                        t.daemon=True
                        t.start()
                elif(counter>=eye_Ar_cosec_frames1):
                    alarmOn=False   
                    cv2.putText(frame,'Alert! You Are Sleepy.....',(170,270),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),1)      
                    cv2.rectangle(frame,(10,10),(630,470),	(0,255,255),3)
            else:
                counter=0
                alarmOn=False  
                     
        cv2.imshow('Alarm System',frame)
        if cv2.waitKey(1)==ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    main()          