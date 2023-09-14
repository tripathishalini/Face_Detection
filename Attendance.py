import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


def attendance(cap):

    path = 'C:/Users/user/Downloads/ImagesAttendance'
    images = []
    Names = []
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        Names.append(os.path.splitext(cl)[0])
    print(Names)

    def findEncodings(images) : 
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    encodeListKnown = findEncodings(images)
    print("Encoding complete")

    with open('C:/Users/user/Downloads/Attendance.csv', 'w+') as f:
        f.writelines(f'\nName,  Time')
        pass

    def markAttendance(name) : 
        with open('C:/Users/user/Downloads/Attendance.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime("%H:%M:%S")
                f.writelines(f'\n{name}, {dtString}')

    #cap = cv2.VideoCapture(0)

    #c = 0 ;
    while True:
        #c = c + 1
        success, img = cap.read()
        imgs = cv2.resize(img, (0,0), None, 0.25, 0.25)
        imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #state the location of face detect
        facesCurFrame = face_recognition.face_locations(imgs)
        #encoding the webcam images
        encodesCurFrame = face_recognition.face_encodings(imgs,facesCurFrame)

        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]: 
                name = Names[matchIndex].upper()
                print(name)
                markAttendance(name)
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows() 
