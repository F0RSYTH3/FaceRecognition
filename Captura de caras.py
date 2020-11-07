import cv2
import os
import imutils

personName = 'GERA'
dataPath = '/Users/martin/PycharmProjects/OpencvPython/mk2'
personPath = dataPath + '/' + personName
#print(personPath)
if not os.path.exists(personPath):
    print('Carpeta creada',personPath)
    os.makedirs(personPath)

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
count = 0
while True:
    succes, img = cap.read()
    if succes == False: break
    img = imutils.resize(img, width=640)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    auximg = img.copy()

    faces = faceCascade.detectMultiScale(imgGray, 1.2, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (182, 252, 3), 2)
        rostro = auximg[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count),rostro)
        count = count + 1

    cv2.imshow("video",img)

    if cv2.waitKey(1) & 0xFF==ord('q') or count >= 300:
        break