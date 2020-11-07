import cv2
import os

dataPath = '/Users/martin/PycharmProjects/OpencvPython/PEOPLE'
imagePaths = os.listdir(dataPath)
print('ImagePaths= ',imagePaths)

#recognizer = cv2.face.EigenFaceRecognizer_create()
#recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer = cv2.face.LBPHFaceRecognizer_create()

#recognizer.read('modeloEigen.xml')
#recognizer.read('modeloFisher.xml')
recognizer.read('modeloLBPH.xml')

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
while True:
    succes, img = cap.read()
    if succes == False: break
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    auximg = imgGray.copy()

    faces = faceCascade.detectMultiScale(imgGray, 1.2, 4)

    for (x, y, w, h) in faces:
        rostro = auximg[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
        result = recognizer.predict(rostro)

        #cv2.putText(img,'{}'.format(result),(x,y-5),1,1.3,(182,252,3),1,cv2.LINE_AA)
        '''
        #Eigen
        if result[1] < 8000:
            cv2.putText(img, '{}'.format(imagePaths[result[0]+1]), (x, y - 25), 2, 1.1, (182, 252, 3), 1, cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x + w, y + h), (182, 252, 3), 2)

        else:
            cv2.putText(img,'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        '''
        '''
        #Fisher
        if result[1] < 800:
            cv2.putText(img, '{}'.format(imagePaths[result[0] + 1]), (x, y - 25), 2, 1.1, (182, 252, 3), 1, cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x + w, y + h), (182, 252, 3), 2)

        else:
            cv2.putText(img, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        '''

        #LBPH
        if result[0] < 60:
            cv2.putText(img, '{}'.format(imagePaths[result[0]+1]), (x, y - 25), 2, 1.1, (182, 252, 3), 1, cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x + w, y + h), (182, 252, 3), 2)

        else:
            cv2.putText(img, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


    cv2.imshow('video',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break