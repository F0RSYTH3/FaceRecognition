import cv2
print("Ready to go")

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

cap.set(3,640)
cap.set(4,480)

while True:
    succes, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (182, 252, 3), 2)

    cv2.imshow("video",img)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break