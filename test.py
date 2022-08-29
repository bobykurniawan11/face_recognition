import numpy as np
import cv2
import numpy as np

names = [ 'Boby' , "WHO ?"]

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font=cv2.FONT_HERSHEY_COMPLEX



recognizer = cv2.face.LBPHFaceRecognizer_create()


recognizer.read('trainer/trainer.yml')

def get_className(classNo):
	return classNo

while True:
	sucess, imgOrignal=cap.read()
	imgOrignal = cv2.flip(imgOrignal,1)

	gray = cv2.cvtColor(imgOrignal,cv2.COLOR_BGR2GRAY)

	faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
	for x,y,w,h in faces:

		crop_img=imgOrignal[y:y+h,x:x+h]
		img=cv2.resize(crop_img, (224,224))
		img=img.reshape(1, 224, 224, 3)
		Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
		cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
		cv2.putText(imgOrignal, str(names[Id]),  (x,y-12), font,1, (255,0,0), 6)
		cv2.putText(imgOrignal, str(" {0} %".format(round(100 - confidence))), (x-15,y+h-10), font, 1, (255,255,0), 1)

	cv2.imshow("Result",imgOrignal)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()





















