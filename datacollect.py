import cv2
import os

video=cv2.VideoCapture(0)


facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count=0

nameID=str(input("Enter Your Name: ")).lower()


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

assure_path_exists("images/")


while True:
	ret,frame=video.read()
	frame = cv2.flip(frame,1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if ret : 
		faces = facedetect.detectMultiScale(
			gray,     
			scaleFactor=1.2,
			minNeighbors=5,     
			minSize=(20, 20)
        )
		for x,y,w,h in faces:
			count=count+1
			name='images/'+nameID+'/'+ str(count) + '.jpg'
			print("Creating Images........." +name)
			cv2.imwrite("images/user." + str(nameID) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
			# cv2.imwrite(name, frame[y:y+h,x:x+w])
			cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
		cv2.imshow("WindowFrame", frame)
		cv2.waitKey(1)
		if count>500:
			break
	
video.release()
cv2.destroyAllWindows()