from importlib.resources import path
import cv2
path = r'D:\MTCNN_FaceNet\Dataset\Facedata\raw\new\dataset'
n = 0
cam = cv2.VideoCapture(1)

while(cam.isOpened()):
    ret, frame = cam.read()
    cv2.imshow('cam',frame)
    cv2.imwrite(path + str(n).zfill(5) + '.png',frame)
    n = n+1
    if cv2.waitKey(1) & n == 100:
        break
cam.release()
cv2.destroyAllWindows()