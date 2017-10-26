import numpy as np
import cv2

def preprocess(img):
	cv2.imshow("s",img)
	imgBlurred = cv2.GaussianBlur(img, (5,5), 0)
	gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
	sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=7)
	#sobelx.convertTo(sobel,cv2.CV_8U);
	#ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imshow("e",sobelx)
	cv2.waitKey(0)
	

if __name__ == '__main__':
	print "START" 
	img = cv2.imread("test.jpeg")
	preprocess(img)