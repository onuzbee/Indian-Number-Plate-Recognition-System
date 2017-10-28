import numpy as np
import cv2
from copy import deepcopy

def preprocess(img):
	cv2.imshow("Input",img)
	imgBlurred = cv2.GaussianBlur(img, (5,5), 0)
	gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)

	#gray = cv2.equalizeHist(gray)

	sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
	ret2,threshold_img = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#cv2.imshow("Threshold",threshold_img)
	return threshold_img

def extract_contours(threshold_img):
	element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
	morph_img_threshold = threshold_img.copy()
	cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
	#cv2.imshow("Morphed",morph_img_threshold)
	#cv2.waitKey(0)

	im2,contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
	return contours

def isMaxWhite(plate):
	avg = np.mean(plate)
	if(avg>=100):
		return True
	else:
 		return False

def validateRotation(rect):
	(x, y), (width, height), rect_angle = rect
	#angle = 90 - rect_angle if (width < height) else -rect_angle
	#if 15 < abs(angle) < 165:
		#return False
	if(width>height):
		angle = -rect_angle
	else:
		angle = 90 + rect_angle

	if angle>25:
	 	return False


	if height == 0 or width == 0:
		return False
	else:
		return True



def func(img,contours):
	count=0
	for i,cnt in enumerate(contours):
		min_rect = cv2.minAreaRect(cnt)

		if validateRotation(min_rect):

			x,y,w,h = cv2.boundingRect(cnt)
			plate_img = deepcopy(img[y:y+h,x:x+h])

			if(isMaxWhite(plate_img)):
				count+=1
				# img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
				# cv2.imshow("RECTANGLES",img)
				# cv2.waitKey(0)

	print "No. of final cont : " , count




if __name__ == '__main__':
	print "START"
	img = cv2.imread("test.jpeg")
	threshold_img = preprocess(img)
	contours= extract_contours(threshold_img)

	if len(contours)!=0:
		print len(contours) #Test
		#cv2.drawContours(img, contours, -1, (0,255,0), 1)
		#cv2.imshow("Contours",img)
		#cv2.waitKey(0)


	func(img,contours)
