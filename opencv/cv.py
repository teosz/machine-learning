import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread("img.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst  = cv2.cornerHarris(gray, 2,3,0.04)
dst = cv2.dilate(dst,None)
img[dst > 0.01*dst.max() ] = [0,100,0]
#cv2.imshow('dst',img)
cv2.imwrite('dts.jpg' , img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
