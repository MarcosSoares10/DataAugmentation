import math
import random
import numpy as np
import cv2
from scipy.ndimage.interpolation import shift, zoom
from scipy.ndimage import rotate

def flip_img(img, flip_axis):
        """flip img along specified axis(x or y)"""
        return np.flip(img, flip_axis)

def shift_img(img, shift_range, shift_axis):
        """shift img by specified range along specified axis(x or y)"""
        shift_lst = [0] * img.ndim
        shift_lst[shift_axis] = math.floor(shift_range * img.shape[shift_axis])
        return shift(img, shift=shift_lst, cval=0)

def resize_img(img, scale_percent):
        """Resize image using """
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        zoomed = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return zoomed

def rotate_img(img, rotate_axis,rotate_angle):
        """rotate img by specified range along specified axis(x or y)"""
        return rotate(img, axes=rotate_axis, angle=rotate_angle, cval=0.0, reshape=False)

def sharpen_img(img):
        #Edge enhance
        kernel = np.array([[-1,-1,-1,-1,-1],
                    [-1,2,2,2,-1],
                    [-1,2,8,2,-1],
                    [-2,2,2,2,-1],
                    [-1,-1,-1,-1,-1]])/8.0
        result=cv2.filter2D(img,-1,kernel)
        return result

def adjust_gamma_img(img, gamma):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)



namefile="NEGATIVE/A"
formatfile=".jpg"


img = cv2.imread(namefile+formatfile)


scale_percent = 30 
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)


#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

flip = flip_img(img, 0)
shift = shift_img(img, 0.05, 1)
resize = resize_img(img,  100)
rotate = rotate_img(img, (0,1), 180)
sharpen = sharpen_img(img)
gammaadjusted = adjust_gamma_img(img, 0.8)

'''cv2.imshow("IMG",img)
cv2.imshow("flip",flip)
cv2.imshow("shift",shift)
cv2.imshow("resize",resize)
cv2.imshow("rotate",rotate)
cv2.imshow("sharpen",sharpen)
cv2.imshow("gammaadjusted",gammaadjusted)'''

cv2.imwrite(namefile+"flip.jpg",flip)
cv2.imwrite(namefile+"_shift.jpg",shift)
cv2.imwrite(namefile+"_resize.jpg",resize)
cv2.imwrite(namefile+"_rotate.jpg",rotate)
cv2.imwrite(namefile+"_sharpen.jpg",sharpen)
cv2.imwrite(namefile+"_gammaadjusted.jpg",gammaadjusted)

cv2.waitKey(0)