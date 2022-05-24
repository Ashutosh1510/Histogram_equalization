import numpy as np
import cv2
import os
from PIL import Image
def histogram(img):
    a = np.zeros((256,),dtype=np.float16)
    b = np.zeros((256,),dtype=np.float16)

    height,width=img.shape

    #finding histogram
    for i in range(width):
        for j in range(height):
            g = img[j,i]
            a[g] = a[g]+1

    #performing histogram equalization
    tmp = 1.0/(height*width)
    b = np.zeros((256,),dtype=np.float16)

    for i in range(256):
        for j in range(i+1):
            b[i] += a[j] * tmp;
        b[i] = round(b[i] * 255);

    # b now contains the equalized histogram
    b=b.astype(np.uint8)


    #Re-map values from equalized histogram into the image
    for i in range(width):
        for j in range(height):
            g = img[j,i]
            img[j,i]= b[g]
    return img

CWD_PATH = os.getcwd()
vid = cv2.VideoCapture('./Input/histogram.avi')
result = cv2.VideoWriter('./Results/histogram_eq.avi', cv2.VideoWriter_fourcc(*'MJPG'),1, (1224,370))
while True:
    ret,frame = vid.read()
    if frame is None:
        break
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img)
    g= histogram(v)
    img= cv2.merge((h,s,g))
    img=cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    result.write(img)
    cv2.imshow('Histogram Equilization',img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == 27:
       break