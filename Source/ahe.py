import cv2
import numpy as np
import os


def ahe(frame):
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    x, y, z = histogram(v)
    img_eq = np.empty((370, 1224), dtype=np.uint8)

    t = np.array_split(v, 72, axis=1)
    img_eq = np.empty((370, 0), dtype=np.uint8)
    for i in range(72):
        img_eq2 = np.empty((0, 17), dtype=np.uint8)
        k = np.array_split(t[i], 2, axis=0)
        for j in range(2):
            a, b, c = histogram(k[j])
            img_eq2 = np.append(img_eq2, a, axis=0)
        img_eq = np.append(img_eq, img_eq2, axis=1)

    img_eq_hsv = cv2.merge([h, s, img_eq])
    img_eq_brg = cv2.cvtColor(img_eq_hsv, cv2.COLOR_HSV2BGR)


    img_eq_hsv0 = cv2.merge([h, s, x])
    img_eq_brg0 = cv2.cvtColor(img_eq_hsv0, cv2.COLOR_HSV2BGR)
    return img_eq_brg

def histogram(image):
    img = np.array(image)
    img_1d = img.ravel()
    img_eq = np.zeros_like(img_1d)

    hist = np.bincount(
        img_1d, minlength=256) / (image.shape[0] * image.shape[1])
    cdf = hist.cumsum() * 255
    # Remap pixel values
    for i in range(len(img_1d)):
        img_eq[i] = cdf[img_1d[i]]

    img_eq = np.reshape(img_eq, (image.shape[0], image.shape[1]))
    return img_eq, hist, cdf

# Main
CWD_PATH = os.getcwd()
vid = cv2.VideoCapture('./Input/histogram.avi')
result = cv2.VideoWriter('./Results/adaptive_histogram_eq.avi', cv2.VideoWriter_fourcc(*'MJPG'),1, (1224,370))
while True:
    ret,frame = vid.read()
    if frame is None:
        break
    img_eq_brg = ahe(frame) 
    result.write(img_eq_brg)
    cv2.imshow('img_equalized', img_eq_brg)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == 27:
        break
