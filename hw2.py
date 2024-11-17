import cv2
import matplotlib.pyplot as plt
import numpy as np

######################################################################

def imgmax(img):
    height = len(img)
    width = len(img[0])
    max = img[0][0]
    for i in range(height):
        for j in range(width):
            if img[i][j] > max:
                max = img[i][j]
    return max

######################################################################

def imgary2hist(img):
    height = len(img)
    width = len(img[0])
    hist = np.zeros(256)
    for i in range(height):
        for j in range(width):
            hist[img[i][j]] += 1
    plt.bar(range(256), hist, color='gray',width=1)
    plt.show()
    
    return hist

######################################################################

def sum_hist(hist):
    sum = 0
    for i in range(256):
        sum += hist[i]
    return sum

######################################################################

def  hist2cdf(hist):
    acc_hist = np.zeros(256)
    acc_hist[0] = hist[0]
    for i in range(1, 256):
        acc_hist[i] = acc_hist[i-1] + hist[i]
    return acc_hist

######################################################################



######################################################################

img = cv2.imread('imgs/img2.jpg', 0)
hist = imgary2hist(img)
acc_hist = hist2cdf(hist)
resimg = np.zeros(256)
max = imgmax(hist)
sum = sum_hist(hist)
for i in range(256):
    acc_hist[i] = acc_hist[i]  / sum * max
for i in range(256):
    resimg[i] = acc_hist[i]

resimg = np.clip(resimg, 0, 255).astype(np.uint8)
cv2.imshow('resimg', resimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
