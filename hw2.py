import cv2
import matplotlib.pyplot as plt
import numpy as np

#=====================================================================#

def img2hist(img,type):
    height = len(img)
    width = len(img[0])
    hist = np.zeros(256)
    for i in range(height):
        for j in range(width):
            hist[img[i][j]] += 1
    plt.bar(range(256), hist, color=type, width=1)
    plt.title(f'{type} Histogram')
    plt.show()
    
    return hist

#=====================================================================#

def sum_hist(hist):
    sum = 0
    for i in range(256):
        sum += hist[i]
    return sum

#=====================================================================#

def hist2cdf(hist):
    cdf_hist = np.zeros(256)
    cdf_hist[0] = hist[0]
    for i in range(1, 256):
        cdf_hist[i] = cdf_hist[i-1] + hist[i]
    return cdf_hist

#=====================================================================#
def equalize_gray_image(img):
    cv2.imshow('OrImg', img)
    cv2.waitKey(0)

    hist = img2hist(img,'Gray')
    cdf_hist = hist2cdf(hist)
    sum_val = sum_hist(hist)

    Equalized_img = np.zeros((len(img), len(img[0])), np.uint8)
    for i in range(len(img)):
        for j in range(len(img[0])):
            Equalized_img[i][j] = 255 * cdf_hist[img[i][j]] / sum_val

    Equalized_img = np.clip(Equalized_img, 0, 255).astype(np.uint8)

    new_hist = np.zeros(256)
    for i in range(len(Equalized_img)):
        for j in range(len(Equalized_img[0])):
            new_hist[Equalized_img[i][j]] += 1

    plt.bar(range(256), new_hist, color='gray', width=1)
    plt.title('Equalized Histogram')
    plt.show()

    cv2.imshow('Equalized_img', Equalized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#=====================================================================#
def equalize_rgb_image(img):
    cv2.imshow('Orimg', img)
    cv2.waitKey(0)

    b, g, r = cv2.split(img)
    b_hist = img2hist(b,'Blue')
    g_hist = img2hist(g,'Green')
    r_hist = img2hist(r,'Red')

    b_cdf_hist = hist2cdf(b_hist)
    g_cdf_hist = hist2cdf(g_hist)
    r_cdf_hist = hist2cdf(r_hist)

    b_sum_val = sum_hist(b_hist)
    g_sum_val = sum_hist(g_hist)
    r_sum_val = sum_hist(r_hist)

    new_b = np.zeros((len(b), len(b[0])), np.uint8)
    new_g = np.zeros((len(g), len(g[0])), np.uint8)
    new_r = np.zeros((len(r), len(r[0])), np.uint8)

    for i in range(len(b)):
        for j in range(len(b[0])):
            new_b[i][j] = 255 * b_cdf_hist[b[i][j]] / b_sum_val

    for i in range(len(g)):
        for j in range(len(g[0])):
            new_g[i][j] = 255 * g_cdf_hist[g[i][j]] / g_sum_val 

    for i in range(len(r)): 
        for j in range(len(r[0])):
            new_r[i][j] = 255 * r_cdf_hist[r[i][j]] / r_sum_val


    new_b = np.clip(new_b, 0, 255).astype(np.uint8) 
    new_g = np.clip(new_g, 0, 255).astype(np.uint8)
    new_r = np.clip(new_r, 0, 255).astype(np.uint8)

    Equalized_img = cv2.merge((new_b, new_g, new_r))

    new_b_hist = np.zeros(256)
    new_g_hist = np.zeros(256)
    new_r_hist = np.zeros(256)

    for i in range(len(new_b)):
        for j in range(len(new_b[0])):
            new_b_hist[new_b[i][j]] += 1

    for i in range(len(new_g)):
        for j in range(len(new_g[0])):
            new_g_hist[new_g[i][j]] += 1

    for i in range(len(new_r)):
        for j in range(len(new_r[0])):
            new_r_hist[new_r[i][j]] += 1

    plt.bar(range(256), new_b_hist, color='blue', width=1)
    plt.title('Equalized Blue Histogram')
    plt.show()

    plt.bar(range(256), new_g_hist, color='green', width=1)
    plt.title('Equalized Green Histogram')
    plt.show()
    

    plt.bar(range(256), new_r_hist, color='red', width=1)
    plt.title('Equalized Red Histogram')
    plt.show()

    cv2.imshow('Equalized_img', Equalized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#--------------------------------main---------------------------------#

img = cv2.imread('imgs/img3.png', 0)
equalize_gray_image(img)

img2 = cv2.imread('imgs/img2.jpg', 1)  
equalize_rgb_image(img2)

img4 = cv2.imread('imgs/img4.png', 1)
equalize_rgb_image(img4)


