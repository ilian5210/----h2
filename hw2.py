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
    if type == 'Lightness':
        c = 'gray'
    else : c = type
    plt.bar(range(256), hist, color=c, width=1)
    plt.title('Original 'f'{type} Histogram')
    plt.xlim(0, 255)
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

def rgb2hsl(img):
    b, g, r = cv2.split(img)
    b = b.astype(np.float32) / 255.0
    g = g.astype(np.float32) / 255.0
    r = r.astype(np.float32) / 255.0

    max_val = np.maximum(r, np.maximum(g, b))
    min_val = np.minimum(r, np.minimum(g, b))


    h = np.zeros((len(img), len(img[0])), np.float32)
    s = np.zeros((len(img), len(img[0])), np.float32)
    l = np.zeros((len(img), len(img[0])), np.float32)

    for i in range(len(img)):
        for j in range(len(img[0])):
            if max_val[i][j] == min_val[i][j]:
                h[i][j] = 0
            elif max_val[i][j] == r[i][j]:
                h[i][j] = 60 * (g[i][j] - b[i][j]) / (max_val[i][j] - min_val[i][j])
            elif max_val[i][j] == g[i][j]:
                h[i][j] = 60 * (b[i][j] - r[i][j]) / (max_val[i][j] - min_val[i][j]) + 120
            elif max_val[i][j] == b[i][j]:
                h[i][j] = 60 * (r[i][j] - g[i][j]) / (max_val[i][j] - min_val[i][j]) + 240

            if h[i][j] < 0:
                h[i][j] += 360

            l[i][j] = (max_val[i][j] + min_val[i][j]) / 2

            if max_val[i][j] == min_val[i][j]:
                s[i][j] = 0
            elif l[i][j] <= 0.5:
                s[i][j] = (max_val[i][j] - min_val[i][j]) / (max_val[i][j] + min_val[i][j])
            elif l[i][j] > 0.5:
                s[i][j] = (max_val[i][j] - min_val[i][j]) / (2 - max_val[i][j] - min_val[i][j])

    h = np.clip(h, 0, 360).astype(np.float32)
    s = np.clip(s, 0, 1).astype(np.float32)
    l = np.clip(l, 0, 1).astype(np.float32)

    return h, s, l

#=====================================================================#

def hsl2rgb(h, s, l):
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    for i in range(len(h)):
        for j in range(len(h[0])):
            c = (1 - abs(2 * l[i][j] - 1)) * s[i][j]
            x = c * (1 - abs((h[i][j] / 60) % 2 - 1))
            m = l[i][j] - c / 2

            if h[i][j] < 60:
                r[i][j] = c
                g[i][j] = x
                b[i][j] = 0
            elif h[i][j] < 120:
                r[i][j] = x
                g[i][j] = c
                b[i][j] = 0
            elif h[i][j] < 180:
                r[i][j] = 0
                g[i][j] = c
                b[i][j] = x
            elif h[i][j] < 240:
                r[i][j] = 0
                g[i][j] = x
                b[i][j] = c
            elif h[i][j] < 300:
                r[i][j] = x
                g[i][j] = 0
                b[i][j] = c
            elif h[i][j] < 360:
                r[i][j] = c
                g[i][j] = 0
                b[i][j] = x

            r[i][j] = (r[i][j] + m) * 255.0
            g[i][j] = (g[i][j] + m) * 255.0
            b[i][j] = (b[i][j] + m) * 255.0

    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)

    return cv2.merge((b, g, r))

#=====================================================================#

def equalize_gray_image(img):
    cv2.imshow('OrImg', img)
    cv2.waitKey(0)

    hist = img2hist(img, 'Gray')
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
    plt.xlim(0, 255)
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
    plt.xlim(0, 255)
    plt.show()

    plt.bar(range(256), new_g_hist, color='green', width=1)
    plt.title('Equalized Green Histogram')
    plt.xlim(0, 255)
    plt.show()
    

    plt.bar(range(256), new_r_hist, color='red', width=1)
    plt.title('Equalized Red Histogram')
    plt.xlim(0, 255)
    plt.show()

    cv2.imshow('Equalized_img', Equalized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#=====================================================================#
def equalize_hsl_image(img):
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)

    h, s, l = rgb2hsl(img)

    l_int = (l * 255).astype(np.uint8)

    l_hist = img2hist(l_int, 'Lightness')
    l_cdf_hist = hist2cdf(l_hist)
    l_sum_val = sum_hist(l_hist)

    new_l = np.zeros_like(l_int)
    for i in range(len(l_int)):
        for j in range(len(l_int[0])):
            new_l[i][j] = 255 * l_cdf_hist[l_int[i][j]] / l_sum_val

    new_l = np.clip(new_l, 0, 255).astype(np.float32) / 255.0

    new_l_hist = np.zeros(256)

    for i in range(len(new_l)):
        for j in range(len(new_l[0])):
            new_l_hist[int(new_l[i][j] * 255)] += 1

    plt.bar(range(256), new_l_hist, color='gray', width=1)
    plt.title('Equalized Lightness Histogram')
    plt.xlim(0, 255)
    plt.show()


    equalized_img = hsl2rgb(h, s, new_l)

    cv2.imshow('HSL Equalized Image', equalized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return equalized_img

#--------------------------------main---------------------------------#

img1 = cv2.imread('imgs/img1.jpg', 0)
equalize_gray_image(img1)

img1 = cv2.imread('imgs/img1.jpg', 1)  
equalize_rgb_image(img1)

img1 = cv2.imread('imgs/img1.jpg', 1)
equalize_hsl_image(img1)

img2 = cv2.imread('imgs/img2.png', 0)
equalize_gray_image(img2)

img2 = cv2.imread('imgs/img2.png', 1)
equalize_rgb_image(img2)

img2 = cv2.imread('imgs/img2.png', 1)
equalize_hsl_image(img2)



