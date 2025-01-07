import numpy as np
import cv2
from matplotlib import pyplot as plt
from UZ_utils import *
from assignement1 import *




I2 = imread('images/bird.jpg')
I2_gray = convert_to_gray(I2)

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.imshow(I2)
plt.title('Original image') 
plt.subplot(1,2,2)
plt.imshow(I2_gray, cmap="gray")
plt.title('Gray image')
plt.show()

threshold = 0.3
I_t1 = np.copy(I2_gray)
I_t1[I_t1 < threshold] = 0
I_t1[I_t1 >= threshold] = 1

I_t2 = np.where(I2_gray < threshold, 0, 1)

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.imshow(I_t1, cmap="gray")
plt.title("Method 1, threshold 0.3")
plt.subplot(1,2,2) 
plt.imshow(I_t2, cmap="gray")
plt.title("Method 2, threshold 0.3")
plt.show()

threshold = 0.25
I_t1 = np.copy(I2_gray)
I_t1[I_t1 < threshold] = 0
I_t1[I_t1 >= threshold] = 1

I_t2 = np.where(I2_gray < threshold, 0, 1)

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.imshow(I_t1, cmap="gray")
plt.title("Method 1, threshold 0.25")
plt.subplot(1,2,2) 
plt.imshow(I_t2, cmap="gray")
plt.title("Method 2, threshold 0.25")
plt.show()


threshold = 0.2
I_t1 = np.copy(I2_gray)
I_t1[I_t1 < threshold] = 0
I_t1[I_t1 >= threshold] = 1

I_t2 = np.where(I2_gray < threshold, 0, 1)

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.imshow(I_t1, cmap="gray")
plt.title("Method 1, threshold 0.2")
plt.subplot(1,2,2) 
plt.imshow(I_t2, cmap="gray")
plt.title("Method 2, threshold 0.2")
plt.show()


def make_mask(image, threshold):
    mask = np.copy(image)
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 1
    return mask    



def myhist(img_gray, num_bins):
    H = np.zeros(num_bins)
    I = img_gray.reshape(-1)
    part = 1/num_bins

    for x in I: 
        if x == 1:
            H[num_bins-1] += 1
        else:
            i = int(x/part)
            H[i] += 1
    return H / np.sum(H)




H20 = myhist(I2_gray, 20)
H100 = myhist(I2_gray, 100)

plt.imshow(I2_gray, cmap="gray")
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.bar(range(20), H20, edgecolor='blue')
plt.title('20 bins')
plt.subplot(1,2,2) 
plt.bar(range(100), H100, edgecolor='blue')
plt.title('100 bins')
plt.show()


def myhist_upgrade(img_gray, num_bins):
    H = np.zeros(num_bins)
    I = img_gray.reshape(-1)
    min = np.min(I)
    max = np.max(I)
    
    part = (max-min)/num_bins

    for x in I: 
        if x == max:
            H[num_bins-1] += 1
        else:
            i = int((x-min)/part)
            H[i] += 1
    return H / np.sum(H)




H10 = myhist(I2_gray, 10)
H10_new = myhist_upgrade(I2_gray, 10)

I = I2_gray.reshape(-1)
min = np.min(I2_gray)
max = np.max(I2_gray)
print("Min. value: ",min,", Max. value: ",max)

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.bar(range(10), H10, edgecolor='blue')
plt.title('Old myhist fun')
plt.subplot(1,2,2) 
plt.bar(range(10), H10_new, edgecolor='blue')
plt.title('New myhist fun')
plt.show()


img = imread('images/test.jpg')
img1 = convert_to_gray(img)

H10 = myhist(img1, 10)
H10_new = myhist_upgrade(img1, 10)

I = img1.reshape(-1)
min = np.min(img1)
max = np.max(img1)
print("Min. value: ",min,", Max. value: ",max)

plt.imshow(img1, cmap="gray")
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.bar(range(10), H10, edgecolor='blue')
plt.title('Old myhist fun')
plt.subplot(1,2,2) 
plt.bar(range(10), H10_new, edgecolor='blue')
plt.title('New myhist fun')
plt.show()


img = imread('images/test1.jpg')
img1 = convert_to_gray(img)

H100 = myhist(img1, 100)
H100_new = myhist_upgrade(img1, 100)

I = img1.reshape(-1)
min = np.min(img1)
max = np.max(img1)
print("Min. value: ",min,", Max. value: ",max)

plt.imshow(img1, cmap="gray")
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.bar(range(100), H100, edgecolor='blue')
plt.title('Old myhist fun')
plt.subplot(1,2,2) 
plt.bar(range(100), H100_new, edgecolor='blue')
plt.title('New myhist fun')
plt.show()



img = imread('images/test2.jpg')
img1 = convert_to_gray(img)

H100 = myhist(img1, 100)
H100_new = myhist_upgrade(img1, 100)

I = img1.reshape(-1)
min = np.min(img1)
max = np.max(img1)
print("Min. value: ",min,", Max. value: ",max)

plt.imshow(img1, cmap="gray")
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.bar(range(100), H100, edgecolor='blue')
plt.title('Old myhist fun')
plt.subplot(1,2,2) 
plt.bar(range(100), H100_new, edgecolor='blue')
plt.title('New myhist fun')
plt.show()


image1 = imread("images/image1.jpg")
g1 = convert_to_gray(image1)
image2 = imread("images/image2.jpg")
g2 = convert_to_gray(image2)
image3 = imread("images/image3.jpg")
g3 = convert_to_gray(image3)
image4 = imread("images/image4.jpg")
g4 = convert_to_gray(image4)

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.imshow(image1)
plt.subplot(1,2,2) 
plt.imshow(image2)
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.imshow(image3)
plt.subplot(1,2,2) 
plt.imshow(image4)
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.imshow(g1, cmap="gray")
plt.subplot(1,2,2) 
plt.imshow(g2, cmap="gray")
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.imshow(g3, cmap="gray")
plt.subplot(1,2,2) 
plt.imshow(g4, cmap="gray")
plt.show()


H1 = myhist(g1, 20)
H2 = myhist(g2, 20)
H3 = myhist(g3, 20)
H4 = myhist(g4, 20)

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.bar(range(20), H1, edgecolor='blue')
plt.subplot(1,2,2) 
plt.bar(range(20), H2, edgecolor='blue')
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.bar(range(20), H3, edgecolor='blue')
plt.subplot(1,2,2) 
plt.bar(range(20), H4, edgecolor='blue')
plt.show()

H1 = myhist(g1, 100)
H2 = myhist(g2, 100)
H3 = myhist(g3, 100)
H4 = myhist(g4, 100)

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.bar(range(100), H1, edgecolor='blue')
plt.subplot(1,2,2) 
plt.bar(range(100), H2, edgecolor='blue')
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.bar(range(100), H3, edgecolor='blue')
plt.subplot(1,2,2) 
plt.bar(range(100), H4, edgecolor='blue')
plt.show()


import numpy as np

def otsu(image):
    
    H = myhist(image, 255)
    vkupno = np.sum(H) 
    sum_all = np.dot(np.arange(len(H)), H) 
    sum = 0 
    n1 = 0
    max = 0
    threshold = 0 
    n12 = vkupno 

    for t in range(len(H)):
        n1 += H[t] 
        n2 = vkupno - n1 
        
        if n1 == 0 or n2 == 0: 
            continue

        sum += t * H[t] 
        m1 = sum / n1
        m2 = (sum_all - sum) / n2 

        variance = n1 * n2 * (m1 - m2) ** 2

        if variance > max:
            max = variance
            threshold = t

    return threshold / 255

image = imread_gray("images/bird.jpg")
threshold = otsu(image)

bird_mask = make_mask(image, threshold)

plt.imshow(bird_mask, cmap="gray")
plt.title("Mask with otsu threshold")
plt.show()


