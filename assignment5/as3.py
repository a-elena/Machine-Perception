import numpy as np
import cv2
from a2_utils import *
from a3_utils import *
from UZ_utils import *




def filter(I, kernel):

    return cv2.filter2D(I, -1, kernel)


def gauss(sigma):
	x = np.arange(np.floor(-3 * sigma), np.ceil(3 * sigma + 1))
	k = np.exp(-(x ** 2) / (2 * sigma ** 2))
	k = k / np.sum(k)
	return np.expand_dims(k, 0)


def gaussdx(sigma):
	x = np.arange(np.floor(-3 * sigma), np.ceil(3 * sigma + 1))
	k = -x * np.exp(-(x ** 2) / (2 * sigma ** 2))
	k /= np.sum(np.abs(k))
	return np.expand_dims(k, 0)
    


def first_derivative(I, sigma=1):

    G = gauss(sigma)
    GT = G.T
    G = np.flip(G)
    GT = np.flip(GT)
    
    D = gaussdx(sigma)
    DT = D.T
    D = np.flip(D)
    DT = np.flip(DT)
    
    dx = filter(filter(I, GT), D)
    dy = filter(filter(I, DT), G)
    
    return dx, dy


def second_derivative(I, sigma=1):
    
    I_dx, I_dy = first_derivative(I, sigma)
    I_dxx, I_dyx = first_derivative(I_dx, sigma)
    I_dxy, I_dyy = first_derivative(I_dy, sigma)
    
    return I_dxx, I_dxy, I_dyy


def gradient_magnitude(I, sigma=1):

    dx, dy = first_derivative(I, sigma)
    m = np.sqrt(dx**2 + dy**2)
    a = np.arctan2(dy,dx)

    return m, a