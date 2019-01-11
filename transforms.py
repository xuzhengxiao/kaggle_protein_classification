import random
import cv2
import numpy as np

def rand0(s):
    return random.random()*(s*2)-s



def rotate_cv(im,deg,mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA):
    r, c,_ = im.shape
    M = cv2.getRotationMatrix2D((c // 2, r // 2), deg, 1)
    return cv2.warpAffine(im, M, (c, r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS + interpolation)


def RandomRotate(img,deg,p=0.75,mode=cv2.BORDER_REFLECT):
    """ Rotates images ."""
    rdeg=rand0(deg)
    rp=random.random()<p
    if rp:
      img=rotate_cv(img,rdeg,mode)
    return img


def RandomDihedral(x):
    """
    Rotates images by random multiples of 90 degrees and/or reflection.
    """
    rot_times = random.randint(0, 3)
    do_flip = random.random() < 0.5
    x = np.rot90(x, rot_times)
    return np.fliplr(x).copy() if do_flip else x

def lighting(im, b, c):
    """ Adjust image balance and contrast """
    if b==0 and c==1: return im
    mu = np.average(im)
    return np.clip((im-mu)*c+mu+b,0.,1.).astype(np.float32)

def RandomLighting(x,b=0.05,c=0.05):
    b=rand0(b)
    c=rand0(c)
    c = -1 / (c - 1) if c < 0 else c + 1
    x = lighting(x, b, c)
    return x

def Normalize(x,m,s):
    x=(x-m)/s
    return x