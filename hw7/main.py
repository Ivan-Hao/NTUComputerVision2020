import matplotlib.pyplot as plt
import numpy as np 
import cv2 

def h(b,c,d,e):
    if b == c and (b != d or b != e):
        return 'q'
    if b == c and (b == d and b == e):
        return 'r'
    return 's'

def f(a1,a2,a3,a4):
    return 5 if a1 == a2 == a3 == a4 == 'r' else [a1,a2,a3,a4].count('q')

def Yokoi_connectivity_number(img):
    img_padding = np.pad(img, ((2,2),(2,2)), 'constant',constant_values = (0,0))
    ret = np.zeros_like(img)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if img[r,c] != 255:
                continue
            a1 = h(img_padding[r+2,c+2], img_padding[r+2,c+3], img_padding[r+1,c+3], img_padding[r+1,c+2])
            a2 = h(img_padding[r+2,c+2], img_padding[r+1,c+2], img_padding[r+1,c+1], img_padding[r+2,c+1])
            a3 = h(img_padding[r+2,c+2], img_padding[r+2,c+1], img_padding[r+3,c+1], img_padding[r+3,c+2])
            a4 = h(img_padding[r+2,c+2], img_padding[r+3,c+2], img_padding[r+3,c+3], img_padding[r+2,c+3])
            ret[r,c] = f(a1,a2,a3,a4)
    return ret

def H(a, m): # do after the yokoi
    return 1 if a == m else 0

def Y(x0, x1, x2, x3, x4, m = 1): # do after the yokoi
    sum_ = H(x1,m) + H(x2,m) + H(x3,m) + H(x4,m)
    return 'p' if sum_ > 0 and x0 == m else 'q'

def pair_relationship(img_Yokoi):
    img_padding = np.pad(img_Yokoi, ((1,1),(1,1)), 'constant',constant_values = (0,0))
    ret = np.full_like(img_Yokoi, ' ', dtype='<U1')
    for r in range(img_Yokoi.shape[0]):
        for c in range(img_Yokoi.shape[1]):
            ret[r,c] = Y(img_padding[r+1,c+1], img_padding[r+1,c+2], img_padding[r,c+1], img_padding[r+1,c], img_padding[r+2,c+1])
    return ret

def G(b,c,d,e):
    return 1 if b == c and (b != d or b != e) else 0

def connected_shrink(img, img_pair):

    img_padding = np.pad(img, ((2,2),(2,2)), 'constant',constant_values = (0,0))
    change = False
    for r in range(img_pair.shape[0]):
        for c in range(img_pair.shape[1]):
            if img_pair[r,c] != 'p':
                continue
            a1 = G(img_padding[r+2,c+2], img_padding[r+2,c+3], img_padding[r+1,c+3], img_padding[r+1,c+2])
            a2 = G(img_padding[r+2,c+2], img_padding[r+1,c+2], img_padding[r+1,c+1], img_padding[r+2,c+1])
            a3 = G(img_padding[r+2,c+2], img_padding[r+2,c+1], img_padding[r+3,c+1], img_padding[r+3,c+2])
            a4 = G(img_padding[r+2,c+2], img_padding[r+3,c+2], img_padding[r+3,c+3], img_padding[r+2,c+3])
            if a1 + a2 + a3 + a4 == 1:
                img_padding[r+2,c+2] = 0
                change = True
                
    return img_padding[2:-2,2:-2], change



if __name__ == "__main__":
    img = cv2.imread('lena.bmp',0)
    img = np.where(img>127, 255, 0).astype(np.uint8) # binarilize
    img = img[::8,::8] # down sampling

    out = cv2.VideoWriter('./output/video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 2, img.shape)

    change = True
    while change:
        change = False
        v = cv2.merge([img,img,img])
        out.write(v)
        img_Yokoi =  Yokoi_connectivity_number(img)
        img_pair = pair_relationship(img_Yokoi)
        img, change = connected_shrink(img, img_pair)

    cv2.imwrite('./output/thinning.png', img)
    out.release()