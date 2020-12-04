import numpy as np
import cv2

def Roberts(img,threshold=30):
    img = img.astype(np.float)
    r1 = np.array([[-1,0],[0,1]],dtype=np.float)
    r2 = np.array([[0,-1],[1,0]],dtype=np.float)
    img_return = np.full_like(img,255,dtype=np.uint8)
    for r in range(img.shape[0]-1):
        for c in range(img.shape[1]-1):
            magnitude_r1 = np.sum(img[r:r+2,c:c+2]*r1)
            magnitude_r2 = np.sum(img[r:r+2,c:c+2]*r2)
            if np.sqrt(magnitude_r1**2 + magnitude_r2**2) >= threshold:
                img_return[r,c] = 0
    return img_return

def Prewitt(img,threshold=24):
    p1 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]],dtype=np.float)
    p2 = p1.T
    img_padding = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE).astype(np.float)
    img_return = np.full_like(img,255,dtype=np.uint8)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            magnitude_p1 = np.sum(img_padding[r:r+3,c:c+3]*p1)
            magnitude_p2 = np.sum(img_padding[r:r+3,c:c+3]*p2)
            if np.sqrt(magnitude_p1**2 + magnitude_p2**2) >= threshold:
                img_return[r,c] = 0
    return img_return

def Sobel(img,threshold=38):
    s1 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=np.float)
    s2 = s1.T
    img_padding = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE).astype(np.float)
    img_return = np.full_like(img,255,dtype=np.uint8)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            magnitude_s1 = np.sum(img_padding[r:r+3,c:c+3]*s1)
            magnitude_s2 = np.sum(img_padding[r:r+3,c:c+3]*s2)
            if np.sqrt(magnitude_s1**2 + magnitude_s2**2) >= threshold:
                img_return[r,c] = 0
    return img_return

def Frei_and_Chen(img,threshold=30):
    f_c1 = np.array([[-1,-np.sqrt(2),-1],[0,0,0],[1,np.sqrt(2),1]],dtype=np.float)
    f_c2 = f_c1.T
    img_padding = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE).astype(np.float)
    img_return = np.full_like(img,255,dtype=np.uint8)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            magnitude_s1 = np.sum(img_padding[r:r+3,c:c+3]*f_c1)
            magnitude_s2 = np.sum(img_padding[r:r+3,c:c+3]*f_c2)
            if np.sqrt(magnitude_s1**2 + magnitude_s2**2) >= threshold:
                img_return[r,c] = 0
    return img_return

def Kirsch(img,threshold=135):
    k0 = np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]],dtype=np.float)
    k1 = np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]],dtype=np.float)
    k2 = np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]],dtype=np.float)
    k3 = np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]],dtype=np.float)
    k4 = np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]],dtype=np.float)
    k5 = np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]],dtype=np.float)
    k6 = np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]],dtype=np.float)
    k7 = np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]],dtype=np.float)
    img_padding = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE).astype(np.float)
    img_return = np.full_like(img,255,dtype=np.uint8)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            magnitude_k0 = np.sum(img_padding[r:r+3,c:c+3]*k0)
            magnitude_k1 = np.sum(img_padding[r:r+3,c:c+3]*k1)
            magnitude_k2 = np.sum(img_padding[r:r+3,c:c+3]*k2)
            magnitude_k3 = np.sum(img_padding[r:r+3,c:c+3]*k3)
            magnitude_k4 = np.sum(img_padding[r:r+3,c:c+3]*k4)
            magnitude_k5 = np.sum(img_padding[r:r+3,c:c+3]*k5)
            magnitude_k6 = np.sum(img_padding[r:r+3,c:c+3]*k6)
            magnitude_k7 = np.sum(img_padding[r:r+3,c:c+3]*k7)
            if max(magnitude_k0,magnitude_k1,magnitude_k2,magnitude_k3,magnitude_k4,magnitude_k5,magnitude_k6,magnitude_k7) >= threshold:
                img_return[r,c] = 0
    return img_return

def Robinson(img,threshold=43):
    r0 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float)
    r1 = np.array([[0,1,2],[-1,0,1],[-2,-1,0]],dtype=np.float)
    r2 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=np.float)
    r3 = np.array([[2,1,0],[1,0,-1],[0,-1,-2]],dtype=np.float)
    r4 = np.negative(r0)
    r5 = np.negative(r1)
    r6 = np.negative(r2)
    r7 = np.negative(r3)
    img_padding = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE).astype(np.float)
    img_return = np.full_like(img,255,dtype=np.uint8)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            magnitude_r0 = np.sum(img_padding[r:r+3,c:c+3]*r0)
            magnitude_r1 = np.sum(img_padding[r:r+3,c:c+3]*r1)
            magnitude_r2 = np.sum(img_padding[r:r+3,c:c+3]*r2)
            magnitude_r3 = np.sum(img_padding[r:r+3,c:c+3]*r3)
            magnitude_r4 = np.sum(img_padding[r:r+3,c:c+3]*r4)
            magnitude_r5 = np.sum(img_padding[r:r+3,c:c+3]*r5)
            magnitude_r6 = np.sum(img_padding[r:r+3,c:c+3]*r6)
            magnitude_r7 = np.sum(img_padding[r:r+3,c:c+3]*r7)
            if max(magnitude_r0,magnitude_r1,magnitude_r2,magnitude_r3,magnitude_r4,magnitude_r5,magnitude_r6,magnitude_r7) >= threshold:
                img_return[r,c] = 0
    return img_return

def Nevatia_Babu(img,threshold=12500):
    n0 = np.array([[100,100,100,100,100],[100,100,100,100,100],[0,0,0,0,0],[-100,-100,-100,-100,-100],[-100,-100,-100,-100,-100]],dtype=np.float)
    n1 = np.negative(n0.T)
    n2 = np.array([[100,100,100,100,100],[100,100,100,78,-32],[100,92,0,-92,-100],[32,-78,-100,-100,-100],[-100,-100,-100,-100,-100]],dtype=np.float)
    n3 = n2.T
    n4 = np.array([[100,100,100,100,100],[-32,78,100,100,100],[-100,-92,0,92,100],[-100,-100,-100,-78,32],[-100,-100,-100,-100,-100]],dtype=np.float)
    n5 = np.negative(n4.T)

    img_padding = cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_REPLICATE).astype(np.float)
    img_return = np.full_like(img,255,dtype=np.uint8)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            magnitude_n0 = np.sum(img_padding[r:r+5,c:c+5]*n0)
            magnitude_n1 = np.sum(img_padding[r:r+5,c:c+5]*n1)
            magnitude_n2 = np.sum(img_padding[r:r+5,c:c+5]*n2)
            magnitude_n3 = np.sum(img_padding[r:r+5,c:c+5]*n3)
            magnitude_n4 = np.sum(img_padding[r:r+5,c:c+5]*n4)
            magnitude_n5 = np.sum(img_padding[r:r+5,c:c+5]*n5)
            if max(magnitude_n0,magnitude_n1,magnitude_n2,magnitude_n3,magnitude_n4,magnitude_n5) >= threshold:
                img_return[r,c] = 0
    return img_return

if __name__ == "__main__":
    img = cv2.imread('lena.bmp',0)
    cv2.imwrite('./output/Roberts.png',Roberts(img))
    cv2.imwrite('./output/Prewitt.png',Prewitt(img))
    cv2.imwrite('./output/Sobel.png',Sobel(img))
    cv2.imwrite('./output/Frei_and_Chen.png',Frei_and_Chen(img))
    cv2.imwrite('./output/Kirsch.png',Kirsch(img))
    cv2.imwrite('./output/Robinson.png',Robinson(img))
    cv2.imwrite('./output/Nevatia_Babu.png',Nevatia_Babu(img))