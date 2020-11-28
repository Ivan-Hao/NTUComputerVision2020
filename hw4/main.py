import numpy as np 
import cv2 

def solve(img, choice, kernel):
    if choice == 0: #  Dilation
        img_padding = np.pad(img,((2,2),(2,2)),'constant',constant_values = (0,0))
        ret = np.zeros_like(img_padding)
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                if img[r,c] != 255:
                    continue
                ret[r:r+5,c:c+5][kernel] = 255
        return ret[2:-2,2:-2]
         
    elif choice == 1: # Erosion
        img_padding = np.pad(img,((2,2),(2,2)),'constant',constant_values = (0,0))
        ret = np.zeros_like(img)
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                if img_padding[r:r+5,c:c+5][kernel].all():
                    ret[r,c] = 255 
        return ret

    elif choice == 2: # Opening
        return solve(solve(img,1,kernel),0,kernel)
        
    elif choice == 3: # Closing
        return solve(solve(img,0,kernel),1,kernel)
        
    elif choice == 4: # Hit-and-miss transform
        j_kernel = np.array([   
            [0,0,0,0,0],
            [0,0,0,0,0],
            [1,1,0,0,0],
            [0,1,0,0,0],
            [0,0,0,0,0] ], dtype=np.bool)
        k_kernel = np.array([
            [0,0,0,0,0],
            [0,1,1,0,0],
            [0,0,1,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0] ], dtype=np.bool)
        reverse = -img + 255
        return solve(img,1,j_kernel) & solve(reverse,1,k_kernel)
        
    else:
        print("invalidated index!")

if __name__ == "__main__":
    kernel = np.array([
        [0,1,1,1,0],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [0,1,1,1,0]], dtype=np.bool)

    img = cv2.imread('lena.bmp',0)
    img = np.where(img>127, 255, 0)
    a = solve(img.copy(),0,kernel)
    cv2.imwrite('./output/dilation.png',a)
    b = solve(img.copy(),1,kernel)
    cv2.imwrite('./output/erosion.png',b)
    c = solve(img.copy(),2,kernel)
    cv2.imwrite('./output/opening.png',c)
    d = solve(img.copy(),3,kernel)
    cv2.imwrite('./output/closing.png',d)
    e = solve(img.copy(),4,None)
    cv2.imwrite('./output/hitandmiss.png',e)
    