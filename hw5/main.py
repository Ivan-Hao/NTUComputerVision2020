import numpy as np 
import cv2 

def solve(img, choice, kernel):
    if choice == 0: #  Dilation
        img_padding = np.pad(img,((2,2),(2,2)),'constant',constant_values = (0,0))
        ret = img_padding.copy()
        row,col = img.shape
        local = kernel.astype(np.bool)
        for r in range(row):
            for c in range(col):
                if img[r,c] == 0:
                    continue
                local_max = img_padding[r:r+5,c:c+5][local].max()
                fill = ret[r:r+5,c:c+5] < (local_max * kernel)
                ret[r:r+5,c:c+5][fill] = local_max
        return ret[2:-2,2:-2]
        
    elif choice == 1: # Erosion
        img_padding = np.pad(img,((2,2),(2,2)),'constant',constant_values = (0,0))
        ret = np.zeros_like(img)
        row,col = img.shape
        local = kernel.astype(np.bool)
        for r in range(row):
            for c in range(col):
                if img_padding[r:r+5,c:c+5][local].all():
                    ret[r,c] = img_padding[r:r+5,c:c+5][local].min()
        return ret

    elif choice == 2: # Opening
        return solve(solve(img,1,kernel),0,kernel)
        
    elif choice == 3: # Closing
        return solve(solve(img,0,kernel),1,kernel)            
    else:
        print("invalidated index!")

if __name__ == "__main__":
    kernel = np.array([
        [0,1,1,1,0],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [0,1,1,1,0] ], dtype=np.uint8)

    img = cv2.imread('lena.bmp',0)
    a = solve(img,0,kernel)
    cv2.imwrite('./output/dilation.png',a)
    b = solve(img,1,kernel)
    cv2.imwrite('./output/erosion.png',b)
    c = solve(img,2,kernel)
    cv2.imwrite('./output/opening.png',c)
    d = solve(img,3,kernel)
    cv2.imwrite('./output/closing.png',d)
    
    