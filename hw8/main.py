import numpy as np 
import cv2 

def SNR(img, img_noise):
    img_noise = img_noise/255
    img = img/255
    noise = img_noise - img
    vs = np.std(img)
    vn = np.std(noise)
    snr = 20 * np.log10(vs/vn)
    return snr

def gaussian_noise(img, amplitude):
    img = img.astype(np.float64)
    img += amplitude * np.random.normal(0,1,img.shape)
    np.clip(img,0,255,out=img)
    return img.astype(np.uint8)

def salt_and_pepper_noise(img, probability):
    img = img.astype(np.float64)
    mask = np.random.uniform(0,1,img.shape)
    img[mask>(1-probability)] = 255
    img[mask<probability] = 0
    return img.astype(np.uint8)

def box_filter(img, size):
    img = img.astype(np.float64)
    kernal = np.ones((size,size)) / (size*size)
    b = size//2
    img_padding = cv2.copyMakeBorder(img,b,b,b,b,cv2.BORDER_REFLECT)
    for r in range(img.shape[0]):
    	for c in range(img.shape[1]):
        	img[r,c] = np.sum(img_padding[r:r+size,c:c+size]*kernal)
    return img.astype(np.uint8)

def median_filter(img, size):
    img = img.astype(np.float64)
    kernal = np.ones((size,size))
    b = size//2
    img_padding = cv2.copyMakeBorder(img,b,b,b,b,cv2.BORDER_REFLECT)
    for r in range(img.shape[0]):
    	for c in range(img.shape[1]):
        	img[r,c] = np.median(img_padding[r:r+size,c:c+size]*kernal)
    return img.astype(np.uint8)

kernel = np.array(
   [[0,1,1,1,0],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [0,1,1,1,0]], dtype=np.bool)
    
def dilation(img):
    ret = np.zeros_like(img)
    img_padding = cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_REFLECT)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            ret[r,c] = img_padding[r:r+5,c:c+5][kernel].max()
    return ret
    
def erosion(img):
    ret = np.zeros_like(img)
    img_padding = cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_REFLECT)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            ret[r,c] = img_padding[r:r+5,c:c+5][kernel].min()
    return ret


def opening(img):
    return dilation(erosion(img))

def closing(img):
    return erosion(dilation(img))

if __name__ == "__main__":
    img = cv2.imread('lena.bmp',0)
    
    gaussian_noise_10 = gaussian_noise(img, 10)
    gaussian_noise_30 = gaussian_noise(img, 30)
    salt_and_pepper_noise_005 = salt_and_pepper_noise(img, 0.05)
    salt_and_pepper_noise_010 = salt_and_pepper_noise(img, 0.1)
    cv2.imwrite('./output/gaussian_noise_10.png',gaussian_noise_10)
    cv2.imwrite('./output/gaussian_noise_30.png',gaussian_noise_30)
    cv2.imwrite('./output/salt_and_pepper_noise_0.05.png',salt_and_pepper_noise_005)
    cv2.imwrite('./output/salt_and_pepper_noise_0.10.png',salt_and_pepper_noise_010)
    
    g_n_10_b_3 = box_filter(gaussian_noise_10,3)
    g_n_10_b_5 = box_filter(gaussian_noise_10,5)
    g_n_10_m_3 = median_filter(gaussian_noise_10,3)
    g_n_10_m_5 = median_filter(gaussian_noise_10,5)
    g_n_10_o_c = closing(opening(gaussian_noise_10))
    g_n_10_c_o = opening(closing(gaussian_noise_10))
    
    g_n_30_b_3 = box_filter(gaussian_noise_30,3)
    g_n_30_b_5 = box_filter(gaussian_noise_30,5)
    g_n_30_m_3 = median_filter(gaussian_noise_30,3)
    g_n_30_m_5 = median_filter(gaussian_noise_30,5)
    g_n_30_o_c = closing(opening(gaussian_noise_30))
    g_n_30_c_o = opening(closing(gaussian_noise_30))
    
    s_n_05_b_3 = box_filter(salt_and_pepper_noise_005,3)
    s_n_05_b_5 = box_filter(salt_and_pepper_noise_005,5)
    s_n_05_m_3 = median_filter(salt_and_pepper_noise_005,3)
    s_n_05_m_5 = median_filter(salt_and_pepper_noise_005,5)
    s_n_05_o_c = closing(opening(salt_and_pepper_noise_005))
    s_n_05_c_o = opening(closing(salt_and_pepper_noise_005))
    
    s_n_1_b_3 = box_filter(salt_and_pepper_noise_010,3)
    s_n_1_b_5 = box_filter(salt_and_pepper_noise_010,5)
    s_n_1_m_3 = median_filter(salt_and_pepper_noise_010,3)
    s_n_1_m_5 = median_filter(salt_and_pepper_noise_010,5)
    s_n_1_o_c = closing(opening(salt_and_pepper_noise_010))
    s_n_1_c_o = opening(closing(salt_and_pepper_noise_010))
    
    cv2.imwrite('./output/gaussian_noise_10_box3.png',g_n_10_b_3)
    cv2.imwrite('./output/gaussian_noise_10_box5.png',g_n_10_b_5)
    cv2.imwrite('./output/gaussian_noise_10_median3.png',g_n_10_m_3)
    cv2.imwrite('./output/gaussian_noise_10_median5.png',g_n_10_m_5)
    
    cv2.imwrite('./output/gaussian_noise_10_opening_closing.png',g_n_10_o_c)
    cv2.imwrite('./output/gaussian_noise_10_closing_opening.png',g_n_10_c_o)
    print(SNR(img,gaussian_noise_10),SNR(img,g_n_10_b_3),SNR(img,g_n_10_b_5),SNR(img,g_n_10_m_3),SNR(img,g_n_10_m_5),SNR(img,g_n_10_o_c),SNR(img,g_n_10_c_o))
    
    cv2.imwrite('./output/gaussian_noise_30_box3.png',g_n_30_b_3)
    cv2.imwrite('./output/gaussian_noise_30_box5.png',g_n_30_b_5)
    cv2.imwrite('./output/gaussian_noise_30_median3.png',g_n_30_m_3)
    cv2.imwrite('./output/gaussian_noise_30_median5.png',g_n_30_m_5)
    cv2.imwrite('./output/gaussian_noise_30_opening_closing.png',g_n_30_o_c)
    cv2.imwrite('./output/gaussian_noise_30_closing_opening.png',g_n_30_c_o)
    print(SNR(img,gaussian_noise_30),SNR(img,g_n_30_b_3),SNR(img,g_n_30_b_5),SNR(img,g_n_30_m_3),SNR(img,g_n_30_m_5),SNR(img,g_n_30_o_c),SNR(img,g_n_30_c_o))
    
    cv2.imwrite('./output/salt_and_pepper_noise_0.05_box3.png',s_n_05_b_3)
    cv2.imwrite('./output/salt_and_pepper_noise_0.05_box5.png',s_n_05_b_5)
    cv2.imwrite('./output/salt_and_pepper_noise_0.05_median3.png',s_n_05_m_3)
    cv2.imwrite('./output/salt_and_pepper_noise_0.05_median5.png',s_n_05_m_5)
    cv2.imwrite('./output/salt_and_pepper_noise_0.05_opening_closing.png',s_n_05_o_c)
    cv2.imwrite('./output/salt_and_pepper_noise_0.05_closing_opening.png',s_n_05_c_o)
    print(SNR(img,salt_and_pepper_noise_005),SNR(img,s_n_05_b_3),SNR(img,s_n_05_b_5),SNR(img,s_n_05_m_3),SNR(img,s_n_05_m_5),SNR(img,s_n_05_o_c),SNR(img,s_n_05_c_o))
    
    cv2.imwrite('./output/salt_and_pepper_noise_0.10_box3.png',s_n_1_b_3)
    cv2.imwrite('./output/salt_and_pepper_noise_0.10_box5.png',s_n_1_b_5)
    cv2.imwrite('./output/salt_and_pepper_noise_0.10_median3.png',s_n_1_m_3)
    cv2.imwrite('./output/salt_and_pepper_noise_0.10_median5.png',s_n_1_m_5)
    cv2.imwrite('./output/salt_and_pepper_noise_0.10_opening_closing.png',s_n_1_o_c)    
    cv2.imwrite('./output/salt_and_pepper_noise_0.10_closing_opening.png',s_n_1_c_o)
    print(SNR(img,salt_and_pepper_noise_010),SNR(img,s_n_1_b_3),SNR(img,s_n_1_b_5),SNR(img,s_n_1_m_3),SNR(img,s_n_1_m_5),SNR(img,s_n_1_o_c),SNR(img,s_n_1_c_o))
    
