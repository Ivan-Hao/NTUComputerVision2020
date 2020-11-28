import numpy as np 
import cv2 
import matplotlib.pyplot as plt

def part1(img, choice):
    if choice == 0: #  original image and its histogram
        statistic = np.zeros(256)
        r, c = img.shape
        for i in range(r):
            for j in range(c):
                statistic[img[i,j]] += 1
        cv2.imwrite('./output/1.jpg',img)
        plt.style.use('seaborn-white')
        plt.bar(range(256) ,statistic)
        plt.xlabel('pixel value')
        plt.ylabel('number')
        plt.savefig('./output/1_histogram.jpg')
        plt.clf()
        
    elif choice == 1: # image with intensity divided by 3 and its histogram
        ret = img//3
        statistic = np.zeros(256)
        r, c = ret.shape
        for i in range(r):
            for j in range(c):
                statistic[ret[i,j]] += 1
        cv2.imwrite('./output/2.jpg',ret)
        plt.style.use('seaborn-white')
        plt.bar(range(256) ,statistic)
        plt.xlabel('pixel value')
        plt.ylabel('number')
        plt.savefig('./output/2_histogram.jpg')
        plt.clf()
        
    elif choice == 2: # image after applying histogram equalization to (b) and its histogram
        img = img//3
        ret = np.zeros_like(img)
        statistic = np.zeros(256)
        statistic_ret = np.zeros(256)
        r, c = ret.shape
        for i in range(r):
            for j in range(c):
                statistic[img[i,j]] += 1
        statistic = np.cumsum(statistic)        
        cdf_min = min(statistic)
        cdf_max = max(statistic)

        for i in range(r):
            for j in range(c):
                ret[i,j] = round((statistic[img[i,j]] - cdf_min)/(cdf_max-cdf_min)*255)
                statistic_ret[ret[i,j]] += 1
        cv2.imwrite('./output/3.jpg',ret)
        plt.style.use('seaborn-white')
        plt.bar(range(256) ,statistic_ret)
        plt.xlabel('pixel value')
        plt.ylabel('number')
        plt.savefig('./output/3_histogram.jpg')
        
    else:
        print("invalidated index!")

if __name__ == "__main__":
    img = cv2.imread('lena.bmp',0)
    part1(img.copy(),0)
    part1(img.copy(),1)
    part1(img.copy(),2)