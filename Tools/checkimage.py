# -*- coding: utf-8 -*-
import cv2
import os

def read_directory(directory_name):
    i=0    
    for filename in os.listdir(directory_name):
        #print(filename)  # 仅仅是为了测试
        img = cv2.imread(directory_name + "/" + filename)
        print(img.shape)
        i=i+1
        #####显示图片#######
        print(img.shape)
        if(img.shape[2]==1):
            print('grey')            
            print(filename)
            print('grey')
        #elif(img.shape[2]==1):
          #  print(filename)
          #  print('grey')
#
        #cv2.imshow(filename, img)
        #cv2.waitKey(0)
        #####################
    print(i)

if __name__ == '__main__':  
    read_directory('./5')   
