# -*- coding: utf-8 -*-
import os, random, shutil
def moveFile(fileDir):
        pathDir = os.listdir(fileDir)  
        filenumber=len(pathDir)
        rate=0.5    
        picknumber=int(filenumber*rate) 
        sample = random.sample(pathDir, picknumber)  
        print (sample)
        for name in sample:
                shutil.move(fileDir+name, tarDir+name)
        return

if __name__ == '__main__':
	fileDir = "./data/images/test/9/"  
	tarDir = './testimg/9/'   
	moveFile(fileDir)

# ./data/images/train/2/
# ./data/images/test/2/
# ./testimg/1/
