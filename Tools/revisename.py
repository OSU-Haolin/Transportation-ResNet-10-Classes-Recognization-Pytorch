# -*- coding: utf-8 -*-
import os, random, shutil

path = './testimg/9/' 
fileList=os.listdir(path)

n=0
for i in fileList:
    

    oldname=path + os.sep + fileList[n]   
    
    newname=path + os.sep +'9'+str(n+1)+'.JPG'
    
    os.rename(oldname,newname)   
    print(oldname,'======>',newname)
    
    n+=1
