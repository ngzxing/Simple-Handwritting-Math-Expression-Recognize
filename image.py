import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from math import ceil,floor
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC


class getImage():
    
    def __init__(self):
        pass
    
    def readImage(self,filepath,th=85):
        #img=cv2.imread(filepath)
        
        file_bytes = np.asarray(bytearray(filepath.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        try:
            if img==None:
                raise TypeError("No specific file available")
        except ValueError:
            pass
        
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,self.binary=cv2.threshold(gray,th,255,0)
              
    def resize(self,size=26):
        imgList=self.imgList
        
        for i in range(len(imgList)):
            img=imgList[i]; x,y=img.shape;
            if(x>y):
                w=int(y/x*size); rest=np.abs((size-w)/2);k=size
                left=np.zeros((size,ceil(rest)))+255; right=np.zeros((size,floor(rest)))+255
                img=cv2.resize(img,(w,k))
                _,img=cv2.threshold(np.hstack((left,img,right)),1,255,0)
                imgList[i]=img
            elif(x<y):
                k=int(x/y*size); rest=np.abs((size-k)/2);w=size
                left=np.zeros((ceil(rest),size))+255; right=np.zeros((floor(rest),size))+255
                img=cv2.resize(img,(w,k))
                _,img=cv2.threshold(np.vstack((left,img,right)),1,255,0)
                imgList[i]=img
            else:
                w=size;rest=0;k=size
                img=cv2.resize(img,(k,w))
                _,img=cv2.threshold(img,1,255,0)
                imgList[i]=img
    
        self.imgList=imgList
        
       
    def detectGrp(self,area_thre=20,percent_thre=0.95,more=0):
        binary=self.binary
        contours,_=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        blank=np.zeros(binary.shape,np.uint8)
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            if (w*h)==(binary.shape[0]*binary.shape[1]):
                continue
            cv2.rectangle(blank,(x-more,y),(x+w+more,y+h),(255,255,255),thickness=5)
            
        
        contoursBlank,_=cv2.findContours(blank,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        border=[]
        for contour in contoursBlank:
            x,y,w,h=cv2.boundingRect(contour)
            if (w*h>area_thre) and (w>5) and (h>5):
                percent=np.sum(binary[y:y+h,x:x+w]/255)/(w*h)
                if (percent<percent_thre) and (percent>0.5):
                    cv2.rectangle(blank,(x,y),(x+w,y+h),(0,0,255))
                    border.append([[x,y],[x+w,y+h]])
        
        border=np.array(border)
        
        if border.shape[0]==0:
            raise Exception("No numbers or symbols detected. Pls make sure use white background, black pen and sharp image.")
        
        self.border=border[np.argsort(border[:,0,0])]
        self.binary=binary
    
    def grpImage(self):
        borders=self.border
        binary=self.binary
        imgList=[]

        for border in borders:
            imgList.append(binary[border[0,1]:border[1,1],border[0,0]:border[1,0]])
            
        self.imgList=imgList
        
        
    def rough(self,thickness=10):
        imgList=self.imgList
        
        for i in range(len(imgList)):
            contours,_=cv2.findContours(imgList[i].astype("uint8"),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            del contours[0]
            for c in contours:
                cv2.drawContours(imgList[i],c, -1, (0,0,0), thickness)
                
                
    def padding(self,size=26):
        pad_size=int((28-size)/2)

        for i in range(len(self.imgList)):
            img=self.imgList[i]
            self.imgList[i]=np.hstack((np.ones((28,pad_size))*255,
                                       np.vstack((np.ones((pad_size,size))*255,
                                       img,np.ones((pad_size,size))*255)),np.ones((28,pad_size))*255))
                
    
    def returnImg(self):
        return self.imgList