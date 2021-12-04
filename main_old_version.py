#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from model import prediction
from image import getImage
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from math import ceil,floor
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Activation
from tensorflow.keras.utils import to_categorical,normalize
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import models
from sympy import *



class prediction():
    
    def __init__(self):
        self.model=models.load_model("CNNxy2")
            
    
    def predict(self,X):
        X=normalize(X)
        prob=self.model.predict(X)
        return np.argsort(prob,axis=1),prob
    
    def checkFB(self,target,predict,prob):
        length=target.shape[0]
        last=length-1
        
        if target[0]==10:
            target[0]=predict[0][predict!=10][-1]
            
        if target[length-1] in [10,11]:
            target[last]=predict[last][(predict[last]!=11)*(predict[last]!=10)][-1]
                
        return target
    
    def checkConjPlus(self,target,predict,prob):
        length=target.shape[0]
        sign_index=np.arange(length)[target==10]
        
        for i in range(sign_index.shape[0]-1):
            k=sign_index[i]; kp1=sign_index[i+1]
            if (k+1)==kp1:
                yhat=predict[k]; yhatp1=predict[kp1]
                indexi=np.arange(14)[yhat==(yhat[yhat!=10][-1])]
                indexip1=np.arange(14)[yhatp1==(yhatp1[yhatp1!=10][-1])]
                
                if prob[k,indexi][0]>prob[kp1,indexip1][0]:
                    target[k]=yhat[indexi]
                else:
                    target[kp1]=yhatp1[indexip1]
                    
        return target
    
    def checkConjMinus(self,target,predict,prob):
        length=target.shape[0]
        sign_index=np.arange(length)[(target==10)+(target==11)]
    
        
        for i in range(sign_index.shape[0]-1):
            if (sign_index[i]+1)==sign_index[i+1]:
                k=sign_index[i]; kp1=sign_index[i+1]
                yhat=predict[k]; yhatp1=predict[kp1]
                indexi=np.arange(14)[yhat==(yhat[yhat!=10][-1])]
                indexip1=np.arange(14)[yhatp1==(yhatp1[yhatp1!=10][-1])]
                if target[k]==10:
                    target[kp1]=14
                        
                elif (target[k]==14) and (target[kp1]==11):
                    target[kp1]=14
                    
                elif (target[k]==target[kp1]):

                    target[kp1]=14
                    
                else:
                    if prob[k,indexi]>prob[kp1,indexip1]:
                        target[k]=yhat[indexi]
                    else:
                        target[kp1]=yhatp1[indexip1]
                        
        if target[0]==11:
            target[0]=14
                        
        return target
                    
    def negNum(self,numList):
        length=len(numList)
        sign_index=np.arange(length)[np.array(numList)==14]
        
        conjNeg=[]
        conjNegList=[]
        print(sign_index)
        for i in range(sign_index.shape[0]-1):
            if (sign_index[i]+1)==sign_index[i+1]:
                conjNeg.append(sign_index[i])
                if (i+2)==sign_index.shape[0]:
                    conjNeg.append(sign_index[i+1])
                    conjNegList.append(conjNeg)
            else:
                conjNeg.append(sign_index[i])
                if len(conjNeg)==1:
                    conjNeg.append(sign_index[i])
                conjNegList.append(conjNeg)
                conjNeg=[]
        
        print(conjNegList)
        
        for neg in conjNegList:
            lengthNeg=neg[1]-neg[0]
        
            if (lengthNeg)%2==0:
                numList[neg[1]+1]*=-1
                for i in range(neg[0],neg[1]+1):
                    numList[i]="a"
                
            else:
                for i in range(neg[0],neg[1]+1):
                    numList[i]="a"
        numList=np.array(numList)
        
        return list(numList[numList!="a"].astype(int))
    
    def power_checker(self,borders,target):
    
        for i in range(borders.shape[0]-1):
            border=borders[i]
            w=border[1,1]-border[0,1]
            
            if (borders[i+1][0,1]<border[0,1]) and (((border[1,1]-borders[i+1][1,1])/w)>0.6):
                target[i+1]=int("9"+str(target[i+1]))
            
            
        return target
                
    
    def grammar(self,predict,prob,borders):
        target=predict[:,13]
        
        target=self.checkFB(target,predict,prob)
        target=self.checkConjPlus(target,predict,prob)
        target=self.checkConjMinus(target,predict,prob)
        target=self.power_checker(borders,target)
        
        symbolList=[]
        numberList=[]
        number=""
        for t in target:
            if t in [10,11]:
                symbolList.append(t)
                numberList.append(int(number))
                number=""
                
            elif t==14:
                numberList.append(t)
                
            elif t in [0,1,2,3,4,5,6,7,8,9,12,13]:
                if t in [12,13]:
                    if number!="":
                        symbolList.append(16)
                        numberList.append(int(number))
                        number=""
                        number+=str(t)
                    else:
                        number+=str(t)
                else:
                    number+=str(t)
                    
            else:
                symbolList.append(15)
                numberList.append(int(number))
                number=""
                number+="".join(list(set(str(t))-{"9"}))
               
                
            
        numberList.append(int(number))
        number=""
        
        print(symbolList)
        print(numberList)
    
        numberList=self.negNum(numberList)
        print(numberList)
        
        return symbolList,numberList

    def operation(self,symbolList,numberList):
        
        if 15 in symbolList:
            newSymbolList=[]; newNumberList=[]
            for i in range(len(symbolList)):
                if symbolList[i]==15:
                    newNumberList.append(numberList[i]**numberList[i+1])
                else:
                    if (i+2)==len(numberList):
                        if symbolList[i-1]!=15:
                            newNumberList.append(numberList[i])
                        
                        newNumberList.append(numberList[i+1])
                        newSymbolList.append(symbolList[i])
                    
                    elif i==0:
                        newNumberList.append(numberList[i])
                        newSymbolList.append(symbolList[i])
                        
                    elif symbolList[i-1]==15:
                        newSymbolList.append(symbolList[i])
                        
                    else:
                        newNumberList.append(numberList[i])
                        newSymbolList.append(symbolList[i])
                    
        
            numberList=newNumberList
            symbolList=newSymbolList
        
        result=numberList[0]
        for i in range(len(symbolList)):
            if symbolList[i]==11:    
                result-=numberList[i+1]
            
            else:
                result+=numberList[i+1]
        
        return result
    
    def checkXY(self,npi):
        if npi==12:
            return "x"
        elif npi==13:
            return "y"
        else:
            return str(npi)
        
    
    def operationVar(self,symbolList,numberList):
        
        if (15 in symbolList):
            newSymbolList=[]; newNumberList=[]
            for i in range(len(symbolList)):
                ni=numberList[i]; nip1=numberList[i+1]
                
                if symbolList[i]==15:
                    newNumberList.append(self.checkXY(ni)+"**"+self.checkXY(nip1))
                
                else:
                    if (i+2)==len(numberList):
                        if symbolList[i-1]!=15:
                            newNumberList.append(self.checkXY(ni))
                        
                        newNumberList.append(self.checkXY(nip1))
                        newSymbolList.append(symbolList[i])
                    
                    elif i==0:
                        newNumberList.append(self.checkXY(ni))
                        newSymbolList.append(symbolList[i])
                        
                    elif symbolList[i-1]==15:
                        newSymbolList.append(symbolList[i])
                        
                    else:
                        newNumberList.append(self.checkXY(ni))
                        newSymbolList.append(symbolList[i])
                    
                   
        numberList=newNumberList
        symbolList=newSymbolList
        
        if (16 in symbolList):
            newSymbolList=[]; newNumberList=[]
            for i in range(len(symbolList)):
                ni=numberList[i]; nip1=numberList[i+1]
                
                if symbolList[i]==16:
                    newNumberList.append(self.checkXY(ni)+"*"+self.checkXY(nip1))
                
                else:
                    if (i+2)==len(numberList):
                        if symbolList[i-1]!=16:
                            newNumberList.append(self.checkXY(ni))
                        
                        newNumberList.append(self.checkXY(nip1))
                        newSymbolList.append(symbolList[i])
                    
                    elif i==0:
                        newNumberList.append(self.checkXY(ni))
                        newSymbolList.append(symbolList[i])
                        
                    elif symbolList[i-1]==16:
                        newSymbolList.append(symbolList[i])
                        
                    else:
                        newNumberList.append(self.checkXY(ni))
                        newSymbolList.append(symbolList[i])
                    
        numberList=newNumberList
        symbolList=newSymbolList  
        
        formula=self.checkXY(numberList[0])
        for i in range(len(symbolList)):
            si=symbolList[i]; npi=numberList[i+1]
            if si==11:
                formula+="-"+self.checkXY(npi)
            
            elif si==10:
                formula+="+"+self.checkXY(npi)
                
            
        
        return formula
    
    def integral(self,formula,var,definite,a=0,b=0):
        
        if definite:
            result=integrate(formula,(var,a,b))
        else:
            result=integrate(formula,Symbol(var))
            
        return result
    
    def deriviative(self,formula,var):
        result=diff(formula,var)
        
        return result