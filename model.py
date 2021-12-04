import numpy as np
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
    
    def power_checker(self,borders,target):
    
        for i in range(borders.shape[0]-1):
            border=borders[i]
            w=border[1,1]-border[0,1]
            
            if (borders[i+1][0,1]<border[0,1]) and (((border[1,1]-borders[i+1][1,1])/w)>0.6):
                target[i+1]=int("9"+str(target[i+1]))        
        return target
                
    def changeSymbol(self,char):
        if char==10:
            return "+"
        elif char==11:
            return "-"
        elif char>=90:
            return "**"+"".join(list(set(str(char))-{"9"}))
        elif char==16:
            return "*"
        elif char==12:
            return "x"
        elif char==13:
            return "y"
        else:
            return str(char)
    
    def grammar(self,predict,prob,borders):
        target=predict[:,13]
        
        target=self.checkFB(target,predict,prob)
        target=self.power_checker(borders,target)
        
        formula=""
        for t in target:
            formula+=(self.changeSymbol(t)) 
            
        formula_list=list(formula)
        num_list=list("0123456789")
        k=0
        for i in range(1,len(formula_list)):
            if (formula_list[i] in ["x","y"]) and (formula_list[i-1] in num_list):
                formula_list.insert(k+i,"*")
                
        formula="".join(formula_list)
        
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
    
    def list_value(self,formula,var,a,b,differ=1):
        if ("x" in set(formula)) and ("y" in set(formula)) or (("x" in set(formula))==0) and (("y" in set(formula))==0):
            return "0","0"
        
        f=lambdify(var,formula,"numpy")
        domain=np.arange(a,b+differ,differ)
        
        return domain,f(domain)
    
    def summation(self,formula,var,a,b,differ=1):
        _,value=self.list_value(formula,var,a,b,differ=differ)
        
        if value=="0":
            return "0"

        return np.sum(value)