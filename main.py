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

from math import ceil,floor
from sklearn.tree import DecisionTreeClassifier as DT
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Activation
from tensorflow.keras.utils import to_categorical,normalize
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import models
from sympy import *



def run(filepath,selection):
    pr=prediction()
    Img=getImage()
    Img=getImage()
    Img.readImage(filepath,th=85)
    Img.detectGrp(percent_thre=1)
    Img.grpImage()
    Img.rough(thickness=10)
    Img.resize(size=20)
    Img.padding(size=20)
    imgList=Img.returnImg()

    trydata=[]
    for img in imgList:
        rel,binary=cv2.threshold(img,1,1,0)
        trydata.append(binary)
    trydata=np.array(trydata)

    predict,proba=pr.predict(trydata)
    formula=pr.grammar(predict,proba,Img.border)

    if (12 in predict[:,13]) or (13 in predict[:,13]):
        pass
 
        
    if selection==0:
        if (12 in predict[:,13]) or (13 in predict[:,13]):
            st.write(sympify(formula,evaluate=False))
        
        else:
            st.write(sympify(formula,evaluate=False))
            st.write(sympify(formula))

    else: 
        st.write("Enter your variable")

        if selection==1:
            var=st.text_input("")
            select1 = st.radio(
            "",
            ('Input parameter',"Calculate Differential"))
            
            if select1=="Calculate Differential":
                formulaD=pr.deriviative(formula,var)
                st.write(formulaD)

        elif selection==2:
            var=st.text_input("")
            a=st.text_input("upper limit")
            b=st.text_input("lower limit")

            select1 = st.radio(
            "",
            ('Input parameter',"Calculate Integral result"))
            
            if select1=="Calculate Integral result":
                formulaDInt=pr.integral(formula,var,True,a=b,b=a)
                st.write(formulaDInt)


        elif selection==3:
            var=st.text_input("")

            select3 = st.radio(
            "",
            ('Input parameter',"Calculate Integral"))
            
            if select3=="Calculate Integral":
                formulaInt=pr.integral(formula,var,False)
                st.write(formulaInt)

        elif selection==4:
            var=st.text_input("")
            a=st.text_input("From")
            b=st.text_input("to")
            c=st.text_input("Increment")

            select4 = st.radio(
            "",
            ('Input parameter',"Calculate List"))
            
            if select4=="Calculate List":
                domain,listVar=pr.list_value(formula,var,int(a),int(b),differ=int(c))

                if listVar=="0":
                    st.write("Sry for not supported")
                else:
                    st.write(pd.DataFrame(listVar,columns=["value"],index=list(domain)))


        elif selection==5:
            var=st.text_input("")
            a=st.text_input("From")
            b=st.text_input("to")
            c=st.text_input("Increment")

            select4 = st.radio(
            "",
            ('Input parameter',"Calculate Sum"))
            
            if select4=="Calculate Sum":
                sumValue=pr.summation(formula,var,int(a),int(b),differ=int(c))

                if sumValue=="0":
                    st.write("Sry for not supported")
                else:
                    st.write(sumValue)

        

            

        
    
    


