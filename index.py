import streamlit as st
from model import prediction
from image import getImage
from main import run

st.title("Handwritten Math Expression Recognize")
st.write("""
##### A simple machine to recognize the number 0-9 and x,y character

***
""")


st.write(""" #### Introduction: 
This app could recognize a simple mathematical expression which 
consists of 0-9 and x,y character. When the input is a series of number by plus 
or minus, the machine could help u calculate the final result.If the input 
including variable x,y. The machine can do differential and integration for a
simple expression
""")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file!=None:
    genre = st.radio(
    "Select the operation: ",
    ('Result', 'Differential' ,'Definite Integral','Indefinite Integral','List Value','Summation'))
 
    if genre == 'Result':
        run(uploaded_file,0)

    elif genre=="Differential":
        run(uploaded_file,1)

    elif genre=="Definite Integral":
        run(uploaded_file,2)

    elif genre=='Indefinite Integral':
        run(uploaded_file,3)

    elif genre=='List Value':
        run(uploaded_file,4)

    elif genre=='Summation':
        run(uploaded_file,5) 
        


st.write(""" ***
made by NG ZI XING
""")