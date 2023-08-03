import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import BaggingRegressor
import warnings
warnings.filterwarnings('ignore')


model=pickle.load(open('bagging.pkl','rb'))

def car_price_pred():
    st.title('Car Price Prediction Using ML')
    img='car.png'
    st.image(img)
    st.subheader('Car Price Predictor')
    st.info('''We need some information to predict Car price''')
car_price_pred()

df=pd.read_csv('CAR DETAILS.csv')
cars=(df['name'].unique())
transmission=(df['transmission'].unique())
seller=(df['seller_type'].unique())
owner=(df['owner'].unique())
fuel=(df['fuel'].unique())

p1=st.selectbox('Select the Car',cars)
if p1 in df['name'].unique():
    p1 = df[df['name'] == p1].index[0]

p2=st.slider('Model Year',2005,2020,2005)
if p2 in df['year'].unique():
    p2 = df[df['year'] == p2].index[0]

p3=st.selectbox('Seller Type',seller)
if p3 in df['seller_type'].unique():
    p3 = df[df['seller_type'] == p3].index[0]

p4=st.selectbox('Owner Type',owner)
if p4 in df['owner'].unique():
    p4 = df[df['owner'] == p4].index[0]

p5=st.selectbox('Transmission Type',transmission)
if p5 in df['transmission'].unique():
    p5 = df[df['transmission'] == p5].index[0]

p6=st.selectbox('Fuel Type',fuel)
if p6 in df['fuel'].unique():
    p6 = df[df['fuel'] == p6].index[0]

p7=(st.slider('KM Driven',500,10000000,500))/100000

x=pd.DataFrame({'name':[p1],'year':[p2],'fuel':[p6]
                ,'seller_type':[p3],'transmission':[p4],'owner':[p5],'km_driven_in_lacks':[p7]})
ok=st.button('Predict Car Price')
if ok:
    prediction=model.predict(x)
    st.success('Predicted Car Price:'+str( prediction*100000) +'Rupees')
    st.caption('Thanks for using!')
    st.balloons()
