import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

model = pk.load(open('model.pkl','rb'))
st.header('Develop By Umar Saadat')
st.header('CAR PRICE PREDICTING ML MODEL')

car_data = pd.read_csv('Cardetails.csv')

def get_Car_name(name):
    car_data = name.split(' ')[0]
    return car_data.strip()

car_data['name'] = car_data['name'].apply(get_Car_name)

name = st.selectbox('Select Car Brand',car_data['name'].unique())
year = st.slider('Car Manufacture Year',1997,2025)
km_driven = st.slider('No of kms Driven',11,200000)
fuel = st.selectbox('Fuel Type',car_data['fuel'].unique())
seller_type = st.selectbox('Seller Type',car_data['seller_type'].unique())
transmission = st.selectbox('Transmission Type',car_data['transmission'].unique())
owner = st.selectbox('Owner Type',car_data['owner'].unique())
mileage = st.slider('Car Mileage',10,40)
engine = st.slider('Engine CC',700,5000)
max_power = st.slider('Max Power',0,200)
seats = st.slider('No of Seats',6,10)

if st.button('Predict'):
    input_data = pd.DataFrame([[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats]],columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])
    input_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'],[1,2,3,4,5],inplace=True)
    input_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4],inplace=True)
    input_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3],inplace=True)
    input_data['transmission'].replace(['Manual', 'Automatic'],[1,2],inplace=True)
    input_data['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],inplace=True)
    
    
    price_predict = model.predict(input_data)
    
    st.markdown(f"<h1 style='text-align : center;'>Predicting Price is : {price_predict[0]:.2f}</h1>",unsafe_allow_html=True)