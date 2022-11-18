import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("# Predict Units Sold: 👉")
st.write("### Choose the amount of Dollars used, Advertisements, and Promotion amount:")

# Price of the product
price= st.slider('Dollar Used in Cost Price? 💲', min_value=1.0,max_value=10.0,value=2.0,step=1.0)

# Advertisement budget:
ads= st.slider('What is the Advertisement Budget? 💲', min_value=10.0,max_value=100.0,value=50.0,step=2.0)

# promotions:
promo= st.slider('what is the promotional budget? 💲', min_value=10.0,max_value=100.0,value=45.0,step=2.0)

rows=[price,ads,promo]
columns=['dollar_price', 'advertisment','promotions']

mktg_scenario=pd.DataFrame(dict(zip(columns,rows)),index=[0])
st.table(mktg_scenario)


# Add Button
if st.button(label="Click to Predict ----find Units Sold of Jewelry Items:"):

    
     #Load The model:

     loaded_model=pickle.load(open('Ridge_jewelry_sold.sav','rb'))

     # Make the predictions and get te prediction probabilities:
     pred=loaded_model.predict(mktg_scenario)[0]

     np.set_printoptions(precision=None, suppress=None)
 

     #st.write(f"Predicted Unit Sold of Jewery Items , 💲📊:{pred: .0f} units")
     st.write("{:9}".format(str(pred)))