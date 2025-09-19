import numpy as np
from sklearn.linear_model import LinearRegression as lr
import streamlit as st

hrs = np.array([0.5,1.0,1.5,2.0,2.5]).reshape(-1,1)
kcal = np.array([150,300,450,600,750])
model = lr()
model.fit(hrs,kcal)
pred = model.predict(hrs)
st.title("Calorie Predictor")
hours = st.number_input("Enter the number of hours you've exercised for (Ex: 0.5 or): ")
hours = [[hours]]
if st.button("Predict"):
    pred = model.predict(hours)
    st.write(f"Predicted calories burnt {pred[0]}")
