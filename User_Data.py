import streamlit as st
import pickle
import numpy as np

# Load the trained model
model_filename = 'trained_model.sav'
loaded_model = pickle.load(open(model_filename, 'rb'))

# Streamlit app title
st.title("Logistic Regression Prediction App")

# Input fields for user to provide data
age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)
estimated_salary = st.number_input("Enter Estimated Salary", min_value=15000, max_value=150000, value=50000)

# Prediction button
if st.button("Predict"):
    # Prepare the input data
    input_data = np.array([[age, estimated_salary]])
    
    # Make prediction
    prediction = loaded_model.predict(input_data)
    
    # Display the result
    if prediction == 1:
        st.success("The model predicts a positive outcome.")
    else:
        st.error("The model predicts a negative outcome.")

# Run the Streamlit app using the command: `streamlit run your_script_name.py`
