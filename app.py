import streamlit as st
import pandas as pd
import joblib
import datetime

# --- 1. Load the Saved Artifacts ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('hotel_model.pkl.gz')
    columns = joblib.load('model_columns.pkl')
    encoders = joblib.load('label_encoders.pkl')
    return model, columns, encoders

model, model_columns, encoders = load_artifacts()

# --- 2. Build the Streamlit UI ---
st.title("StaySure")
st.subheader("You're AI helper for endless successful bookings")
st.write("Enter the booking details below to predict if the reservation will be canceled.")

# Create a dictionary to hold user inputs
input_data = {}

col1, col2 = st.columns(2)

with col1:
    st.subheader("Stay Details")
    input_data['number of weekend nights'] = st.number_input("Weekend Nights", min_value=0, value=0)
    input_data['number of week nights'] = st.number_input("Week Nights", min_value=0, value=1)
    input_data['average price'] = st.number_input("Average Price per Night", min_value=0.0, value=100.0)
    
    date_res = st.date_input("Date of Reservation", datetime.date.today())
    input_data['month'] = date_res.month
    input_data['year'] = date_res.year

with col2:
    st.subheader("Guest Details")
    input_data['number of adults'] = st.number_input("Adults", min_value=0, value=2)
    input_data['number of children'] = st.number_input("Children", min_value=0, value=0)

st.divider()
st.subheader("Additional Information")

# Dynamically create select boxes for every categorical column you trained on
# The selectbox will show the original text categories, which we will encode later
for col, le in encoders.items():
    # Provide the classes_ (original string values) to the user
    input_data[col] = st.selectbox(f"Select {col.title()}", le.classes_)

# --- 3. Process Inputs & Predict ---
if st.button("Predict Booking Status", type="primary"):
    
    # Calculate the engineered features exactly as done in training
    input_data['total_night'] = input_data['number of weekend nights'] + input_data['number of week nights']
    input_data['total_guests'] = input_data['number of adults'] + input_data['number of children']
    input_data['total_price'] = input_data['average price'] * input_data['total_night']
    
    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode the categorical variables using the saved encoders
    for col, le in encoders.items():
        input_df[col] = le.transform(input_df[col])
        
    # Ensure the columns are in the exact same order as the training data
    # Missing numeric columns will be filled with 0 just in case
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0 
            
    input_df = input_df[model_columns]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Display Result (Adjust "1" and "0" text based on how your target is encoded)
    st.divider()
    if prediction == 0:
        st.error("🚨 Your client will propaply **Cancel** the booking")
    else:
        st.success("✅ Your client will propaply **Not Cancel** the booking")
