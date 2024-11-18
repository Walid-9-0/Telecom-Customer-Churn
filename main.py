import numpy as np
import pickle as pkl
import streamlit as st
import pandas as pd
from collections import Counter

# Load encoders
Internetservice_le = pkl.load(open('Encoding/InternetService_le.pkl', 'rb'))
Paymentmethod_le = pkl.load(open('Encoding/PaymentMethod_le.pkl', 'rb'))
Contract_Oe = pkl.load(open('Encoding/Contract_oe.pkl', 'rb'))

# Load scaler and models
scaler = pkl.load(open('Scaling/scaler.pkl', 'rb'))
models = {
    'Decision Tree': pkl.load(open('Models/Decision Tree.pkl', 'rb')),
    'Logistic Regression': pkl.load(open('Models/LogisticRegression.pkl', 'rb')),
    'SVC': pkl.load(open('Models/SVC.pkl', 'rb')),
    'KNN': pkl.load(open('Models/KNN.pkl', 'rb')),
    'Random Forest': pkl.load(open('Models/RandomForest.pkl', 'rb')),
    'Gradient Boosting': pkl.load(open('Models/GradientBoosting.pkl', 'rb')),
    'XGBoost': pkl.load(open('Models/XGBoost.pkl', 'rb')),
    'AdaBoost': pkl.load(open('Models/AdaBoost.pkl', 'rb')),
    'Stacking': pkl.load(open('Models/Stacking.pkl', 'rb')),
}

# Function to predict churn
def predict_churn(input_data):
    predictions = {}
    for name, model in models.items():
        pred = model.predict(input_data)[0]
        predictions[name] = "Churn" if pred == 1 else "Not Churn"
    return predictions

# Function to display predictions with mode highlighted
def display_predictions(predictions):
    st.markdown("<h2 style='text-align: center; color: #4A90E2;'>⚜️Prediction Results⚜️</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)  # Three equal-width columns
    
    # Split predictions into three parts
    predictions_items = list(predictions.items())
    num_items = len(predictions_items)
    first_third = predictions_items[:num_items // 3]
    second_third = predictions_items[num_items // 3:2 * num_items // 3]
    last_third = predictions_items[2 * num_items // 3:]

    # Display first third in the first column
    with col1:
        for model, result in first_third:
            st.markdown(
                f"<p style='font-size:15px; text-align: center;'>"
                f"<b>{model}:</b> <span style='color: #FFA500;'>{result}</span>"
                f"</p>", unsafe_allow_html=True
            )

    # Display second third in the second column
    with col2:
        for model, result in second_third:
            st.markdown(
                f"<p style='font-size:15px; text-align: center;'>"
                f"<b>{model}:</b> <span style='color: #FFA500;'>{result}</span>"
                f"</p>", unsafe_allow_html=True
            )

    # Display last third in the third column
    with col3:
        for model, result in last_third:
            st.markdown(
                f"<p style='font-size:15px; text-align: center;'>"
                f"<b>{model}:</b> <span style='color: #FFA500;'>{result}</span>"
                f"</p>", unsafe_allow_html=True
            )

    # Calculate and display mode in larger font
    mode_result = Counter(predictions.values()).most_common(1)[0][0]
    if mode_result == "Not Churn":
        st.markdown(f"""<h1 style='text-align: center;font-size:55px; color: white;'>The Customer Status: <span style='color: #27AE60;'>{mode_result}✅</span></h1>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<h1 style='text-align: center;font-size:55px; color: white;'>The Customer Status: <span style='color: #E74C3C;'>{mode_result}😡</span></h1>""", unsafe_allow_html=True)


# Function to save user input in a dictionary
def save_user_data(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
                   MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
                   DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
                   Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, prediction):
    
    # Create a dictionary with all inputs and the prediction result
    user_data = {
        'Gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'Tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'Prediction': prediction
    }

    # Save the data in session_state for persistence
    if 'user_data_list' not in st.session_state:
        st.session_state['user_data_list'] = []
    
    st.session_state['user_data_list'].append(user_data)

# Initialize session state for predictions
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None

# Streamlit UI
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")
st.title(" 📞Telecom Customer Churn Prediction")
st.markdown('Predict whether a customer will churn based on various attributes.')

# Photo
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.pexels.com/photos/19683578/pexels-photo-19683578/free-photo-of-analog-telephone-in-sunlight.jpeg?auto=compress&cs=tinysrgb&w=600");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Input Form
st.markdown("---")
st.subheader("📝 Enter Customer Attributes:")

c1, c2, c3 = st.columns(3)
with c1:
    gender = st.selectbox("👥 Gender", options=['Male', 'Female'])
    SeniorCitizen = st.selectbox("🎖️ Senior Citizen", options=['Yes', 'No'])
    Partner = st.selectbox("💍 Partner", options=['Yes', 'No'])
    Dependents = st.selectbox("👶 Dependents", options=['Yes', 'No'])
    tenure = st.number_input("📅 Tenure (months)", min_value=0, value=0)
    PhoneService = st.selectbox("📞 Phone Service", options=['Yes', 'No'])

with c2:
    MultipleLines = st.selectbox("📶 Multiple Lines", options=['Yes', 'No'])
    InternetService = st.selectbox("🌐 Internet Service", options=['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.selectbox("🔐 Online Security", options=['Yes', 'No'])
    OnlineBackup = st.selectbox("💾 Online Backup", options=['Yes', 'No'])
    DeviceProtection = st.selectbox("🛡️ Device Protection", options=['Yes', 'No'])
    TechSupport = st.selectbox("🛠️ Tech Support", options=['Yes', 'No'])
    StreamingTV = st.selectbox("📺 Streaming TV", options=['Yes', 'No'])

with c3:
    StreamingMovies = st.selectbox("🎥 Streaming Movies", options=['Yes', 'No'])
    Contract = st.selectbox("📃 Contract", options=['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.selectbox("📨 Paperless Billing", options=['Yes', 'No'])
    PaymentMethod = st.selectbox("💳 Payment Method", Paymentmethod_le.classes_)
    MonthlyCharges = st.number_input("💵 Monthly Charges", min_value=0.0, value=0.0)
    TotalCharges = st.number_input("💰 Total Charges", min_value=0.0, value=0.0)

# Encoding Input
gender_encod = 1 if gender == 'Male' else 0
SeniorCitizen_encod = 1 if SeniorCitizen == 'Yes' else 0
Partner_encod = 1 if Partner == 'Yes' else 0
Dependents_encod = 1 if Dependents == 'Yes' else 0
PhoneService_encod = 1 if PhoneService == 'Yes' else 0
MultipleLines_encod = 1 if MultipleLines == 'Yes' else 0
InternetService_encod = Internetservice_le.transform([InternetService])[0]
OnlineSecurity_encod = 1 if OnlineSecurity == 'Yes' else 0
OnlineBackup_encod = 1 if OnlineBackup == 'Yes' else 0
DeviceProtection_encod = 1 if DeviceProtection == 'Yes' else 0
TechSupport_encod = 1 if TechSupport == 'Yes' else 0
StreamingTV_encod = 1 if StreamingTV == 'Yes' else 0
StreamingMovies_encod = 1 if StreamingMovies == 'Yes' else 0
Contract_encod = Contract_Oe.transform([[Contract]])[0][0]
PaperlessBilling_encod = 1 if PaperlessBilling == 'Yes' else 0
PaymentMethod_encod = Paymentmethod_le.transform([PaymentMethod])[0]

# Create input data array
input_data1 = np.array([[ SeniorCitizen_encod, Partner_encod, Dependents_encod
                        , PhoneService_encod, MultipleLines_encod, InternetService_encod,
                        OnlineSecurity_encod, OnlineBackup_encod, DeviceProtection_encod,
                        TechSupport_encod, StreamingTV_encod, StreamingMovies_encod,
                        Contract_encod, PaperlessBilling_encod, PaymentMethod_encod
                        ]])
input_data2 = np.array([[MonthlyCharges, TotalCharges, tenure]])
# Scale input data
input_data3 = scaler.transform(input_data2)
input_data = np.concatenate((input_data1, input_data3), axis=1)




# Button to trigger prediction and save data
if st.sidebar.button("🔍 Predict Churn",use_container_width=True):
    predictions = predict_churn(input_data)
    mode_result = Counter(predictions.values()).most_common(1)[0][0]

    # Save the user's inputs along with the prediction result
    save_user_data(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
                   MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
                   DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
                   Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, mode_result)

    # Display the predictions
    display_predictions(predictions)

# Optional: Display saved data
if st.sidebar.button("Show All User Data",use_container_width=True):
    if 'user_data_list' in st.session_state and st.session_state['user_data_list']:
        st.write(pd.DataFrame(st.session_state['user_data_list']))
    else:
        st.write("No user data saved yet.")
