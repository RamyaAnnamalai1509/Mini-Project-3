import pickle
import pandas as pd
import streamlit as st
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

loaded_model1 = pickle.load(open(r'C:\framework\env\Scripts\random_forest_classifier_kidney.pkl', 'rb'))
loaded_model2 = pickle.load(open(r'C:\framework\env\Scripts\random_forest_classifier_liver.pkl', 'rb'))
loaded_model3 = pickle.load(open(r'C:\framework\env\Scripts\random_forest_classifier_parkinson.pkl','rb'))
st.set_page_config(page_title="Disease Predictions", page_icon=":stethoscope:")
st.title("Disease Predictions :🚑:")
page = st.sidebar.selectbox("Select a Prediction", ("Kidney","Liver","Parkinsons"))
if page == "Kidney":
    st.header("Enter Patient Data:")
    age = st.number_input("Age", min_value=0, max_value=150, value=30)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=300, value=80)
    al = st.number_input("Albumin", min_value=0.0, value=4.0)
    su = st.number_input("Sugar", min_value=0, value=0)
    rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
    ba = st.selectbox("Bacteria", ["notpresent", "present"])
    bgr = st.number_input("Blood Glucose Random", min_value=0, value=100)
    bu = st.number_input("Blood Urea", min_value=0, value=30)
    sc = st.number_input("Serum Creatinine", min_value=0.0, value=1.0)
    pot = st.number_input("POT", min_value=0.0, value=2.5)  
    wc = st.number_input("Waist", min_value=2000, value=2500)
    htn = st.selectbox("Hypertension", ["yes", "no"])
    dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
    pe = st.selectbox("Pedal Edema", ["yes", "no"])
    ane = st.selectbox("Anemia", ["yes", "no"])


# Create a dictionary from user inputs
    input_data = {
        'age': age,
        'bp': bp,
        'al': al,
        'su': su,
        'rbc': rbc,
        'ba': ba,
        'bgr': bgr,
        'bu': bu,
        'sc': sc,
        'pot': pot,        
        'wc': wc,       
        'htn': htn,
        'dm': dm,        
        'pe': pe,
        'ane': ane,
        }

# Function to preprocess input data (similar to your preprocessing steps)

    def preprocess_input(input_data):
        df_input = pd.DataFrame([input_data])

        # Convert categorical features to numerical
        le = preprocessing.LabelEncoder()
        for col in ['rbc', 'ba', 'htn', 'dm', 'pe', 'ane']:
            df_input[col] = le.fit_transform(df_input[col])

        return df_input

    # Preprocess the input data
    input_df = preprocess_input(input_data)


# Make predictions
    if st.button("Predict"):
        prediction = loaded_model1.predict(input_df)
        probability = loaded_model1.predict_proba(input_df)[:,1]

        if prediction[0] == 0:
            st.write("Prediction: No Kidney Disease :😎💪: ")
        else:
            st.write("Prediction: Kidney Disease :🚨😪:")
        st.write(f"Probability: {probability[0]:.2f}")
elif page == "Liver":
    st.header("Patient Information")
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"]) 
    Total_Bilirubin = st.number_input("Total Bilirubin", min_value=0.0, value=2.0)
    Direct_Bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, value=0.0)
    Alkaline_Phosphotase = st.number_input("Alkaline Phosphotase", min_value=0, value=100)
    Alamine_Aminotransferase = st.number_input("Alamine Aminotransferase", min_value=0, value=10)
    Aspartate_Aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=0, value=10)
    Total_Protiens = st.number_input("Total Proteins", min_value=0.0, value=5.0)
    Albumin = st.number_input("Albumin", min_value=0.0, value=1.0)    
    Albumin_and_Globulin_Ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.00, value=0.20)
    gender_encoded = 1 if gender == "Male" else 0


# Create a dataframe from user input
    user_input = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_encoded],
        'Total_Bilirubin' : [Total_Bilirubin],
        'Direct_Bilirubin' : [Direct_Bilirubin],
        'Alkaline_Phosphotase' : [Alkaline_Phosphotase],
        'Alamine_Aminotransferase' : [Alamine_Aminotransferase],
        'Aspartate_Aminotransferase' : [Aspartate_Aminotransferase],
        'Total_Protiens': [Total_Protiens],
        'Albumin': [Albumin],
        'Albumin_and_Globulin_Ratio': [Albumin_and_Globulin_Ratio]
    })
    if st.button("Predict"):
        prediction = loaded_model2.predict(user_input)
        probability = loaded_model2.predict_proba(user_input)[:,1]

        if prediction[0] == 1:
            st.write("Prediction: No Liver Disease :☺️: ")
        else:
            st.write("Prediction: Liver Disease :😪:")
        st.write(f"Probability: {probability[0]:.2f}")

elif page == "Parkinsons":
    st.title("Parkinson's Disease Prediction")
    feature_names = loaded_model3.feature_names_in_  # Assuming your model has this attribute
    input_data = {}

    for feature in feature_names:
        input_data[feature] = st.number_input(f"Enter {feature}:", value=0.0, format="%.6f", step=0.000001)

# Create a prediction button
    if st.button("Predict"):
    # Create a DataFrame from the user input
        input_df = pd.DataFrame([input_data])
    
        try:
        # Make a prediction
            prediction = loaded_model3.predict(input_df)[0]      

            if prediction == 0:  # Assuming 0 represents healthy
                st.success("Prediction: Healthy 😊") 
                st.balloons()
            else:
                st.error("Prediction: Parkinson's Disease 😔")
                st.snow()
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")