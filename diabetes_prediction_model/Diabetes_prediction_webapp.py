import numpy as np
import pickle
import streamlit



#load the save models

loaded_model=pickle.load(open("C:/Users/HP/OneDrive/Desktop/project/m_l projects/diabetes_prediction_model/expert_model.sav","rb"))

# creating a function for  prediction models
def diabetes_prediction(input_data):
    
    
    
    
    


# Changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)




    prediction=loaded_model.predict(input_data_reshaped )
    print(prediction)

    if (prediction[0]==0):
        return "The person is not Diabetic"
    else:
        return "The Person is Diabetic"


# Streamlit app
import streamlit as st

# Streamlit app
def main():
    # Giving the title of the app
    st.title("Diabetes Prediction Web App")
    
    # Getting the input data from users
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure (mm Hg)")
    SkinThickness = st.text_input("Skin Thickness (mm)")
    Insulin = st.text_input("Insulin Level (mu U/ml)")
    BMI = st.text_input("Body Mass Index (BMI)")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age (in years)")
    
    #code for printing
    diagnosis=""

    # Here, you can add a button to trigger the prediction when the user submits the data
    if st.button("Diabetes test results"):
        diagnosis=diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        # You would include the code to preprocess this input and make predictions here
    
        
        # You would then convert this input_data to the required format for prediction (e.g., converting to numpy array, standardizing, etc.)
        # Call your prediction function here, using the loaded model
        
        # Example prediction logic would go here
        # prediction = loaded_model.predict(std_data)
        # st.write(f"The predicted outcome is: {prediction[0]}")
    st.success(diagnosis)    
        
# Run the app
if __name__ == '__main__':
    main()





