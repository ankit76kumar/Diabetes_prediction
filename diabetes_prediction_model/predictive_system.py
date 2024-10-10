import numpy as np
import pandas as pd
import pickle

loaded_model=pickle.load(open("C:/Users/HP/OneDrive/Desktop/project/m_l projects/diabetes_prediction_model/expert_model.sav","rb"))


input_data =(5,116,74,0,0,25.6,0.201,30)


# Changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)




prediction=loaded_model.predict(input_data_reshaped )
print(prediction)

if (prediction[0]==0):
    print("The person is not Diabetic")
else:
    print("The Person is Diabetic")