import numpy as np
import pickle
import streamlit as st


# Loading the saved model
loaded_model = pickle.load(open('SVM_trained_model.sav', 'rb'))
STD_trained_model_data = pickle.load(open('STD_trained_model.sav', 'rb'))

# Creating a function for prediction
def diabetes_prediction(input_data):
    
    # Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data,dtype=np.float64)  #shape:(8,)

    # reshape the array as we are prediction for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) #shape:(1,8)

    # standardize the input data
    std_data = STD_trained_model_data.transform(input_data_reshaped)


    prediction = loaded_model.predict(std_data) #pickle model

    print(prediction)

    if prediction[0]==0:
        return 'The person is not diabetic !'
    else:
        return 'The person is diabetic !'
  
    
def main():
    
    # Giving a title 
    st.title('Diabetes Prediction Web App')
    
    
    # Getting the input data from the user
    
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure value")
    SkinThickness = st.text_input("Skin Thickness value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age of the Person")
    
    
    # Code for Prediction
    diagnosis = ''
    
    # Creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        try:
            input_values = [
            float(Pregnancies), float(Glucose), float(BloodPressure),
            float(SkinThickness), float(Insulin), float(BMI),
            float(DiabetesPedigreeFunction), float(Age)
        ]
            diagnosis = diabetes_prediction(input_values)
        except ValueError:
            diagnosis = "⚠️ Please enter valid numeric values in all fields."

    st.success(diagnosis)


    
    
if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
