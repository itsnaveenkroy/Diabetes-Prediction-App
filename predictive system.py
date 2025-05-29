# -*- coding: utf-8 -*-
import numpy as np
import pickle

# Loading the saved model
loaded_model = pickle.load(open('/Users/itsnaveenkroy/Documents/Codes/Diabetes Model/SVM_trained_model.sav', 'rb'))
STD_trained_model_data = pickle.load(open('/Users/itsnaveenkroy/Documents/Codes/Diabetes Model/STD_trained_model.sav', 'rb'))

input_data = (1,85,66,29,0,26.6,0.351,31)
# Changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data,dtype=np.float64)  #shape:(8,)

# reshape the array as we are prediction for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) #shape:(1,8)

# standardize the input data
std_data = STD_trained_model_data.transform(input_data_reshaped)
# print(std_data)

prediction = loaded_model.predict(std_data) #pickle model

print(prediction)

if prediction[0]==0:
  print ('The person is not diabetic !')
else:
  print ('The person is diabetic !')
  