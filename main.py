import numpy as np
import pickle
import streamlit as st

# Loading the saved model :
loaded_model = pickle.load(open('Resources/trained_model.sav', 'rb'))


def diabetes_prediction(input_data):
    # Changing the data to a numpy array :
    new_input = np.asarray(input_data)

    # Standardize the input data :# Reshape the array as we are predicting for one instance :
    reshaped_input = new_input.reshape(1, -1)
    prediction = loaded_model.predict(reshaped_input)

    if prediction[0] == 0:
        return 'The person is non-diabetic'
    else:
        return 'The person is diabetic'


def main():
    st.title('Diabetes Prediction Webapp')

    # Input data from from user:
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the person')

    # Prediction :
    diagnosis = ''

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction(
            [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)


if __name__ == '__main__':
    main()
