import streamlit as st
import pickle
import numpy as np

def load_model():
   with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_Province = data["le_Province"]
le_education = data["le_education"]

def show_predict_page():
    st.title("IT Engineer Salary Prediction In Indonesia")
    st.write("""### we need some information to predict the salary""")
    
    countries = (
        "DKI Jakarta",
        "Yogyakarta",
        "Surabaya",
        "Sidoarjo",
        "Bekasi",
        "Denpasar",
        "Malang",
        "Tasikmalaya",
        "Kediri",
        "Tanggerang",
        "Balikpapan",
        "Depok",
        "Makassar",
        "Kendari",
        "Pasuruan",
        "Ternate",
        "Sukabumi",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    Province = st.selectbox("Province", countries)
    education = st.selectbox("Education Level", education)

    expericence = st.slider("Years of Experience", 0, 30, 5)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[Province, education, expericence ]])
        X[:, 0] = le_Province.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is IDR {salary[0]:.2f}")