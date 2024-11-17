#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Load dataset
heart_data = pd.read_csv('heart.csv')

# Check for missing values
print(heart_data.isnull().sum())

# Explore relationships
import seaborn as sns
sns.pairplot(heart_data, hue='target')  # Visualize relationships


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data
X = heart_data.drop('target', axis=1)
y = heart_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")


# In[4]:


import pickle

# Save the model
with open('heart_disease_model.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[5]:


import streamlit as st
import pandas as pd

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app title
st.title("Heart Disease Prediction")
st.write("Enter the patient details below to predict the likelihood of heart disease:")

# Input features
age = st.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.selectbox("Sex", options=["Male", "Female"])
cp = st.selectbox("Chest Pain Type (CP)", options=[
    "0: Typical Angina",
    "1: Atypical Angina",
    "2: Non-Anginal Pain",
    "3: Asymptomatic"
])
trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL (1 = True, 0 = False)", options=["0", "1"])
restecg = st.selectbox("Resting ECG Results", options=[
    "0: Normal",
    "1: ST-T wave abnormality",
    "2: Probable or definite LVH"
])
thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", options=["0", "1"])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=["0: Upsloping", "1: Flat", "2: Downsloping"])
ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
thal = st.selectbox("Thalassemia", options=[
    "0: Normal",
    "1: Fixed Defect",
    "2: Reversible Defect"
])

# Convert inputs into a DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [1 if sex == "Male" else 0],
    'cp': [int(cp.split(":")[0])],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [int(fbs)],
    'restecg': [int(restecg.split(":")[0])],
    'thalach': [thalach],
    'exang': [int(exang)],
    'oldpeak': [oldpeak],
    'slope': [int(slope.split(":")[0])],
    'ca': [ca],
    'thal': [int(thal.split(":")[0])]
})

# Make predictions
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# Display the result
if prediction == 1:
    st.success(f"The patient is likely to have heart disease. (Confidence: {probability:.2f})")
else:
    st.success(f"The patient is unlikely to have heart disease. (Confidence: {1 - probability:.2f})")


# In[ ]:




