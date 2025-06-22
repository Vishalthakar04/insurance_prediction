import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv("insurance (1).csv")

# Binary target based on charges median
median_charge = df['charges'].median()
df['high_charges'] = (df['charges'] > median_charge).astype(int)

# Encode categorical features
label_encoders = {}
for col in ['sex', 'smoker', 'region']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['high_charges']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Streamlit App
st.set_page_config(page_title="Insurance Charge Predictor", layout="wide")

# Sidebar navigation
page = st.sidebar.selectbox("Select a page", ["Prediction", "Visualization"])

if page == "Prediction":
    st.title("Insurance Charge Prediction")
    
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", label_encoders['sex'].classes_)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input("Children", min_value=0, max_value=5, value=0)
    smoker = st.selectbox("Smoker", label_encoders['smoker'].classes_)
    region = st.selectbox("Region", label_encoders['region'].classes_)

    if st.button("Predict"):
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [label_encoders['sex'].transform([sex])[0]],
            'bmi': [bmi],
            'children': [children],
            'smoker': [label_encoders['smoker'].transform([smoker])[0]],
            'region': [label_encoders['region'].transform([region])[0]],
        })
        prediction = model.predict(input_data)[0]
        result = "High Charges" if prediction == 1 else "Low Charges"
        st.subheader(f"Prediction: {result}")

elif page == "Visualization":
    st.title("Data Visualization")

    st.subheader("Distribution of Charges")
    fig, ax = plt.subplots()
    sns.histplot(df['charges'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Smoker vs Charges")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='smoker', y='charges', data=df, ax=ax2)
    ax2.set_xticklabels(label_encoders['smoker'].inverse_transform([0, 1]))
    st.pyplot(fig2)

    st.subheader("Region Distribution")
    fig3, ax3 = plt.subplots()
    region_names = label_encoders['region'].inverse_transform(sorted(df['region'].unique()))
    region_counts = df['region'].value_counts().sort_index()
    ax3.bar(region_names, region_counts)
    st.pyplot(fig3)

    st.subheader("Correlation Heatmap")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)
