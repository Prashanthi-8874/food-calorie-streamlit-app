import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

st.set_page_config(page_title="Food Calorie App", layout="centered")

st.title("🍎 Food Calorie Prediction App")

# ------------------ SAMPLE DATA ------------------
# Internal dataset (no external file needed)
data = pd.DataFrame({
    "Protein": [3, 1, 25, 8, 10],
    "Fat": [1, 0, 15, 4, 5],
    "Carbs": [30, 25, 0, 12, 20],
    "Calories": [130, 95, 300, 150, 200],
    "Healthy": [0, 1, 0, 1, 1]
})

X = data[["Protein", "Fat", "Carbs"]]
y_cal = data["Calories"]
y_health = data["Healthy"]

# ------------------ MODELS ------------------
cal_model = LinearRegression()
cal_model.fit(X, y_cal)

health_model = LogisticRegression()
health_model.fit(X, y_health)

# ------------------ USER INPUT ------------------
st.header("Enter Food Details")

protein = st.number_input("Protein", min_value=0.0, value=5.0)
fat = st.number_input("Fat", min_value=0.0, value=5.0)
carbs = st.number_input("Carbs", min_value=0.0, value=20.0)

if st.button("Predict"):
    input_data = np.array([[protein, fat, carbs]])

    # Prediction
    calories = cal_model.predict(input_data)[0]
    health = health_model.predict(input_data)[0]

    st.subheader("Results")
    st.write(f"🔥 Predicted Calories: {calories:.2f}")

    if health == 1:
        st.success("✅ Healthy Food")
    else:
        st.error("❌ Unhealthy Food")

    # ------------------ GRAPH ------------------
    st.subheader("📊 Visualization")

    fig, ax = plt.subplots()
    labels = ["Protein", "Fat", "Carbs"]
    values = [protein, fat, carbs]

    ax.bar(labels, values)
    ax.set_title("Nutritional Values")

    st.pyplot(fig)

# ------------------ EXTRA DATA VIEW ------------------
if st.checkbox("Show Dataset"):
    st.write(data)
