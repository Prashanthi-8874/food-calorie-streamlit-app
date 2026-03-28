import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

st.title("🍔 Food Calorie Predictor")

# internal dataset
data = {
    "Quantity": [100,150,200,250,300,120,180,220,260,310],
    "Sugar":    [10,15,20,25,30,12,18,22,27,32],
    "Fat":      [5,8,12,15,18,6,10,14,16,20],
    "Protein":  [2,3,4,5,6,2,3,4,5,6],
    "Calories": [150,200,300,350,400,180,260,320,370,420]
}

df = pd.DataFrame(data)

X = df[["Quantity","Sugar","Fat","Protein"]]
y = df["Calories"]

# models
reg = LinearRegression()
reg.fit(X, y)

df["Health"] = df["Calories"].apply(lambda x: 1 if x < 300 else 0)

clf = LogisticRegression()
clf.fit(X, df["Health"])

# user inputs
st.header("Enter Food Details")

quantity = st.number_input("Quantity", 0)
sugar = st.number_input("Sugar", 0)
fat = st.number_input("Fat", 0)
protein = st.number_input("Protein", 0)

if st.button("Predict"):
    sample = pd.DataFrame([[quantity, sugar, fat, protein]],
                          columns=X.columns)

    calories = reg.predict(sample)[0]
    health = clf.predict(sample)[0]

    st.subheader(f"🔥 Predicted Calories: {calories:.2f}")

    if health == 1:
        st.success("✅ Healthy Food")
    else:
        st.error("❌ Unhealthy Food")