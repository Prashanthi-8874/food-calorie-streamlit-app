import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

# internal dataset
data = {
    "Quantity": [100,150,200,250,300,120,180,220,260,310],
    "Sugar":    [10,15,20,25,30,12,18,22,27,32],
    "Fat":      [5,8,12,15,18,6,10,14,16,20],
    "Protein":  [2,3,4,5,6,2,3,4,5,6],
    "Calories": [150,200,300,350,400,180,260,320,370,420]
}

df = pd.DataFrame(data)

# features and target
X = df[["Quantity","Sugar","Fat","Protein"]]
y = df["Calories"]

# regression model
reg = LinearRegression()
reg.fit(X, y)

# sample input
sample = pd.DataFrame([[200,20,10,4]], columns=X.columns)

calories_pred = reg.predict(sample)[0]
print("Predicted Calories:", calories_pred)

# classification (healthy/unhealthy)
df["Health"] = df["Calories"].apply(lambda x: 1 if x < 300 else 0)

clf = LogisticRegression()
clf.fit(X, df["Health"])

health_pred = clf.predict(sample)[0]
print("Healthy(1)/Unhealthy(0):", health_pred)