import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Quantity": [100,150,200,250,300,120,180,220,260,310],
    "Calories": [150,200,300,350,400,180,260,320,370,420]
}

df = pd.DataFrame(data)

# simple classification (healthy/unhealthy)
df["Health"] = df["Calories"].apply(lambda x: 1 if x < 300 else 0)

counts = df["Health"].value_counts()

counts.plot(kind="bar")
plt.title("Healthy (1) vs Unhealthy (0)")
plt.xlabel("Class")
plt.ylabel("Count")

plt.show()