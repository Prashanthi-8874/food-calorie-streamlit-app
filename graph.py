import pandas as pd
import matplotlib.pyplot as plt

# internal dataset (no csv needed)
data = {
    "Quantity": [100,150,200,250,300,120,180,220,260,310],
    "Calories": [150,200,300,350,400,180,260,320,370,420]
}

df = pd.DataFrame(data)

plt.scatter(df["Quantity"], df["Calories"])
plt.xlabel("Quantity")
plt.ylabel("Calories")
plt.title("Quantity vs Calories")

plt.show()