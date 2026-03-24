import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "Day": [1, 2, 3, 4, 5, 6, 7],
    "Temperature": [30, 31, 29, 32, 34, 35, 36]
}

df = pd.DataFrame(data)

X = df[["Day"]]
y = df["Temperature"]

model = LinearRegression()
model.fit(X, y)

future_days = pd.DataFrame({"Day": [8, 9, 10]})
predictions = model.predict(future_days)

print("Predicted Temperatures:")
for d, t in zip(future_days["Day"], predictions):
    print(f"Day {d}: {t:.2f} °C")

plt.figure()
plt.scatter(df["Day"], df["Temperature"])
plt.plot(df["Day"], model.predict(X))
plt.scatter(future_days["Day"], predictions)
plt.xlabel("Day")
plt.ylabel("Temperature (°C)")
plt.title("Weather Prediction")
plt.grid()
plt.show()