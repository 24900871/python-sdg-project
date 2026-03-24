

# 🌍 SDG 13: Weather Prediction Project

## 📌 Overview
This project is a simple **Weather Prediction System** built using Python.  
It predicts future temperature trends using **Machine Learning (Linear Regression)**.

This project supports **United Nations Sustainable Development Goal 13 (Climate Action)** by analyzing and forecasting climate-related data.

---

## 🚀 Features
- 📊 Data analysis using Pandas  
- 🤖 Machine Learning model (Linear Regression)  
- 📈 Future temperature prediction  
- 📉 Data visualization using Matplotlib  

---

## 🛠️ Technologies Used
- Python 🐍  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## CODE :

~~~
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

~~~
## OUTPUT:
<img width="263" height="115" alt="image" src="https://github.com/user-attachments/assets/921c90b1-a679-4ee0-96c3-627e6bf478db" />
<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/f0d292d4-c9de-4a5c-8665-9aa2d102f52b" />

## Result :

The weather prediction model was successfully implemented using Linear Regression.
 

