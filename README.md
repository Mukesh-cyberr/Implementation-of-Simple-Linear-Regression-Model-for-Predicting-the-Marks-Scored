# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Get the Independent Variable X and Dependent Variable Y - Load the dataset and extract X (study hours) and Y (marks scored).

Step 2: Calculate the Mean of the X-values and the Mean of the Y-values - Compute X̄ = (Σxi) / n and Ȳ = (Σyi) / n.

Step 3: Find the Slope m of the Line of Best Fit Using the Formula - Calculate m = Σ[(xi - X̄)(yi - Ȳ)] / Σ[(xi - X̄)²].

Step 4: Compute the Y-intercept of the Line Using the Formula - Calculate b = Ȳ - m·X̄.

Step 5: Use the Slope m and the Y-intercept to Form the Equation of the Line - Form the equation Y = mX + b.

Step 6: Obtain the Straight Line Equation Y = mX + b and Plot the Scatterplot - Plot the scatter plot of original data points and overlay the regression line Y = mX + b.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Mukesh Raj D
RegisterNumber:  212224100038
*/


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)

print("Dataset:\n", df.head())
df

X = df[["Hours_Studied"]]  
y = df["Marks_Scored"]      

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()
hours = 7.5
predicted_marks = model.predict([[hours]])
print(f"\nPredicted marks for {hours} hours of study = {predicted_marks[0]:.2f}")

```

## Output:
<img width="1142" height="710" alt="image" src="https://github.com/user-attachments/assets/f2baef6f-b4b2-4fd2-a029-4067a8314f76" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
