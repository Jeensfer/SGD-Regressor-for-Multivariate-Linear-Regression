# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Input house features such as size of the house and location score.
2.Prepare the target variables as house price and number of occupants.
3.Scale the input features using StandardScaler for faster SGD convergence.
4.Train a MultiOutput SGD Regressor model using the prepared dataset.
5.Use the trained model to predict the house price and number of occupants for new inputs. 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Jeensfer Jo
RegisterNumber:212225240058
*/
#Predict Price of house and number of occupants using SGD (Based on features like sqft and location score)
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Dataset
# Features: [Size(sqft), location score]
X = np.array([[800, 2],[1000, 3],[1200, 3],[1500, 4],[1800, 4]], dtype=float)

# Targets: [price, occupants]
y = np.array([[40, 3],[55, 4],[65, 4],[80, 5],[95, 6]], dtype=float)

# 2. Model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", MultiOutputRegressor(
        SGDRegressor(
            learning_rate="constant",
            eta0=0.01,
            max_iter=1000,
            tol=1e-6
        )
    ))
])

# 3. Train model
model.fit(X, y)

# 4. Prediction
H=int(input("Enter the size(sqft) of the house (600-2000 sqft): "))
L=int(input("Enter the location score of the house (1-5): "))
test_house = np.array([[H, L]])
prediction = model.predict(test_house)

print(f"Predicted House Price (Lakhs):  {prediction[0][0]:.2f}")
print("Predicted Number of Occupants:", round(prediction[0][1]))
```

## Output:
<img width="870" height="171" alt="image" src="https://github.com/user-attachments/assets/3663e010-c2a4-4bf1-9f46-c7d44b2a42af" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
