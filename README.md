# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
<img width="688" height="710" alt="al ex3 ml" src="https://github.com/user-attachments/assets/8e98d670-f3a5-4679-8d4f-8212415ace83" />

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Kabira A
RegisterNumber:212224040146
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
print(data.head())
print("\n")
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print("\n")
print(X1_Scaled)
print("\n")
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
    
*/
```

## Output:
### DATA INFORMATION

<img width="741" height="163" alt="data structure" src="https://github.com/user-attachments/assets/c5f37be4-9b71-4633-a4c1-d1cec11df338" />

### VALUE OF X

<img width="301" height="723" alt="ex3 2" src="https://github.com/user-attachments/assets/4e4d1353-fdef-4638-a6bf-56737e0057c7" />

### VALUE OF X1_SCALED

<img width="382" height="783" alt="ex3 3 ml" src="https://github.com/user-attachments/assets/b86293fb-f243-45d8-8498-14e199042666" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
