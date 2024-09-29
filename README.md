# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.
 ```
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Padmavathi M
RegisterNumber:  212223040141
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
print(mse)
r2=metrics.r2_score(y_test,y_pred)
print(r2)
dt.predict([[5,6]])
```
## Output:
## Data.head()
![image](https://github.com/user-attachments/assets/d01ad517-272f-4f9c-bfb9-6a10415ac7e8)
## data.info()
![image](https://github.com/user-attachments/assets/19122a2a-c302-4567-8a1b-b4296548ee04)
## isnull() and sum()
![image](https://github.com/user-attachments/assets/0d87e70e-83e5-4620-b404-aebcc63061bd)
## MSE value
![image](https://github.com/user-attachments/assets/ba78d986-da4d-4c00-b77a-09a4ca4f283a)
## R2 value
![image](https://github.com/user-attachments/assets/adc14317-ddfa-422f-9bf2-0a23913b07c7)
## Data prediction 
![image](https://github.com/user-attachments/assets/9df2f6b2-3562-4627-9b2c-7e9049b869e4)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
