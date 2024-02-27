# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```
```


## Program:
```
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: E sreenithi
RegisterNumber: 212223220109 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
```


### Output:
df.haed():

![image](https://github.com/sreenithi123/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145743046/10c41eed-c208-4758-8677-f9bc3689fa17)

plot:
![image](https://github.com/sreenithi123/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145743046/4fc8d362-eabb-4e00-a38e-7c27cb2ead34)

iloc:
![image](https://github.com/sreenithi123/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145743046/9a70701e-02be-47f4-b572-9ca2e8d96962)
Traning:
![image](https://github.com/sreenithi123/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145743046/06166831-e09e-4201-a67a-0effba57a61f)

Graph:
![image](https://github.com/sreenithi123/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145743046/4fafd679-9e81-4ec7-a78b-86442054a62a)









## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
