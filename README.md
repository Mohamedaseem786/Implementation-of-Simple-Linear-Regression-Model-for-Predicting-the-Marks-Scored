# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data and Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Karthikeyan.K
RegisterNumber:  212221230046
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
df.head()
df.tail()
#segregation data to variables
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
#spliting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
#displaying predicted value
Y_pred
#displaying actual value
Y_test
#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_train,regressor.predict(X_train),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
from sklearn.metrics import mean_absolute_error,mean_squared_error
mse=mean_squared_error(Y_test,Y_pred)
print('MSC=',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE=',mae)

rmse=np.sqrt(mse)
print("RMSE=",rmse)
```

## Output:
![op_head](https://user-images.githubusercontent.com/93427303/198876059-34b6a735-69a2-4846-8de1-2e6cf481f813.png)

![op1](https://user-images.githubusercontent.com/93427303/196491270-d83638e5-3596-45af-99cc-3be9c892280a.png)

![op2](https://user-images.githubusercontent.com/93427303/196491294-5032a4d0-28ea-4464-8f60-9f3e15e15fa8.png)

![op3](https://user-images.githubusercontent.com/93427303/196491316-4f09f6c0-00f3-488f-8d7d-ea135310aa77.png)

![op4](https://user-images.githubusercontent.com/93427303/196491341-05fd33f7-5737-42f5-ae76-548882474303.png)

![op5](https://user-images.githubusercontent.com/93427303/196491354-141e4c07-dccc-4508-8029-e56941a58fad.png)

![op6](https://user-images.githubusercontent.com/93427303/196491367-ef6b0679-05ab-4ebb-9d16-487d3fccdce4.png)

![op_error](https://user-images.githubusercontent.com/93427303/198876063-f8d4b755-1c13-4d98-98c7-d9999c04e37f.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
