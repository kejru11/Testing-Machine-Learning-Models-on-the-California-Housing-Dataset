from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#getting the dataset and adding the target column
housing= fetch_california_housing();
print(housing)
housing_df= pd.DataFrame(housing["data"], columns= housing["feature_names"]);
housing_df["target"]= housing["target"];
housing_df.to_csv("California_Housing_Dataset.csv")

#importing the algorithm

#importing the Ridge() algorithm
from sklearn.linear_model import Ridge

#setting up the random seed
np.random.seed(42)

#dropping the target value from the dataset

X= housing_df.drop("target", axis=1)
y= housing_df["target"]

#splitting the data into training and test sets
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2)
model = Ridge()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

#the score of the Ridge model is about 0.57 which is not ideal. 

#importing the Lasso Algorithm
from sklearn import linear_model
model1= linear_model.Lasso(alpha=0.1)
model1.fit(X_train, y_train)
print(model1.score(X_test, y_test))

#the Lasso model returns a result of about 0.53 which is even worse than the previous one.

#importing the ElasticNet Model
from sklearn.linear_model import ElasticNet
model2= ElasticNet(random_state=0)
model2.fit(X_train, y_train)
print(model2.score(X_test, y_test))

#the ElasticNet model returns a result of about 0.41 that is not ideal

#importing the SVR kernel=linear Model
from sklearn.svm import LinearSVR
model3= LinearSVR(random_state=0)
model3.fit(X_train, y_train)
print(model3.score(X_test, y_test))
\
#the LinearSVR model returns a result of about 0.149 that is the worse of them all

#imprting the Ensemble Regressors
from sklearn.ensemble import RandomForestRegressor
model4= RandomForestRegressor()
model4.fit(X_train, y_train)
print(model4.score(X_test, y_test))

#the RnadomForestRegressor() model returns a result of about 0.806 that is the best of them all