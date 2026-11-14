import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score



titanic_set = pd.read_csv("data/Titanic-Dataset.csv") # importing data set

print(titanic_set.duplicated().sum()) # checking to verify if there are any duplicates, if none, continue with processing

print(titanic_set.isna().sum())  # checking for empty rows - if rows are empty, delete

# I am NOT removing the empty rows for the Embarked and Cabin columns because they are not going to be in the testing or training data

titanic_set.dropna(subset="Age", inplace=True) # Deleting all rows with an empty age cell, this will drop 177 rows from the whole thing.
print(titanic_set.isna().sum()) # checking that cells were deleted

X = titanic_set.drop(["PassengerId", "Survived", "Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)
y = titanic_set["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # splitting csv data into training and test sets

cat_features = ["Sex"] # this category is the only non-numeric category I have
ohe = OneHotEncoder()
transformer = ColumnTransformer([("one_hot_encoder", ohe, cat_features)], remainder = "passthrough") # transforming to numbers

transformed_x = transformer.fit_transform(X_train)

X_train, X_test, y_train, y_test = train_test_split(transformed_x, y_train, test_size=0.2) #re-establishing training and test with new data

model = LogisticRegression()
model.fit(X_train, y_train)

# This is the metric for evaluating the model - an accuracy score
y_preds = model.predict(X_test)
print("This is the accuracy score of the model: ", accuracy_score(y_test, y_preds))
