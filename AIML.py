import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet



data = {
    'Size': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    'Price': [30, 50, 65, 90, 110, 130, 150, 170, 190, 210]
}

df=pd.DataFrame(data)
print(df)

X = df[['Size']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

ridge_model=Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

lasso_model=Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Predicted prices:", predictions)
print("Actual prices:", y_test.values)

print("Ridge prediction ",ridge_model.predict(X_test))
print("Lasso prediction ",lasso_model.predict(X_test))


elastic_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic_model.fit(X_train, y_train)

print("ElasticNet prediction ", elastic_model.predict(X_test))

# logical regression (binary classification )

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = {
    'HoursStudied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Passed': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
print(df)

X = df[['HoursStudied']]
y = df['Passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Predicted (0/1):", predictions)
print("Actual (0/1):", y_test.values)

probabilities = model.predict_proba(X_test)
print("Probabilities:", probabilities)

# Naive Baye's :

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = {
    'Weight': [100, 110, 120, 130, 140, 150, 155, 160, 165, 170, 175, 180, 190, 200, 210, 220],
    'IsOrange': [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
print(df)

X = df[['Weight']]
y = df['IsOrange']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Predicted:", predictions)
print("Actual:", y_test.values)

new_fruits = pd.DataFrame({'Weight': [160, 170]})
new_predictions = model.predict(new_fruits)
print("Predictions for 160 and 170:", new_predictions)

probabilities = model.predict_proba(new_fruits)
print("Probabilities:", probabilities)

#KNeighbors Classifier:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = {
    'Weight': [100, 110, 120, 130, 140, 150, 155, 160, 165, 170, 175, 180, 190, 200, 210, 220],
    'IsOrange': [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
print(df)

X = df[['Weight']]
y = df['IsOrange']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Predicted:", predictions)
print("Actual:", y_test.values)

new_fruits = pd.DataFrame({'Weight': [160, 170]})
new_predictions = model.predict(new_fruits)
print("Predictions for 160 and 170:", new_predictions)

probabilities = model.predict_proba(new_fruits)
print("Probabilities:", probabilities)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

df = pd.read_csv(r'C:\Users\Usert\Desktop\train.csv')

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

features = df[['Pclass', 'Sex', 'Age', 'Fare']]
target = df['Survived']

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25, random_state=42)

model = LogisticRegression()
model.fit(features_train, target_train)

predictions = model.predict(features_test)

accuracy = accuracy_score(target_test, predictions)
print("Accuracy:", accuracy)

cm = confusion_matrix(target_test, predictions)
print("Confusion Matrix:\n", cm)

precision = precision_score(target_test, predictions)
recall = recall_score(target_test, predictions)
print("Precision:", precision)
print("Recall:", recall)

# Decision tree classifiers:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

df = pd.read_csv(r'C:\Users\Usert\Desktop\train.csv')

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

features = df[['Pclass', 'Sex', 'Age', 'Fare']]
target = df['Survived']

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25, random_state=42)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(features_train, target_train)

predictions = model.predict(features_test)

accuracy = accuracy_score(target_test, predictions)
print("Accuracy:", accuracy)

cm = confusion_matrix(target_test, predictions)
print("Confusion Matrix:\n", cm)

precision = precision_score(target_test, predictions)
recall = recall_score(target_test, predictions)
print("Precision:", precision)
print("Recall:", recall)


# Decision tree regression:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = {
    'Size': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    'Price': [30, 50, 65, 90, 110, 130, 150, 170, 190, 210]
}

df = pd.DataFrame(data)
print(df)

X = df[['Size']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=3, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Predicted prices:", predictions)
print("Actual prices:", y_test.values)