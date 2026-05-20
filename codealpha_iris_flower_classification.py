!pip install kaggle

import os

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_7e0e55be19f9b603e8861a59b504b43b'

!kaggle datasets download -d saurabh00007/iriscsv
!unzip iriscsv.zip
!ls

import pandas as pd
df = pd.read_csv("Iris.csv")
df.head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy of Random Forest Model:", accuracy)

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=df['Species'].unique()
))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("\nPredicted Flower Species:")
print(prediction[0])
