import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

df = pd.read_csv('./data/titanic.csv')
df.drop(['Embarked', 'Cabin', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Name'], axis=1, inplace=True)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
y = df['Survived']
x = df.drop(['Survived'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
params = {

}
clf = LogisticRegression()
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
print('Score on train: '+str(clf.score(x_train, y_train)))
print('Score on test: '+str(clf.score(x_test, y_test)))
