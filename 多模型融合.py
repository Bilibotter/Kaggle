import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression


def age_fare(train_data):
    ages = []
    fares = []
    for line1, line2 in zip(train_data['Age'], train_data['Fare']):
        x = int((line1/20))
        x = 1 if x > 1 else x
        ages.append(x)
        y = int((line2/15))
        y = 1 if y > 1 else y
        fares.append(y)
    train_data['Age'] = ages
    train_data['Fare'] = fares
    return train_data


def family_size(train_data):
    s = train_data['SibSp']
    p = train_data['Parch']
    f = [i+j for i, j in zip(s, p)]
    train_data['Parch'] = f
    return train_data


def fill_cabin(train_data):
    cabin = []
    for line in train_data['Cabin']:
        if pd.isnull(line):
            cabin.append(1)
        else:
            cabin.append(0)
    train_data['Cabin'] = cabin
    return train_data


def fill_train_date(train_data):
    # fill age
    mean_age = int(train_data['Age'].mean())
    train_data.fillna({'Age': mean_age}, inplace=True)
    # fill embarked
    dic = {}
    trans = lambda x: float('%.2f' % x)
    S = train_data[train_data['Embarked'] == 'S']
    C = train_data[train_data['Embarked'] == 'C']
    Q = train_data[train_data['Embarked'] == 'Q']
    dic['S'] = 0
    dic['C'] = 1
    dic['Q'] = 0
    train_data.replace(to_replace=dic, inplace=True)
    train_data.fillna({'Embarked': 1}, inplace=True)
    # fill sex
    train_data.replace(to_replace={'male': 0, 'female': 1}, inplace=True)
    # Fare
    mean_fare = int(train_data['Fare'].mean())
    train_data.fillna({'Fare': mean_fare}, inplace=True)
    return train_data


dic_frame = {}
dic_frame0 = {}
signed = ['Pclass', 'Sex', 'Age', 'Parch', 'Embarked', 'Fare', 'Cabin']
dic = {'S': 0, 'C': 1, 'Q': 0, 'male': 0, 'female': 1, 'Age': 29}
train_data = pd.read_csv('train.csv')
train_data = fill_train_date(train_data)
train_data = fill_cabin(train_data)
train_data = family_size(train_data)
train_data = age_fare(train_data)
for col in signed:
    dic_frame0[col] = train_data[col]
y = train_data['Survived']
train_data1 = pd.DataFrame(dic_frame0)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data1, y)
logistic = LogisticRegression()
logistic.fit(train_data1, y)
SVM = svm.SVC()
SVM.fit(train_data1, y)
dic_frame0 = {}
test_data = pd.read_csv('test.csv')
passenger_id = test_data['PassengerId']
test_data = fill_train_date(test_data)
test_data = fill_cabin(test_data)
test_data = family_size(test_data)
test_data = age_fare(test_data)
for col in signed:
    dic_frame0[col] = test_data[col]
# dic_frame0为决策树
test_data1 = pd.DataFrame(dic_frame0)
s_pred = SVM.predict(test_data1)
l_pred = logistic.predict(test_data1)
c_pred = clf.predict(test_data1)
y_pred = []
for i, j, k in zip(s_pred, l_pred, c_pred):
    if j == k:
        y_pred.append(j)
    else:
        y_pred.append(i)
dataframe = pd.DataFrame({'PassengerId': passenger_id, 'Survived': y_pred})
dataframe.to_csv("kaggle8.csv", index=False, sep=',')
