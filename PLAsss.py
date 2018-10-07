import random
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC


class PLA():
    def __init__(self):
        pass

    def fit(self, arrs, labels):
        arrs = np.array(arrs)
        # 前行后列
        n, m = arrs.shape
        self.m = m
        threshold = 0
        pocket = {
            'weights': [0 for i in range(m)],
            'biasing': 0,
            'errs': np.inf
        }
        weights = pocket['weights']
        biasing = pocket['biasing']
        while True:
            errs = []
            errs_l = []
            w = weights.copy()
            b = biasing
            for row, label in zip(arrs, labels):
                c = np.dot(row, weights)
                if np.sign(c+biasing) != label:
                    errs.append(row)
                    errs_l.append(label)
                    threshold += 1
            if threshold >= 20000 or not errs:
                print('errs:', pocket['errs']/len(arrs))
                self.weights = pocket['weights'].copy()
                self.biasing = pocket['biasing']
                return
            if len(errs) <= pocket['errs']:
                pocket['errs'] = len(errs)
                pocket['weights'] = w.copy()
                pocket['biasing'] = b
                pre = len(errs)/len(arrs)
                print(f'Error {pre} with {threshold} correction')
            for row, label in zip(errs, errs_l):
                c = np.dot(row, weights)
                p = random.uniform(0, 1)
                weights = weights + p * label * row
                biasing = biasing * p - c * (1 - p)

    def predict(self, arrs):
        return np.array([np.sign(np.dot(self.weights, row)+self.biasing) for row in arrs])


def fill_name(df):
    places = []
    noble = ('Dona', 'Lady', 'Sir', 'the Countess', 'Royalty')
    # middle = ('Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer')
    for name in df['Name']:
        if any([title in name for title in noble]):
            places.append(1)
        # elif any([title in name for title in middle]):
        #     places.append(2)
        else:
            places.append(0)
    df['Name'] = places
    return df


def age_fare(train_data):
    ages = []
    fares = []
    for line1, line2 in zip(train_data['Age'], train_data['Fare']):
        x = int((line1/20))
        if x > 3:
            x = 2
        elif x > 0:
            x = 1
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
    def num(ij):
        if not ij:
            return ij
        if ij <= 3:
            return 2
        else:
            return 1
    f = [num(i+j) for i, j in zip(s, p)]
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


# pla:0.77511
# svm:0.76555
tran = lambda x: x if int(x) == 1 else -1
p = PLA()
dic_frame = {}
dic_frame0 = {}
signed = ['Pclass', 'Sex', 'Age', 'Parch', 'Embarked', 'Fare', 'Cabin', 'Name']
test_data = pd.read_csv('test.csv')
passenger_id = test_data['PassengerId']
test_data = fill_train_date(test_data)
test_data = fill_cabin(test_data)
test_data = family_size(test_data)
test_data = age_fare(test_data)
test_data = fill_name(test_data)
test_data = pd.DataFrame(test_data, columns=signed)
train_data = pd.read_csv('train.csv')
train_data = fill_train_date(train_data)
train_data = fill_cabin(train_data)
train_data = family_size(train_data)
train_data = age_fare(train_data)
train_data = fill_name(train_data)
y = train_data['Survived']
y = [tran(i) for i in y]
train_data = pd.DataFrame(train_data, columns=signed)
print(train_data.info())
p.fit(train_data.values, y)
print(list(p.weights))
print(p.biasing)
p_pred = p.predict(test_data.values)
rev = lambda x: x if int(x) == 1 else 0
p_pred = [np.int64(rev(i)) for i in p_pred]
dataframe = pd.DataFrame({'PassengerId': passenger_id, 'Survived': p_pred})
dataframe.to_csv("application.csv", index=False, sep=',')
