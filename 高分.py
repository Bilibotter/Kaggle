import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression


def k_div(df):
    signed = ['Pclass', 'Sex', 'Age', 'Parch', 'Embarked', 'Fare', 'Cabin', 'Name']
    keys = lambda: {key: [] for key in signed}
    dic = {str(i): keys() for i in range(10)}
    return dic


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
            x = 1.5
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
    f = [i+j for i, j in zip(s, p)]
    train_data['Parch'] = f
    return train_data


# def fill_cabin(df):
#     nums = []
#     token = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'nan', 'T')
#     num = (10, 20, 15, 20, 20, 15, 10, 0, 0)
#     c_to_num = dict(zip(token, num))
#     for cabin in df['Cabin']:
#         if pd.isna(cabin):
#             nums.append(c_to_num['nan'])
#         else:
#             start = cabin.split()[0]
#             num = c_to_num[start[0]]
#             nums.append(num)
#     df['Cabin'] = nums
#     return df
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
signed = ['Pclass', 'Sex', 'Age', 'Parch', 'Embarked', 'Fare', 'Cabin', 'Name']
test_data = pd.read_csv('test.csv')
passenger_id = test_data['PassengerId']
test_data = fill_train_date(test_data)
test_data = fill_cabin(test_data)
test_data = family_size(test_data)
test_data = age_fare(test_data)
test_data = fill_name(test_data)
for col in signed:
    dic_frame0[col] = test_data[col]
test_data = pd.DataFrame(dic_frame0)
train_datas = pd.read_csv('train.csv')
y_preds = np.array([0 for i in range(len(test_data))])
for i in range(10):
    train_data = train_datas.loc[train_datas['PassengerId'] % 10 != i].copy()
    train_data = fill_train_date(train_data)
    train_data = fill_cabin(train_data)
    train_data = family_size(train_data)
    train_data = age_fare(train_data)
    train_data = fill_name(train_data)
    for col in signed:
        dic_frame0[col] = train_data[col]
    y = train_data['Survived']
    train_data = pd.DataFrame(dic_frame0)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, y)
    logistic = LogisticRegression()
    logistic.fit(train_data, y)
    SVM = svm.SVC()
    SVM.fit(train_data, y)
    s_pred = SVM.predict(test_data)
    l_pred = logistic.predict(test_data)
    c_pred = clf.predict(test_data)
    # y_pred = []
    # for i, j, k in zip(s_pred, l_pred, c_pred):
    #     if j == k:
    #         y_pred.append(j)
    #     else:
    #         y_pred.append(i)
    y_preds += np.array(s_pred)
    y_preds += np.array(l_pred)
    y_preds += np.array(c_pred)
y_preds = [int(num>14) for num in y_preds]
dataframe = pd.DataFrame({'PassengerId': passenger_id, 'Survived': y_preds})
dataframe.to_csv("kaggle16.csv", index=False, sep=',')
