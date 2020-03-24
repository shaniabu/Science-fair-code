from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
'''
資料輸入
'''


df = pd.read_csv(r'C:\0Luna\高中\AI\randomforest\kidney_data-2.csv', header = None)
x = df.loc[:, 1:3].values
y = df.loc[:, 4].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_
X_train, X_test, y_train, y_test=train_test_split(x,y, random_state=41)

'''
Random Forest
'''
def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))
def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

a = []
b = []
c = []
for i in range (1,300):
    forest = RandomForestClassifier(criterion='gini',
                                n_estimators=i, 
                                random_state=1,
                                n_jobs=3)
    forest.fit(X_train, y_train)
    y_predict = forest.predict(X_test)
    a.append(i)
    MSE= mean_squared_error(y_test, y_predict)
    b.append(MSE)
    test_y_predicted = forest.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, test_y_predicted)
    c.append(accuracy)


import matplotlib.pyplot as plt
x = a
y = c
plt.plot(x, y) 
plt.xlabel("Number of Decision Trees")
plt.ylabel("Accuracy")
plt.title("The Number of decision Trees and Accuracy")






