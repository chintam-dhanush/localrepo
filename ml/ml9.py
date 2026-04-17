# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

diabetic=pd.read_csv('/content/dia.csv')

diabetic

diabetic.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sb

sb.heatmap(diabetic.corr())

Y=diabetic['Outcome']
Y

X=diabetic.drop(['Outcome'],axis=1)
X

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.ensemble import AdaBoostClassifier

model=AdaBoostClassifier()

model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)

Y_pred

from sklearn.metrics import accuracy_score
print('Ada Boost accuarcy is',accuracy_score(Y_test,Y_pred)*100)

from sklearn.metrics import confusion_matrix

print('Confusion matrix is \n',confusion_matrix(Y_test,Y_pred)*100)

