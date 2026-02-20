import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

df = pd.read_csv('/content/diabetes (5).csv')

X = df.drop('Outcome', axis=1)   # drop target column
y = df['Outcome']

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33)

clf = GaussianNB().fit(xtrain, ytrain)
pred = clf.predict(xtest)

print(metrics.confusion_matrix(ytest, pred))
print("Accuracy:", metrics.accuracy_score(ytest, pred))
print("Precision:", metrics.precision_score(ytest, pred))
print("Recall:", metrics.recall_score(ytest, pred))
print("F1:", metrics.f1_score(ytest, pred))

print("Individual prediction:",
      clf.predict([[6,148,72,35,0,33.6,0.627,50]]))