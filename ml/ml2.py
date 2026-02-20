import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df = pd.read_csv("/content/Play Tennis.csv")

df = df.drop('Day', axis=1)   # drop useless column early

le = LabelEncoder()
for c in df.columns:
    df[c] = le.fit_transform(df[c])

X = df.drop('Play_Tennis', axis=1)
y = df['Play_Tennis']

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

tree.plot_tree(clf, class_names=['No','Yes'])