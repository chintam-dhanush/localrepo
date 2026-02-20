import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Wrong Prediction:", 1 - accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

sample = [[5.1, 3.5, 7.4, 0.2]]
pred = knn.predict(sample)

print("Predicted class:", pred)
print("Flower name:", iris.target_names[pred])