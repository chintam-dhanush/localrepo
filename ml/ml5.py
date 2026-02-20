import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'])
y = iris.target

colors = np.array(['red','lime','black'])

# 🔹 Real Classification
plt.figure()
plt.scatter(X.Petal_Length, X.Petal_Width, c=colors[y])
plt.title("Real Classification")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

# 🔹 K-Means
km = KMeans(n_clusters=2).fit(X)

plt.figure()
plt.scatter(X.Petal_Length, X.Petal_Width, c=colors[km.labels_])
plt.title("K-Means Classification")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

print("K-Means Accuracy:", accuracy_score(y, km.labels_))
print(confusion_matrix(y, km.labels_))

# 🔹 EM (GMM)
Xs = StandardScaler().fit_transform(X)
gmm = GaussianMixture(n_components=3).fit(Xs)
y_gmm = gmm.predict(Xs)

plt.figure()
plt.scatter(X.Petal_Length, X.Petal_Width, c=colors[y_gmm])
plt.title("EM (GMM) Classification")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

print("EM Accuracy:", accuracy_score(y, y_gmm))
print(confusion_matrix(y, y_gmm))