

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

iris = load_iris()
X = iris.data
y_true = iris.target
print('Dataset shape:', X.shape)

linked = linkage(X, method='ward')
plt.figure()
dendrogram(linked)
plt.title('Dendrogram for Iris Dataset')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

hc = AgglomerativeClustering(n_clusters=3)
hc_labels = hc.fit_predict(X)
print('Hierarchical Cluster Labels:')
print(hc_labels)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
print('K-Means Cluster Labels:')
print(kmeans_labels)

hc_sil = silhouette_score(X, hc_labels)
km_sil = silhouette_score(X, kmeans_labels)
hc_ari = adjusted_rand_score(y_true, hc_labels)
km_ari = adjusted_rand_score(y_true, kmeans_labels)
print('Hierarchical Silhouette Score:', hc_sil)
print('K-Means Silhouette Score:', km_sil)
print('Hierarchical ARI:', hc_ari)
print('K-Means ARI:', km_ari)

plt.scatter(X[:,0], X[:,1], c=kmeans_labels)
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

plt.scatter(X[:,0], X[:,1], c=hc_labels)
plt.title('Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

