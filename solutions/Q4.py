import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("D:\study\semester 6\AI\AI_A5_2023Spring\AI_A5_2023Spring\pythonProject6/data.csv")
X = data.values
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)


def kmeans(X, k, max_iterations=100):
    data = np.array(X)

    centroids_idx = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[centroids_idx]

    P = np.zeros(data.shape[0])

    for _ in range(max_iterations):

        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        new_P = np.argmin(distances, axis=0)

        if np.array_equal(P, new_P) or _ == max_iterations - 1:
            break

        P = new_P
        for i in range(k):
            centroids[i] = data[P == i].mean(axis=0)

    return P, centroids


wcss = []
for k in range(1, 11):
    _, centroids = kmeans(X, k)
    wcss.append(sum(np.min(distance.cdist(X, centroids, 'euclidean'), axis=1)) / X.shape[0])

optimal_k = np.argmin(wcss) + 1
P, centroids = kmeans(X, optimal_k)

X = sc.inverse_transform(X)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=P)
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='*', c='red')
plt.title(f'K-means clustering with {optimal_k} clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
