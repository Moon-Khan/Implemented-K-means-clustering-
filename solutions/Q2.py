import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("D:\study\semester 6\AI\AI_A5_2023Spring\AI_A5_2023Spring\pythonProject6/data.csv")
data.head()

X = data.values
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)


def kmeans(X, k=2, max_iterations=100):

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

    return P


P = kmeans(X)
assert len(data) == len(P)
X = sc.inverse_transform(X)
plt.figure(figsize=(15,10))
plt.scatter(X[:,0],X[:,1],c=P)
plt.show()
