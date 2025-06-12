#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Euclidean Distance Function
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Custom KNN Classifier
class MyKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)  # Vectorized distance
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

# Train and evaluate the custom KNN
my_knn = MyKNN(k=5)
my_knn.fit(X_train, y_train)
custom_accuracy = my_knn.score(X_test, y_test)
print(f"🧠 Custom KNN Accuracy: {custom_accuracy * 100:.2f}%")

# Train and evaluate scikit-learn's KNN for comparison
sklearn_knn = KNeighborsClassifier(n_neighbors=5)
sklearn_knn.fit(X_train, y_train)
sklearn_accuracy = sklearn_knn.score(X_test, y_test)
print(f"🤖 scikit-learn KNN Accuracy: {sklearn_accuracy * 100:.2f}%")

# PCA: Reduce to 2D for visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Refit custom KNN on the reduced data for plotting
my_knn.fit(X_reduced, y)
custom_predictions = my_knn.predict(X_reduced)

# Plot predicted classes using PCA components
plt.figure(figsize=(8, 6))
for i, label in enumerate(np.unique(y)):
    plt.scatter(
        X_reduced[np.array(custom_predictions) == label, 0],
        X_reduced[np.array(custom_predictions) == label, 1],
        label=f"Predicted {target_names[label]}",
        alpha=0.6
    )

plt.title("KNN Prediction (PCA-reduced Iris Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




