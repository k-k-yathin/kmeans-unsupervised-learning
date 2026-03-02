import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # Randomly initialize centroids
        random_indices = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # Step 1: Assign clusters
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            # Step 2: Update centroids
            new_centroids = np.array([
                X[self.labels == i].mean(axis=0)
                for i in range(self.k)
            ])

            # Step 3: Check convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)


# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)

# Apply K-Means
kmeans = KMeans(k=3)
kmeans.fit(X)
labels = kmeans.labels

# Plot results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.title("True Clusters")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(
    kmeans.centroids[:, 0],
    kmeans.centroids[:, 1],
    c='red',
    marker='X',
    s=200
)
plt.title("K-Means Clustering")

plt.show()

print("Centroids:\n", kmeans.centroids)
