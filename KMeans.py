import numpy as np
from cluster import cluster

class My_KMeans(cluster):

    def __init__(self, k = 5, max_iterations = 100):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        centroids = X[np.random.choice(X.shape[0], size=self.k, replace=False)]
        assignments = np.zeros(len(X), dtype=int)

        for _ in range(self.max_iterations):
            prev_centroids = centroids.copy()
            
            for i, point in enumerate(X):
                closest = [-1, float('inf')]
                for j, c in enumerate(centroids):
                    dist = np.sqrt(sum((p - c) ** 2 for p, c in zip(point, c)))
                    if dist < closest[1]:
                        closest = [j, dist]
                assignments[i] = closest[0]

            centroids = np.array([X[assignments == i].mean(axis=0) for i in range(self.k)])

            if np.all(prev_centroids == centroids):
                break

        self.centroids = centroids
        self.assignments = assignments
        self.X = X
        return assignments, centroids

    def inertia_(self):
        inertia = 0
        
        for i, point in enumerate(self.X):
            centroid = self.centroids[self.assignments[i]]
            inertia += np.sum((point - centroid) ** 2)
            
        return inertia