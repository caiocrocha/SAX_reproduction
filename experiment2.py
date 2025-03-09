import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sax import SAX
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import contextlib
import io

class KMeansWithInertia(KMeans):
    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-4, verbose=0, random_state=None, copy_x=True, algorithm='lloyd'):
        super().__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        self.inertia_values = []

    def fit(self, X, y=None):
        super().fit(X, y)
        return self

    def _kmeans_single_lloyd(self, X, sample_weight, centers_init, max_iter=300, verbose=False, tol=1e-4, x_squared_norms=None, random_state=None, reassignment_ratio=0.01):
        for i in range(max_iter):
            centers_old = centers_init.copy()
            labels, inertia = self._labels_inertia(X, sample_weight, centers_init, x_squared_norms)
            self.inertia_values.append(inertia)
            centers_init = self._centers_dense(X, sample_weight, labels, self.n_clusters, centers_old)
            if np.sum((centers_old - centers_init) ** 2) < tol:
                break
        return labels, inertia, centers_init, i + 1

def capture_kmeans(X, k=2):
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        kmeans = KMeansWithInertia(n_clusters=k, random_state=42, max_iter=300, n_init=1, verbose=1)
        kmeans.fit(X)

    output = output.getvalue().split('\n')[1:-2]
    kmeans.inertia_values = np.array([float(line.strip('.').split()[-1]) for line in output])
    return kmeans

if __name__ == '__main__':
    # Load dataset
    df = pd.read_csv('household/household_power_consumption.txt', delimiter=';', low_memory=False)
    df = df.dropna()  # Drop rows with missing values

    # Preprocess data
    X = df.iloc[:, 2:].astype(float).values  # Features

    # Normalize data BEFORE subsequence extraction
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split X into subsequences of 512 observations
    subsequence_length = 512
    num_subsequences = 1000
    X_subsequences = X[:num_subsequences * subsequence_length].reshape(-1, subsequence_length)

    # Perform KMeans clustering on raw data
    k = 2
    kmeans_raw = capture_kmeans(X_subsequences, k)
    kmeans_raw_mean = kmeans_raw.inertia_values.mean()

    # SAX Transformation
    w = 8
    a = 4
    X_sax = []
    for i in range(X_subsequences.shape[0]):  # Iterate through subsequences
        sax_representation = SAX(w=w, a=a, ts=X_subsequences[i, :]).transform() #Transform each subsequence separately
        X_sax.append([ord(char) - ord('a') + 1 for char in sax_representation]) #Convert to numerical representation

    X_sax = np.array(X_sax)

    # Scale SAX data
    scaler_sax = StandardScaler()
    X_sax = scaler_sax.fit_transform(X_sax)

    # Perform KMeans clustering on SAX data
    kmeans_sax = capture_kmeans(X_sax, k)
    kmeans_sax_mean = kmeans_sax.inertia_values.mean()
    kmeans_sax.inertia_values = kmeans_sax.inertia_values * (kmeans_raw_mean / kmeans_sax_mean)

    # Plot convergence of inertia
    plt.figure()
    plt.plot(range(len(kmeans_raw.inertia_values)), kmeans_raw.inertia_values, marker='D', label='Raw data', c='black')
    plt.plot(range(len(kmeans_sax.inertia_values)), kmeans_sax.inertia_values, marker='s', label='SAX', c='grey')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Inertia')
    plt.title(f'KMeans Clustering Convergence (K={k})')
    plt.legend()
    plt.grid(axis='y')

    plt.savefig('kmeans_convergence.png')
    plt.show()
