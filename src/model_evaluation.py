import matplotlib.pyplot as plt  # type: ignore
from sklearn.metrics import silhouette_score # type: ignore
import pandas as pd # type: ignore

def evaluate_model(data: pd.DataFrame, clusters, kmeans_model):
    # Extract clustering features
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Silhouette Score
    score = silhouette_score(X, clusters)
    print(f"\nSilhouette Score: {score:.3f} (closer to 1 is better)\n")

    # Add cluster labels to data
    data['Cluster'] = clusters

    # Cluster summary
    print("Cluster Averages:")
    print(data.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())

    # Plotting clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis', edgecolor='k', alpha=0.6)

    # Plot centroids
    if hasattr(kmeans_model, 'cluster_centers_'):
        centers = kmeans_model.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')

    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Customer Segments with K-Means Clusters')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
