from sklearn.cluster import KMeans  # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def find_optimal_k(X, max_k=10):
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    # Plot Elbow Curve
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method - Inertia')

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'go-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs k')

    plt.tight_layout()
    plt.show()

    # Automatically select k with highest silhouette score
    best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"\n[INFO] Optimal number of clusters (k) based on silhouette score: {best_k}")
    return best_k

def train_model(data):
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    print("Finding optimal number of clusters using Elbow + Silhouette...")
    best_k = find_optimal_k(X)

    print(f"\nTraining final model with k = {best_k}")
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    clusters = kmeans.fit_predict(X)
    return kmeans, clusters
