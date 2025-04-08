from sklearn.cluster import KMeans # type: ignore

def train_model(data):
    # Select features for clustering
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Train K-Means model
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    return kmeans, clusters

if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    data = preprocess_data()
    model, clusters = train_model(data)
    print("Trained Model and Clusters:")
    print(clusters)