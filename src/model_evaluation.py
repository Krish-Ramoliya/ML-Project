import matplotlib.pyplot as plt # type: ignore

def evaluate_model(data, clusters):
    # Visualize the clusters
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Customer Segments')
    plt.show()

if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    from model_Training import train_model
    data = preprocess_data()
    _, clusters = train_model(data)
    evaluate_model(data, clusters)