def main():
    print("Starting the Customer Segmentation Pipeline...")

    print("Step 1: Preprocessing Data...")
    data = preprocess_data()
    print("Data Preprocessing Completed.")

    print("Step 2: Training the Model...")
    kmeans_model, clusters = train_model(data)
    print("Model Training Completed.")

    print("Step 3: Evaluating the Model...")
    evaluate_model(data, clusters, kmeans_model)

if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    from model_Training import train_model
    from model_evaluation import evaluate_model
    main()
