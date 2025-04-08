from data_preprocessing import preprocess_data
from model_Training import train_model
from model_evaluation import evaluate_model

def main():
    print("Starting the Customer Segmentation Pipeline...")

    # Step 1: Data Preprocessing
    print("Step 1: Preprocessing Data...")
    data = preprocess_data()
    print("Data Preprocessing Completed.")

    # Step 2: Model Training
    print("Step 2: Training the Model...")
    model, clusters = train_model(data)
    print("Model Training Completed.")

    # Step 3: Model Evaluation
    print("Step 3: Evaluating the Model...")
    evaluate_model(data, clusters)
    print("Model Evaluation Completed.")

    print("Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()