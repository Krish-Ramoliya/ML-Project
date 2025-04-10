import pandas as pd # type: ignore

def preprocess_data():
    # Load the dataset
    data = pd.read_csv("dataset/Mall_Customers.csv")
    
    # Perform preprocessing (e.g., handling missing values, encoding)
    data = data.dropna()  # Example: Drop missing values
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})  # Encode Gender
    
    return data

if __name__ == "__main__":
    data = preprocess_data()
    print("Preprocessed Data:")
    print(data.head())