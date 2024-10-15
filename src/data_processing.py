import pandas as pd

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess the data for model training."""
    # Handle missing values, if any
    data = data.dropna()

    # Select features and target variable
    X = data[['Underground Temperature (°C)', 'Target Object Temperature (°C)', 
               'Humidity (%)', 'Distance (cm)', 'Proximity', 'Motion Detected']]
    y = data['Human Presence']
    
    return X, y

if __name__ == "__main__":
    # Example usage
    data = load_data('C:\SURVEILLANCE_HUMAN_DETECTION\data\synthetic_sensor_dataset.csv')
    X, y = preprocess_data(data)
    print(X.head())  # Display the first few rows of features
