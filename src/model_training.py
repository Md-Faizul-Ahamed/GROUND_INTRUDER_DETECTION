from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os  # Import os to work with directories
from . import data_processing as dp

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    model_dir = 'model'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(model, os.path.join(model_dir, 'decision_tree_model.joblib'))
    
    return model, X_test, y_test

if __name__ == "__main__":
    data = dp.load_data('C:\SURVEILLANCE_HUMAN_DETECTION\data\synthetic_sensor_dataset.csv')
    
    X, y = dp.preprocess_data(data)
    
    model, X_test, y_test = train_model(X, y)
    
    print("Model training complete.")

