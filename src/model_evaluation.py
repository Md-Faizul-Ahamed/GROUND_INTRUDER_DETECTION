import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from . import data_processing as dp

def evaluate_model(X_test, y_test):
    """Evaluate the decision tree model."""
    # Load the trained model
    model = joblib.load('model/decision_tree_model.joblib')
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    data = dp.load_data('C:\SURVEILLANCE_HUMAN_DETECTION\data\synthetic_sensor_dataset.csv')
    X, y = dp.preprocess_data(data)
    _, X_test, y_test = train_model(X, y)  # Call model training to get test data
    evaluate_model(X_test, y_test)
