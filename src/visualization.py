import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import joblib
from . import data_processing as dp

def visualize_tree():
    model = joblib.load('model/decision_tree_model.joblib')
    
    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=['Underground Temperature (°C)', 
                                                  'Target Object Temperature (°C)', 
                                                  'Humidity (%)', 
                                                  'Distance (cm)', 
                                                  'Proximity', 
                                                  'Motion Detected'], 
              class_names=['No Human', 'Human'], rounded=True)
    plt.title("Decision Tree Visualization")
    plt.show()

if __name__ == "__main__":
    visualize_tree()
