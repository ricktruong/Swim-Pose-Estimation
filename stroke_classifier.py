import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings

# Filter out the specific warning
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics._classification')

def prepare_data(csv_path):
    """Prepare the data for training by grouping keypoints by frame."""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Group by stroke and frame
    grouped = df.groupby(['stroke', 'frame'])
    
    # Create features by flattening x,y coordinates for each frame
    X = []
    y = []
    
    for (stroke, frame), group in grouped:
        # Sort by index to ensure consistent order of keypoints
        group = group.sort_index()
        
        # Extract x,y coordinates and flatten them
        coords = group[['x', 'y']].values.flatten()
        
        # Only include frames where we have all keypoints (no zeros)
        if not np.any(coords == 0):
            X.append(coords)
            y.append(stroke)
    
    return np.array(X), np.array(y)

def train_classifier(X, y):
    """Train a Random Forest classifier on the keypoint data."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate the classifier
    y_pred = clf.predict(X_test_scaled)
    
    # Calculate and print test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save the model and scaler
    joblib.dump(clf, 'models/stroke_classifier.joblib')
    joblib.dump(scaler, 'models/stroke_scaler.joblib')
    
    return clf, scaler, test_accuracy

def predict_stroke(frame_keypoints, clf, scaler):
    """Predict stroke type for a single frame's keypoints."""
    # Ensure the input is in the correct shape (34x1)
    if len(frame_keypoints.shape) == 1:
        frame_keypoints = frame_keypoints.reshape(1, -1)
    
    # Scale the keypoints
    scaled_keypoints = scaler.transform(frame_keypoints)
    
    # Predict
    prediction = clf.predict(scaled_keypoints)
    probabilities = clf.predict_proba(scaled_keypoints)
    
    return prediction[0], probabilities[0]

def main():
    KEYPOINTS_CSV_FILE = 'data/keypoints/keypoints.csv'

    # Prepare the data
    X, y = prepare_data(KEYPOINTS_CSV_FILE)
    print(X)
    print(y)
    print("Shape of X:", X.shape)  # Should be (n_samples, 34)
    print("Number of unique strokes:", len(np.unique(y)))
    print("Unique strokes:", np.unique(y))
    
    if len(X) < 10:
        print("\nWarning: Very small dataset detected. Consider collecting more data.")
        print("The model might not perform well with such limited data.")
    
    # Train the classifier
    clf, scaler, test_accuracy = train_classifier(X, y)
    
    # Example of how to use the classifier for a single frame
    print("\nExample prediction:")
    # Get a random frame from the test set
    X_test, _, y_test, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    random_idx = np.random.randint(len(X_test))
    example_frame = X_test[random_idx]
    
    prediction, probabilities = predict_stroke(example_frame, clf, scaler)
    print(f"Predicted stroke: {prediction}")
    print(f"True stroke: {y_test[random_idx]}")
    print("Class probabilities:", dict(zip(clf.classes_, probabilities)))

if __name__ == "__main__":
    main() 