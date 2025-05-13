import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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
        coords = group[['x', 'y']].values
        
        # Only include frames where we have all keypoints (no zeros)
        # and ensure the shape is exactly (17, 2) before flattening
        if coords.shape == (17, 2):
            # Flatten the coordinates to a 34-dimensional vector and reshape to (34, 1)
            flattened_coords = coords.flatten().reshape(34, 1)
            X.append(flattened_coords)
            y.append(stroke)
    
    # Convert to numpy arrays and validate shapes
    X = np.array(X)
    y = np.array(y)
    
    # Validate the shapes
    if len(X) == 0:
        raise ValueError("No valid frames found in the dataset")
    
    if X.shape[1:] != (34, 1):  # Each sample should be 34x1
        raise ValueError(f"Invalid keypoint shape. Expected (n_samples, 34, 1), got {X.shape}")
    
    print(f"Processed {len(X)} frames with shape {X.shape}")
    return X, y

def train_classifier(X, y):
    """Train a Random Forest classifier on the keypoint data."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Reshape X to 2D array for sklearn (n_samples, n_features)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the classifier
    clf = RandomForestClassifier(
        n_estimators=150,
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
    # Ensure the input is in the correct shape (34, 1)
    if len(frame_keypoints.shape) == 1:
        frame_keypoints = frame_keypoints.reshape(34, 1)
    
    # Reshape to 2D array for sklearn (1, n_features)
    frame_keypoints = frame_keypoints.reshape(1, -1)
    
    # Scale the keypoints
    scaled_keypoints = scaler.transform(frame_keypoints)
    
    # Predict
    prediction = clf.predict(scaled_keypoints)
    probabilities = clf.predict_proba(scaled_keypoints)
    
    return prediction[0], probabilities[0]

def train_svm(X, y):
    """Train a Support Vector Machine classifier on the keypoint data."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Reshape X to 2D array for sklearn (n_samples, n_features)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the SVM classifier
    svm = SVC(
        kernel='poly',  # Radial Basis Function kernel
        C=1.0,         # Regularization parameter
        degree=3,
        probability=True,  # Enable probability estimates
        random_state=42
    )
    
    svm.fit(X_train_scaled, y_train)
    
    # Evaluate the classifier
    y_pred = svm.predict(X_test_scaled)
    
    # Calculate and print test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nSVM Test Accuracy: {test_accuracy:.4f}")
    
    print("\nSVM Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save the model and scaler
    joblib.dump(svm, 'models/stroke_svm.joblib')
    joblib.dump(scaler, 'models/stroke_svm_scaler.joblib')
    
    return svm, scaler, test_accuracy

def predict_stroke_svm(frame_keypoints, svm, scaler):
    """Predict stroke type for a single frame's keypoints using SVM."""
    # Ensure the input is in the correct shape (34, 1)
    if len(frame_keypoints.shape) == 1:
        frame_keypoints = frame_keypoints.reshape(34, 1)
    
    # Reshape to 2D array for sklearn (1, n_features)
    frame_keypoints = frame_keypoints.reshape(1, -1)
    
    # Scale the keypoints
    scaled_keypoints = scaler.transform(frame_keypoints)
    
    # Predict
    prediction = svm.predict(scaled_keypoints)
    probabilities = svm.predict_proba(scaled_keypoints)
    
    return prediction[0], probabilities[0]

def main():
    KEYPOINTS_CSV_FILE = 'data/keypoints/keypoints.csv'

    # Prepare the data
    X, y = prepare_data(KEYPOINTS_CSV_FILE)
    print("Shape of X:", X.shape)  # Should be (n_samples, 34)
    print("Shape of y:", y.shape)  # Should be (n_samples, 34)
    print("Number of unique strokes:", len(np.unique(y)))
    print("Unique strokes:", np.unique(y))
    
    if len(X) < 10:
        print("\nWarning: Very small dataset detected. Consider collecting more data.")
        print("The model might not perform well with such limited data.")
    
    # Train both classifiers
    print("\nTraining Random Forest Classifier...")
    clf, scaler, rf_accuracy = train_classifier(X, y)
    
    print("\nTraining SVM Classifier...")
    svm, svm_scaler, svm_accuracy = train_svm(X, y)
    
    # Example of how to use both classifiers for a single frame
    print("\nExample predictions:")
    # Get a random frame from the test set
    X_test, _, y_test, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    random_idx = np.random.randint(len(X_test))
    example_frame = X_test[random_idx]
    
    # Random Forest prediction
    rf_prediction, rf_probabilities = predict_stroke(example_frame, clf, scaler)
    print(f"\nRandom Forest Prediction:")
    print(f"Predicted stroke: {rf_prediction}")
    print(f"True stroke: {y_test[random_idx]}")
    print("Class probabilities:", dict(zip(clf.classes_, rf_probabilities)))
    
    # SVM prediction
    svm_prediction, svm_probabilities = predict_stroke_svm(example_frame, svm, svm_scaler)
    print(f"\nSVM Prediction:")
    print(f"Predicted stroke: {svm_prediction}")
    print(f"True stroke: {y_test[random_idx]}")
    print("Class probabilities:", dict(zip(svm.classes_, svm_probabilities)))

if __name__ == "__main__":
    main() 