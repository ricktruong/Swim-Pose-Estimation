import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
# Filter out all warnings
warnings.filterwarnings('ignore')

class KeypointDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        # Convert string labels to integers
        self.label_encoder = LabelEncoder()
        self.y = torch.LongTensor(self.label_encoder.fit_transform(y))
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment:
            # Add small random noise to keypoints
            noise = torch.randn_like(x) * 0.1
            x = x + noise
            
            # Random horizontal flip (50% chance)
            if torch.rand(1) > 0.5:
                x[:, 0] = -x[:, 0]  # Flip x coordinates
        
        return x, y

class StrokeMLP(nn.Module):
    def __init__(self, num_classes):
        super(StrokeMLP, self).__init__()
        # Input shape: (batch_size, 17, 2) -> flattened to (batch_size, 34)
        
        # Flatten the input
        self.flatten = nn.Flatten()
        
        # MLP layers
        self.mlp = nn.Sequential(
            # First fully connected layer
            nn.Linear(34, 128),  # 17 keypoints * 2 coordinates = 34 input features
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            # Second fully connected layer
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            # Third fully connected layer
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            # Fourth fully connected layer
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            # Fifth fully connected layer
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            # Sixth fully connected layer
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            # Seventh fully connected layer
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            # Eighth fully connected layer
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            
            # Ninth fully connected layer
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        # Flatten the input from (batch_size, 17, 2) to (batch_size, 34)
        x = self.flatten(x)
        return self.mlp(x)

def prepare_data(csv_path):
    """Prepare the data for training by grouping keypoints by frame."""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Group by stroke and frame
    grouped = df.groupby(['stroke', 'frame'])
    
    # Create features by keeping x,y coordinates as 17x2 arrays
    X = []
    y = []
    
    for (stroke, frame), group in grouped:
        # Sort by index to ensure consistent order of keypoints
        group = group.sort_index()
        
        # Extract x,y coordinates as 17x2 array
        coords = group[['x', 'y']].values
        
        # Only include frames where we have all keypoints (no zeros)
        # and ensure the shape is exactly (17, 2)
        if coords.shape == (17, 2):
            X.append(coords)
            y.append(stroke)
    
    return np.array(X), np.array(y)

def train_mlp(X, y, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train an MLP classifier on the keypoint data."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 2)).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 2)).reshape(X_test.shape)
    
    # Create datasets and dataloaders
    train_dataset = KeypointDataset(X_train_scaled, y_train, augment=True)
    test_dataset = KeypointDataset(X_test_scaled, y_test, augment=False)
    
    # Get the label encoder from the training dataset
    label_encoder = train_dataset.label_encoder
    
    train_loader = DataLoader(train_dataset, batch_size=min(32, len(X_train)), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=min(32, len(X_test)), shuffle=False)
    
    # Initialize model
    model = StrokeMLP(num_classes).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluate the model
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    # Convert numeric predictions back to original labels
    all_preds = label_encoder.inverse_transform(all_preds)
    all_labels = label_encoder.inverse_transform(all_labels)
    
    # Calculate and print test accuracy
    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    
    # Save the model, scaler, and label encoder
    torch.save(model.state_dict(), 'models/stroke_mlp.pth')
    joblib.dump(scaler, 'models/stroke_mlp_scaler.joblib')
    joblib.dump(label_encoder, 'models/stroke_mlp_label_encoder.joblib')
    
    return model, scaler, label_encoder, test_accuracy

def predict_stroke_mlp(frame_keypoints, model, scaler, label_encoder, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Predict stroke type for a single frame's keypoints using MLP."""
    # Ensure the input is in the correct shape (17x2)
    if len(frame_keypoints.shape) == 1:
        frame_keypoints = frame_keypoints.reshape(17, 2)
    
    # Scale the keypoints
    scaled_keypoints = scaler.transform(frame_keypoints.reshape(-1, 2)).reshape(17, 2)
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.FloatTensor(scaled_keypoints).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1)
    
    # Convert numeric prediction back to original label
    prediction = label_encoder.inverse_transform([prediction.item()])[0]
    probabilities = probabilities[0].cpu().numpy()
    
    return prediction, probabilities

def print_model_architecture(model, input_shape=(1, 17, 2)):
    """Print the model architecture in a readable format."""
    print("\nModel Architecture:")
    print("=" * 50)
    
    # Create a dummy input tensor
    dummy_input = torch.randn(input_shape)
    
    # Print model summary
    print(f"Input shape: {input_shape}")
    print("\nLayer Structure:")
    print("-" * 50)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print("-" * 50)
    
    # Print each layer's information
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            print(f"\n{name}:")
            for sub_name, sub_module in module.named_children():
                print(f"  {sub_name}: {sub_module}")
        else:
            print(f"\n{name}: {module}")
    
    print("=" * 50)

def main():
    KEYPOINTS_CSV_FILE = 'data/keypoints/keypoints.csv'

    # Prepare the data
    X, y = prepare_data(KEYPOINTS_CSV_FILE)
    print("Shape of X:", X.shape)  # Should be (n_samples, 17, 2)
    print("Number of unique strokes:", len(np.unique(y)))
    print("Unique strokes:", np.unique(y))
    
    if len(X) < 10:
        print("\nWarning: Very small dataset detected. Consider collecting more data.")
        print("The model might not perform well with such limited data.")
    
    # Initialize and print model architecture
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = StrokeMLP(num_classes=len(np.unique(y))).to(device)
    print_model_architecture(model)
    
    # Train the MLP classifier
    model, scaler, label_encoder, test_accuracy = train_mlp(X, y, num_classes=len(np.unique(y)), device=device)
    
    # Example of how to use the classifier for a single frame
    print("\nExample prediction:")
    # Get a random frame from the test set
    X_test, _, y_test, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    random_idx = np.random.randint(len(X_test))
    example_frame = X_test[random_idx]
    
    prediction, probabilities = predict_stroke_mlp(example_frame, model, scaler, label_encoder, device)
    print(f"Predicted stroke: {prediction}")
    print(f"True stroke: {y_test[random_idx]}")
    print("Class probabilities:", dict(zip(label_encoder.classes_, probabilities)))

if __name__ == "__main__":
    main() 