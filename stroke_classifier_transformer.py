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
from transformers import AutoModel, AutoTokenizer
warnings.filterwarnings('ignore')

class KeypointDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.label_encoder = LabelEncoder()
        self.y = torch.LongTensor(self.label_encoder.fit_transform(y))
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment:
            noise = torch.randn_like(x) * 0.1
            x = x + noise
            if torch.rand(1) > 0.5:
                x[:, 0] = -x[:, 0]
        
        return x, y

class StrokeTransformer(nn.Module):
    def __init__(self, num_classes, model_name='bert-base-uncased'):
        super(StrokeTransformer, self).__init__()
        
        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Project keypoints to transformer input dimension
        self.keypoint_projection = nn.Linear(2, self.transformer.config.hidden_size)
        
        # Position embeddings for keypoints
        self.position_embeddings = nn.Parameter(
            torch.randn(17, self.transformer.config.hidden_size)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, 17, 2)
        batch_size = x.size(0)
        
        # Project keypoints to transformer dimension
        x = self.keypoint_projection(x)  # (batch_size, 17, hidden_size)
        
        # Add position embeddings
        x = x + self.position_embeddings.unsqueeze(0)
        
        # Get transformer outputs
        transformer_outputs = self.transformer(inputs_embeds=x)[0]
        
        # Use [CLS] token output for classification
        cls_output = transformer_outputs[:, 0, :]
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits

def prepare_data(csv_path):
    """Prepare the data for training by grouping keypoints by frame."""
    df = pd.read_csv(csv_path)
    grouped = df.groupby(['stroke', 'frame'])
    
    X = []
    y = []
    
    for (stroke, frame), group in grouped:
        group = group.sort_index()
        coords = group[['x', 'y']].values
        
        if coords.shape == (17, 2):
            X.append(coords)
            y.append(stroke)
    
    return np.array(X), np.array(y)

def train_transformer(X, y, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train a transformer classifier on the keypoint data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 2)).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 2)).reshape(X_test.shape)
    
    train_dataset = KeypointDataset(X_train_scaled, y_train, augment=True)
    test_dataset = KeypointDataset(X_test_scaled, y_test, augment=False)
    
    label_encoder = train_dataset.label_encoder
    
    train_loader = DataLoader(train_dataset, batch_size=min(16, len(X_train)), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=min(16, len(X_test)), shuffle=False)
    
    model = StrokeTransformer(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    num_epochs = 100
    best_accuracy = 0
    
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
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
            
            # Evaluate on test set
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
            
            all_preds = label_encoder.inverse_transform(all_preds)
            all_labels = label_encoder.inverse_transform(all_labels)
            
            test_accuracy = accuracy_score(all_labels, all_preds)
            print(f"Test Accuracy: {test_accuracy:.4f}")
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                # Save the best model
                torch.save(model.state_dict(), 'models/stroke_transformer.pth')
                joblib.dump(scaler, 'models/stroke_transformer_scaler.joblib')
                joblib.dump(label_encoder, 'models/stroke_transformer_label_encoder.joblib')
    
    print("\nFinal Classification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    
    return model, scaler, label_encoder, best_accuracy

def predict_stroke_transformer(frame_keypoints, model, scaler, label_encoder, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Predict stroke type for a single frame's keypoints using transformer."""
    if len(frame_keypoints.shape) == 1:
        frame_keypoints = frame_keypoints.reshape(17, 2)
    
    scaled_keypoints = scaler.transform(frame_keypoints.reshape(-1, 2)).reshape(17, 2)
    input_tensor = torch.FloatTensor(scaled_keypoints).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1)
    
    prediction = label_encoder.inverse_transform([prediction.item()])[0]
    probabilities = probabilities[0].cpu().numpy()
    
    return prediction, probabilities

def print_model_architecture(model, input_shape=(1, 17, 2)):
    """Print the model architecture in a readable format."""
    print("\nModel Architecture:")
    print("=" * 50)
    
    dummy_input = torch.randn(input_shape)
    
    print(f"Input shape: {input_shape}")
    print("\nLayer Structure:")
    print("-" * 50)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print("-" * 50)
    
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

    X, y = prepare_data(KEYPOINTS_CSV_FILE)
    print("Shape of X:", X.shape)
    print("Number of unique strokes:", len(np.unique(y)))
    print("Unique strokes:", np.unique(y))
    
    if len(X) < 10:
        print("\nWarning: Very small dataset detected. Consider collecting more data.")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = StrokeTransformer(num_classes=len(np.unique(y))).to(device)
    print_model_architecture(model)
    
    model, scaler, label_encoder, test_accuracy = train_transformer(X, y, num_classes=len(np.unique(y)), device=device)
    
    print("\nExample prediction:")
    X_test, _, y_test, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    random_idx = np.random.randint(len(X_test))
    example_frame = X_test[random_idx]
    
    prediction, probabilities = predict_stroke_transformer(example_frame, model, scaler, label_encoder, device)
    print(f"Predicted stroke: {prediction}")
    print(f"True stroke: {y_test[random_idx]}")
    print("Class probabilities:", dict(zip(label_encoder.classes_, probabilities)))

if __name__ == "__main__":
    main() 