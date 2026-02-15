"""
Tabular Transformer for Accident Classification
Uses learned embeddings for categorical features and attention mechanisms
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import joblib
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular accident data"""
    
    def __init__(self, categorical_data, numerical_data, targets):
        self.categorical_data = torch.LongTensor(categorical_data)
        self.numerical_data = torch.FloatTensor(numerical_data)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return (
            self.categorical_data[idx],
            self.numerical_data[idx],
            self.targets[idx]
        )


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def split_heads(self, x):
        """Split into multiple heads"""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """Combine multiple heads"""
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        
        return output, attn_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Single transformer encoder block"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights


class TabTransformer(nn.Module):
    """
    Tabular Transformer for accident classification
    
    Architecture:
    1. Categorical features -> Embeddings
    2. Numerical features -> Linear projection
    3. Concatenate embeddings + numerical
    4. Transformer encoder blocks
    5. Classification head
    """
    
    def __init__(
        self,
        categorical_dims: List[int],
        numerical_dim: int,
        num_classes: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.1,
        embedding_dim: int = 16
    ):
        super(TabTransformer, self).__init__()
        
        self.categorical_dims = categorical_dims
        self.numerical_dim = numerical_dim
        self.d_model = d_model
        
        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim)
            for num_categories in categorical_dims
        ])
        
        # Project embeddings to d_model
        self.embedding_projection = nn.Linear(
            len(categorical_dims) * embedding_dim,
            d_model
        )
        
        # Project numerical features to d_model
        self.numerical_projection = nn.Linear(numerical_dim, d_model)
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2 because we concat cat + num
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, categorical_data, numerical_data):
        batch_size = categorical_data.size(0)
        
        # Process categorical features
        embedded_features = []
        for i, embedding_layer in enumerate(self.embeddings):
            embedded = embedding_layer(categorical_data[:, i])
            embedded_features.append(embedded)
        
        # Concatenate all embeddings
        cat_embedded = torch.cat(embedded_features, dim=1)
        cat_projected = self.embedding_projection(cat_embedded)
        cat_projected = cat_projected.unsqueeze(1)  # Add sequence dimension
        
        # Process numerical features
        num_projected = self.numerical_projection(numerical_data)
        num_projected = num_projected.unsqueeze(1)  # Add sequence dimension
        
        # Concatenate categorical and numerical
        x = torch.cat([cat_projected, num_projected], dim=1)  # [batch, 2, d_model]
        
        # Apply transformer blocks
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x)
            attention_weights.append(attn_weights)
        
        # Pool: take both tokens and concatenate
        pooled = x.view(batch_size, -1)  # Flatten
        
        # Classification
        output = self.classifier(pooled)
        
        return output, attention_weights


class AccidentTabTransformer:
    """Wrapper class for training and using TabTransformer"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.categorical_encoders = {}
        self.numerical_scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        
        # Feature definitions - using only available features
        self.categorical_features = ['lum', 'agg', 'int', 'day_of_week']
        self.numerical_features = ['hour', 'num_users']
        
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        print("Loading and preparing data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Prepare features
        X_cat = df[self.categorical_features].copy()
        X_num = df[self.numerical_features].copy()
        
        # Target: collision type
        if 'col' in df.columns:
            y = df['col'].values
        else:
            raise ValueError("Target column 'col' not found")
        
        # Encode categorical features
        categorical_dims = []
        for col in self.categorical_features:
            le = LabelEncoder()
            X_cat[col] = le.fit_transform(X_cat[col].astype(str))
            self.categorical_encoders[col] = le
            categorical_dims.append(len(le.classes_))
        
        # Scale numerical features
        X_num_scaled = self.numerical_scaler.fit_transform(X_num)
        
        # Encode target
        y_encoded = self.target_encoder.fit_transform(y)
        
        print(f"✓ Data prepared:")
        print(f"  Samples: {len(df)}")
        print(f"  Categorical features: {len(self.categorical_features)}")
        print(f"  Numerical features: {len(self.numerical_features)}")
        print(f"  Classes: {len(self.target_encoder.classes_)}")
        
        return X_cat.values, X_num_scaled, y_encoded, categorical_dims
    
    def train(
        self,
        X_cat, X_num, y, categorical_dims,
        epochs=50,
        batch_size=128,
        learning_rate=0.001,
        test_size=0.2
    ):
        """Train the TabTransformer model"""
        print(f"\nTraining TabTransformer on {self.device}...")
        
        # Split data
        X_cat_train, X_cat_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
            X_cat, X_num, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create datasets
        train_dataset = TabularDataset(X_cat_train, X_num_train, y_train)
        test_dataset = TabularDataset(X_cat_test, X_num_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        num_classes = len(np.unique(y))
        self.model = TabTransformer(
            categorical_dims=categorical_dims,
            numerical_dim=len(self.numerical_features),
            num_classes=num_classes,
            d_model=64,
            num_heads=4,
            num_layers=3,
            d_ff=128,
            dropout=0.1,
            embedding_dim=16
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_test_acc = 0
        train_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for cat_data, num_data, targets in train_loader:
                cat_data = cat_data.to(self.device)
                num_data = num_data.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(cat_data, num_data)
                loss = criterion(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            train_losses.append(train_loss)
            
            # Testing
            self.model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0
            
            with torch.no_grad():
                for cat_data, num_data, targets in test_loader:
                    cat_data = cat_data.to(self.device)
                    num_data = num_data.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs, _ = self.model(cat_data, num_data)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += targets.size(0)
                    test_correct += (predicted == targets).sum().item()
            
            test_loss /= len(test_loader)
            test_acc = 100 * test_correct / test_total
            test_accuracies.append(test_acc)
            
            # Learning rate scheduling
            scheduler.step(test_loss)
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                # Save best model with accuracy
                self.save_model('models/tab_transformer_best.pth', test_accuracy=best_test_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        print(f"\n✓ Training complete! Best test accuracy: {best_test_acc:.2f}%")
        
        # Final evaluation
        self.evaluate(test_loader)
        
        return train_losses, test_accuracies, best_test_acc
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for cat_data, num_data, targets in test_loader:
                cat_data = cat_data.to(self.device)
                num_data = num_data.to(self.device)
                
                outputs, _ = self.model(cat_data, num_data)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            all_targets,
            all_preds,
            target_names=[str(c) for c in self.target_encoder.classes_]
        ))
    
    def predict(self, categorical_data, numerical_data):
        """
        Make predictions
        
        Args:
            categorical_data: Dict with categorical feature values
            numerical_data: Dict with numerical feature values
        
        Returns:
            Predicted class and probabilities
        """
        self.model.eval()
        
        # Encode categorical features
        cat_encoded = []
        for feat in self.categorical_features:
            value = categorical_data[feat]
            encoded = self.categorical_encoders[feat].transform([str(value)])[0]
            cat_encoded.append(encoded)
        
        # Scale numerical features
        num_values = [numerical_data[feat] for feat in self.numerical_features]
        num_scaled = self.numerical_scaler.transform([num_values])[0]
        
        # Convert to tensors
        cat_tensor = torch.LongTensor([cat_encoded]).to(self.device)
        num_tensor = torch.FloatTensor([num_scaled]).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs, attention_weights = self.model(cat_tensor, num_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Decode prediction
        predicted_label = self.target_encoder.inverse_transform([predicted_class])[0]
        probs_dict = {
            self.target_encoder.inverse_transform([i])[0]: probabilities[0, i].item()
            for i in range(len(self.target_encoder.classes_))
        }
        
        return predicted_label, probs_dict, attention_weights
    
    def save_model(self, path: str, test_accuracy: float = None):
        """Save model and preprocessors"""
        # Ensure directory exists
        from pathlib import Path
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'categorical_encoders': self.categorical_encoders,
            'numerical_scaler': self.numerical_scaler,
            'target_encoder': self.target_encoder,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'test_accuracy': test_accuracy
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str, categorical_dims: List[int], num_classes: int):
        """Load model and preprocessors"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model = TabTransformer(
            categorical_dims=categorical_dims,
            numerical_dim=len(self.numerical_features),
            num_classes=num_classes
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.categorical_encoders = checkpoint['categorical_encoders']
        self.numerical_scaler = checkpoint['numerical_scaler']
        self.target_encoder = checkpoint['target_encoder']
        self.categorical_features = checkpoint['categorical_features']
        self.numerical_features = checkpoint['numerical_features']
        
        print(f"✓ Model loaded from {path}")


def main():
    """Example usage"""
    # Initialize TabTransformer
    tab_transformer = AccidentTabTransformer(data_path='data/model_ready.csv')
    
    # Load and prepare data
    X_cat, X_num, y, categorical_dims = tab_transformer.load_and_prepare_data()
    
    # Train model
    train_losses, test_accuracies, best_accuracy = tab_transformer.train(
        X_cat, X_num, y, categorical_dims,
        epochs=50,
        batch_size=128,
        learning_rate=0.001
    )
    
    # Save final model with accuracy
    tab_transformer.save_model('models/tab_transformer_final.pth', test_accuracy=best_accuracy)
    
    # Example prediction
    categorical_data = {
        'lum': 1,  # Daylight
        'agg': 1,  # Urban
        'int': 1,  # No intersection
        'day_of_week': 0  # Monday
    }
    
    numerical_data = {
        'hour': 14,  # 2 PM
        'num_users': 2
    }
    
    predicted_class, probabilities, attention = tab_transformer.predict(
        categorical_data,
        numerical_data
    )
    
    print(f"\nPrediction: {predicted_class}")
    print("Probabilities:")
    for cls, prob in probabilities.items():
        print(f"  {cls}: {prob:.4f}")
    
    print("\n✓ TabTransformer training and prediction complete!")


if __name__ == "__main__":
    main()
