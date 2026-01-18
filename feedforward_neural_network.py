# =============================================================================
# Feedforward Neural Network Implementation
# Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
# Source: UCI Machine Learning Repository / Scikit-learn
# Author: Academic Implementation
# =============================================================================

# -----------------------------------------------------------------------------
# Section 0: Install Required Libraries (Run Once)
# -----------------------------------------------------------------------------
# Uncomment the following lines to install libraries:
# !pip install tensorflow numpy pandas scikit-learn matplotlib seaborn

# Or run this command in terminal/command prompt:
# pip install -r requirements.txt

# Or run this command directly:
# pip install tensorflow numpy pandas scikit-learn matplotlib seaborn

# -----------------------------------------------------------------------------
# Section 1: Import Required Libraries
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import time

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("Feedforward Neural Network - Breast Cancer Classification")
print("=" * 60)

# -----------------------------------------------------------------------------
# Section 2: Data Loading and Description
# -----------------------------------------------------------------------------
print("\n[1] Loading Dataset...")

# Load the Breast Cancer Wisconsin Dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# Create DataFrame for better visualization
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"\nDataset Shape: {X.shape}")
print(f"Number of Samples: {X.shape[0]}")
print(f"Number of Features: {X.shape[1]}")
print(f"Target Classes: {target_names}")
print(f"\nClass Distribution:")
print(f"  - Malignant (0): {np.sum(y == 0)} samples")
print(f"  - Benign (1): {np.sum(y == 1)} samples")

# Display dataset statistics
print("\n[2] Dataset Statistics:")
print(df.describe().round(2))

# -----------------------------------------------------------------------------
# Section 3: Data Preprocessing
# -----------------------------------------------------------------------------
print("\n[3] Data Preprocessing...")

# 3.1 Check for missing values
print(f"\nMissing Values: {df.isnull().sum().sum()}")

# 3.2 Train-Validation-Test Split (60-20-20)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

print(f"\nData Split:")
print(f"  - Training Set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  - Validation Set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"  - Test Set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# 3.3 Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling: StandardScaler applied (mean=0, std=1)")

# -----------------------------------------------------------------------------
# Section 4: Model Architecture Definition
# -----------------------------------------------------------------------------
print("\n[4] Building Feedforward Neural Network Model...")

def build_feedforward_model(input_dim):
    """
    Build a Feedforward Neural Network for binary classification.
    
    Architecture:
    - Input Layer: 30 features
    - Hidden Layer 1: 64 neurons, ReLU activation, BatchNorm, Dropout(0.3)
    - Hidden Layer 2: 32 neurons, ReLU activation, BatchNorm, Dropout(0.3)
    - Hidden Layer 3: 16 neurons, ReLU activation
    - Output Layer: 1 neuron, Sigmoid activation
    """
    model = Sequential([
        # Input Layer and First Hidden Layer
        Dense(64, input_dim=input_dim, activation='relu', 
              kernel_initializer='he_normal', name='hidden_layer_1'),
        BatchNormalization(name='batch_norm_1'),
        Dropout(0.3, name='dropout_1'),
        
        # Second Hidden Layer
        Dense(32, activation='relu', 
              kernel_initializer='he_normal', name='hidden_layer_2'),
        BatchNormalization(name='batch_norm_2'),
        Dropout(0.3, name='dropout_2'),
        
        # Third Hidden Layer
        Dense(16, activation='relu', 
              kernel_initializer='he_normal', name='hidden_layer_3'),
        
        # Output Layer
        Dense(1, activation='sigmoid', name='output_layer')
    ])
    
    return model

# Build the model
model = build_feedforward_model(X_train_scaled.shape[1])

# Display model summary
print("\nModel Architecture:")
model.summary()

# -----------------------------------------------------------------------------
# Section 5: Model Compilation
# -----------------------------------------------------------------------------
print("\n[5] Compiling Model...")

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100

# Compile the model
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"\nHyperparameters:")
print(f"  - Optimizer: Adam")
print(f"  - Learning Rate: {LEARNING_RATE}")
print(f"  - Loss Function: Binary Cross-Entropy")
print(f"  - Batch Size: {BATCH_SIZE}")
print(f"  - Epochs: {EPOCHS}")

# -----------------------------------------------------------------------------
# Section 6: Callbacks Definition
# -----------------------------------------------------------------------------
print("\n[6] Setting up Callbacks...")

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Reduce Learning Rate on Plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]
print("  - EarlyStopping: patience=15, monitor=val_loss")
print("  - ReduceLROnPlateau: factor=0.5, patience=5")

# -----------------------------------------------------------------------------
# Section 7: Model Training
# -----------------------------------------------------------------------------
print("\n[7] Training Model...")
print("-" * 60)

start_time = time.time()

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print("-" * 60)
print(f"\nTraining completed in {training_time:.2f} seconds")
print(f"Total epochs run: {len(history.history['loss'])}")

# -----------------------------------------------------------------------------
# Section 8: Model Evaluation
# -----------------------------------------------------------------------------
print("\n[8] Evaluating Model...")

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Set Performance:")
print(f"  - Test Loss: {test_loss:.4f}")
print(f"  - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Make predictions
y_pred_prob = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Calculate additional metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nClassification Metrics:")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall: {recall:.4f}")
print(f"  - F1-Score: {f1:.4f}")

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# -----------------------------------------------------------------------------
# Section 9: Visualization
# -----------------------------------------------------------------------------
print("\n[9] Generating Visualizations...")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Feedforward Neural Network - Training Results', fontsize=14, fontweight='bold')

# Plot 1: Training and Validation Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 0].set_title('Model Accuracy Over Epochs', fontsize=12)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Training and Validation Loss
axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 1].set_title('Model Loss Over Epochs', fontsize=12)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend(loc='upper right')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=target_names, yticklabels=target_names)
axes[1, 0].set_title('Confusion Matrix', fontsize=12)
axes[1, 0].set_xlabel('Predicted Label')
axes[1, 0].set_ylabel('True Label')

# Plot 4: Performance Metrics Bar Chart
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [test_accuracy, precision, recall, f1]
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
bars = axes[1, 1].bar(metrics_names, metrics_values, color=colors, edgecolor='black')
axes[1, 1].set_title('Performance Metrics Comparison', fontsize=12)
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_ylim(0, 1.1)
for bar, value in zip(bars, metrics_values):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()
print("  - Saved: training_results.png")

# -----------------------------------------------------------------------------
# Section 10: Feature Importance Visualization
# -----------------------------------------------------------------------------
# Create correlation heatmap for top features
plt.figure(figsize=(12, 8))
top_features = df[feature_names].corrwith(df['target']).abs().sort_values(ascending=False)[:10]
top_feature_names = top_features.index.tolist()
correlation_matrix = df[top_feature_names + ['target']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap - Top 10 Features with Target', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
plt.show()
print("  - Saved: feature_correlation.png")

# -----------------------------------------------------------------------------
# Section 11: Results Summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"""
Model: Feedforward Neural Network
Dataset: Breast Cancer Wisconsin (Diagnostic)
Task: Binary Classification (Malignant vs Benign)

Training Configuration:
  - Architecture: 30 → 64 → 32 → 16 → 1
  - Activation: ReLU (hidden), Sigmoid (output)
  - Optimizer: Adam (lr={LEARNING_RATE})
  - Regularization: Dropout(0.3), BatchNormalization
  - Epochs: {len(history.history['loss'])} (with Early Stopping)

Final Performance:
  - Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
  - Test Loss:      {test_loss:.4f}
  - Precision:      {precision:.4f}
  - Recall:         {recall:.4f}
  - F1-Score:       {f1:.4f}

Training Time: {training_time:.2f} seconds
""")
print("=" * 60)

# Save model
model.save('feedforward_model.h5')
print("\nModel saved as: feedforward_model.h5")
print("\n[COMPLETED] Feedforward Neural Network Training and Evaluation")
