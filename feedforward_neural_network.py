# =============================================================================
# Feedforward Neural Network Implementation
# Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
# Source: UCI Machine Learning Repository / Scikit-learn
# =============================================================================

"""
This script implements a Feedforward Neural Network (FNN) for binary 
classification on the Breast Cancer Wisconsin dataset.

Dataset Information:
    - 569 samples
    - 30 features (computed from digitized images of breast mass)
    - 2 classes: Malignant (0), Benign (1)
    
Author: Academic Implementation
"""

# =============================================================================
# CONFIGURATION
# =============================================================================
# Hyperparameters (based on experiments from hyperparameter_experiments.py)
CONFIG = {
    'random_seed': 42,
    'test_size': 0.2,
    'val_size': 0.25,  # from remaining after test split
    
    # Model Architecture
    'hidden_layers': [64, 32, 16],
    'dropout_rate': 0.3,
    'activation': 'relu',
    'output_activation': 'sigmoid',
    
    # Training Parameters
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'optimizer': 'adam',
    
    # Callbacks
    'early_stopping_patience': 15,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
}

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

# Scikit-learn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(CONFIG['random_seed'])
tf.random.set_seed(CONFIG['random_seed'])

print("=" * 60)
print("Feedforward Neural Network - Breast Cancer Classification")
print("=" * 60)

# =============================================================================
# SECTION 2: DATA LOADING
# =============================================================================
print("\n[STEP 1] Loading Dataset...")
print("-" * 40)

# Load the Breast Cancer Wisconsin Dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# Create DataFrame for analysis
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Display dataset information
print(f"Dataset Shape:     {X.shape}")
print(f"Total Samples:     {X.shape[0]}")
print(f"Total Features:    {X.shape[1]}")
print(f"Target Classes:    {list(target_names)}")
print(f"\nClass Distribution:")
print(f"  Malignant (0):   {np.sum(y == 0)} samples ({np.sum(y == 0)/len(y)*100:.1f}%)")
print(f"  Benign (1):      {np.sum(y == 1)} samples ({np.sum(y == 1)/len(y)*100:.1f}%)")
print(f"\nMissing Values:    {df.isnull().sum().sum()}")

# =============================================================================
# SECTION 3: DATA PREPROCESSING
# =============================================================================
print("\n[STEP 2] Preprocessing Data...")
print("-" * 40)

# Split data: 60% train, 20% validation, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, 
    test_size=CONFIG['test_size'], 
    random_state=CONFIG['random_seed'], 
    stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, 
    test_size=CONFIG['val_size'], 
    random_state=CONFIG['random_seed'], 
    stratify=y_train_val
)

print(f"Data Split:")
print(f"  Training Set:    {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
print(f"  Validation Set:  {len(X_val)} samples ({len(X_val)/len(X)*100:.0f}%)")
print(f"  Test Set:        {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")

# Feature Scaling with StandardScaler (zero mean, unit variance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\nFeature Scaling:   StandardScaler applied")
print(f"  Mean → 0, Std → 1")

# =============================================================================
# SECTION 4: MODEL ARCHITECTURE
# =============================================================================
print("\n[STEP 3] Building Neural Network Model...")
print("-" * 40)

def build_model(input_dim, config):
    """
    Build a Feedforward Neural Network for binary classification.
    
    Architecture:
        Input (30) → Dense(64) → BatchNorm → Dropout(0.3) →
                     Dense(32) → BatchNorm → Dropout(0.3) →
                     Dense(16) → BatchNorm → Dropout(0.3) →
                     Dense(1, sigmoid)
    
    Parameters:
        input_dim: Number of input features
        config: Configuration dictionary with hyperparameters
    
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential(name='Feedforward_Neural_Network')
    
    # Build hidden layers
    for i, neurons in enumerate(config['hidden_layers']):
        if i == 0:
            # First layer needs input dimension
            model.add(Dense(
                neurons, 
                input_dim=input_dim,
                activation=config['activation'],
                kernel_initializer='he_normal',
                name=f'hidden_layer_{i+1}'
            ))
        else:
            model.add(Dense(
                neurons,
                activation=config['activation'],
                kernel_initializer='he_normal',
                name=f'hidden_layer_{i+1}'
            ))
        
        model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
        model.add(Dropout(config['dropout_rate'], name=f'dropout_{i+1}'))
    
    # Output layer
    model.add(Dense(1, activation=config['output_activation'], name='output'))
    
    return model

# Build the model
model = build_model(X_train_scaled.shape[1], CONFIG)

# Display architecture
print("\nModel Architecture:")
print(f"  Input:  {X_train_scaled.shape[1]} features")
for i, neurons in enumerate(CONFIG['hidden_layers']):
    print(f"  Hidden {i+1}: {neurons} neurons (ReLU + BatchNorm + Dropout)")
print(f"  Output: 1 neuron (Sigmoid)")

print("\n" + "=" * 60)
model.summary()
print("=" * 60)

# =============================================================================
# SECTION 5: MODEL COMPILATION
# =============================================================================
print("\n[STEP 4] Compiling Model...")
print("-" * 40)

# Optimizer
optimizer = Adam(learning_rate=CONFIG['learning_rate'])

# Compile
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"Optimizer:         Adam")
print(f"Learning Rate:     {CONFIG['learning_rate']}")
print(f"Loss Function:     Binary Cross-Entropy")
print(f"Metrics:           Accuracy")

# =============================================================================
# SECTION 6: CALLBACKS
# =============================================================================
print("\n[STEP 5] Setting Up Callbacks...")
print("-" * 40)

# Early Stopping - stops training when validation loss stops improving
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=CONFIG['early_stopping_patience'],
    restore_best_weights=True,
    verbose=1
)

# Reduce LR on Plateau - reduces learning rate when validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=CONFIG['reduce_lr_factor'],
    patience=CONFIG['reduce_lr_patience'],
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]

print(f"EarlyStopping:")
print(f"  Monitor:         val_loss")
print(f"  Patience:        {CONFIG['early_stopping_patience']} epochs")
print(f"  Restore Best:    True")
print(f"\nReduceLROnPlateau:")
print(f"  Monitor:         val_loss")
print(f"  Factor:          {CONFIG['reduce_lr_factor']}")
print(f"  Patience:        {CONFIG['reduce_lr_patience']} epochs")

# =============================================================================
# SECTION 7: MODEL TRAINING
# =============================================================================
print("\n[STEP 6] Training Model...")
print("=" * 60)

start_time = time.time()

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time

print("=" * 60)
print(f"\nTraining Complete!")
print(f"  Total Epochs:    {len(history.history['loss'])}")
print(f"  Training Time:   {training_time:.2f} seconds")
print(f"  Final Train Acc: {history.history['accuracy'][-1]:.4f}")
print(f"  Final Val Acc:   {history.history['val_accuracy'][-1]:.4f}")

# =============================================================================
# SECTION 8: MODEL EVALUATION
# =============================================================================
print("\n[STEP 7] Evaluating Model on Test Set...")
print("-" * 40)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

# Generate predictions
y_pred_prob = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Calculate metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nTest Set Performance:")
print(f"  Loss:       {test_loss:.4f}")
print(f"  Accuracy:   {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Precision:  {precision:.4f}")
print(f"  Recall:     {recall:.4f}")
print(f"  F1-Score:   {f1:.4f}")

print("\n" + "-" * 40)
print("Classification Report:")
print("-" * 40)
print(classification_report(y_test, y_pred, target_names=target_names))

# =============================================================================
# SECTION 9: VISUALIZATION
# =============================================================================
print("[STEP 8] Generating Visualizations...")
print("-" * 40)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Feedforward Neural Network - Training Results', fontsize=14, fontweight='bold')

# Plot 1: Training and Validation Accuracy
ax = axes[0, 0]
ax.plot(history.history['accuracy'], label='Training', linewidth=2, color='#3498db')
ax.plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='#e74c3c')
ax.set_title('Model Accuracy Over Epochs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Plot 2: Training and Validation Loss
ax = axes[0, 1]
ax.plot(history.history['loss'], label='Training', linewidth=2, color='#3498db')
ax.plot(history.history['val_loss'], label='Validation', linewidth=2, color='#e74c3c')
ax.set_title('Model Loss Over Epochs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Plot 3: Confusion Matrix
ax = axes[1, 0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=target_names, yticklabels=target_names,
            annot_kws={'size': 14})
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')

# Plot 4: Performance Metrics
ax = axes[1, 1]
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [test_accuracy, precision, recall, f1]
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor='black')
ax.set_title('Performance Metrics')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
for bar, value in zip(bars, metrics_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{value:.3f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: training_results.png")

# =============================================================================
# SECTION 10: FEATURE CORRELATION
# =============================================================================
# Find top correlated features with target
plt.figure(figsize=(12, 8))
correlations = df[feature_names].corrwith(df['target']).abs().sort_values(ascending=False)
top_features = correlations[:10].index.tolist()
correlation_matrix = df[top_features + ['target']].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
            annot_kws={'size': 9})
plt.title('Feature Correlation Heatmap (Top 10 Features)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: feature_correlation.png")

# =============================================================================
# SECTION 11: RESULTS SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

summary = f"""
┌─────────────────────────────────────────────────────────────┐
│                    MODEL INFORMATION                        │
├─────────────────────────────────────────────────────────────┤
│  Model Type:      Feedforward Neural Network                │
│  Dataset:         Breast Cancer Wisconsin (Diagnostic)      │
│  Task:            Binary Classification                     │
├─────────────────────────────────────────────────────────────┤
│                    ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────┤
│  Input Layer:     30 features                               │
│  Hidden Layers:   64 → 32 → 16 neurons                      │
│  Activation:      ReLU (hidden), Sigmoid (output)           │
│  Regularization:  BatchNorm + Dropout(0.3)                  │
├─────────────────────────────────────────────────────────────┤
│                    TRAINING CONFIG                          │
├─────────────────────────────────────────────────────────────┤
│  Optimizer:       Adam (lr={CONFIG['learning_rate']})                      │
│  Batch Size:      {CONFIG['batch_size']}                                       │
│  Epochs:          {len(history.history['loss'])} (with Early Stopping)               │
│  Training Time:   {training_time:.2f} seconds                           │
├─────────────────────────────────────────────────────────────┤
│                    PERFORMANCE                              │
├─────────────────────────────────────────────────────────────┤
│  Test Accuracy:   {test_accuracy:.4f} ({test_accuracy*100:.2f}%)                        │
│  Test Loss:       {test_loss:.4f}                                 │
│  Precision:       {precision:.4f}                                 │
│  Recall:          {recall:.4f}                                 │
│  F1-Score:        {f1:.4f}                                 │
└─────────────────────────────────────────────────────────────┘
"""
print(summary)

# =============================================================================
# SECTION 12: SAVE MODEL
# =============================================================================
model.save('feedforward_model.h5')
print("Model saved: feedforward_model.h5")
print("\n[COMPLETED] Feedforward Neural Network Training and Evaluation")
print("=" * 60)
