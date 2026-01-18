# =============================================================================
# Hyperparameter Experiments for Feedforward Neural Network (FNN / MLP)
# Dataset: Breast Cancer Wisconsin (Diagnostic) (WDBC)
# Source: UCI ML Repository via scikit-learn (load_breast_cancer)
# Purpose: Run a controlled hyperparameter sweep and export a results table.
# =============================================================================

import os
import time
import itertools
import warnings

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Data: load + split + scale
# (fixed split to make experiments fair)
# -----------------------------
def load_and_prepare_data(seed: int = 42):
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # 60/20/20 split with stratification
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=seed, stratify=y_train_val
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, X.shape[1]

# -----------------------------
# Model builder
# -----------------------------
def build_model(
    input_dim: int,
    hidden_units: list[int],
    dropout_rate: float,
    use_batchnorm: bool,
    learning_rate: float,
) -> tf.keras.Model:
    model = Sequential(name="FNN_Sweep")

    # Hidden blocks
    for i, units in enumerate(hidden_units, start=1):
        if i == 1:
            model.add(Dense(units, activation="relu", kernel_initializer="he_normal", input_dim=input_dim,
                            name=f"dense_{i}"))
        else:
            model.add(Dense(units, activation="relu", kernel_initializer="he_normal", name=f"dense_{i}"))

        if use_batchnorm:
            model.add(BatchNormalization(name=f"bn_{i}"))

        if dropout_rate and dropout_rate > 0:
            model.add(Dropout(dropout_rate, name=f"dropout_{i}"))

    # Output
    model.add(Dense(1, activation="sigmoid", name="output"))

    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

# -----------------------------
# Single experiment run
# -----------------------------
def run_experiment(
    X_train, y_train, X_val, y_val, X_test, y_test,
    input_dim: int,
    config: dict,
    max_epochs: int = 100,
) -> dict:
    tf.keras.backend.clear_session()

    model = build_model(
        input_dim=input_dim,
        hidden_units=config["hidden_units"],
        dropout_rate=config["dropout"],
        use_batchnorm=config["batchnorm"],
        learning_rate=config["lr"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0),
    ]

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=config["batch_size"],
        verbose=0,
        callbacks=callbacks
    )
    train_time = time.time() - start
    epochs_ran = len(history.history["loss"])

    # Evaluate on test
    y_prob = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)

    # Also keep best val loss as a selection signal
    best_val_loss = float(np.min(history.history["val_loss"]))

    return {
        "architecture": "-".join(map(str, config["hidden_units"])),
        "lr": config["lr"],
        "batch_size": config["batch_size"],
        "dropout": config["dropout"],
        "batchnorm": config["batchnorm"],
        "epochs_ran": epochs_ran,
        "best_val_loss": best_val_loss,
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1": test_f1,
        "train_time_sec": train_time,
        "params": int(model.count_params()),
    }

# -----------------------------
# Main: define sweep space
# -----------------------------
def main():
    print("=" * 70)
    print("FNN Hyperparameter Experiments (Controlled Sweep) - WDBC")
    print("=" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test, input_dim = load_and_prepare_data(SEED)

    # ✅ You can edit these lists to expand/limit experiments
    architectures = [
        [64, 32, 16],
        [128, 64, 32],
        [32],
        [64, 32],
    ]
    learning_rates = [1e-3, 5e-4, 1e-4]
    batch_sizes = [16, 32]
    dropouts = [0.0, 0.2, 0.3]
    batchnorms = [True, False]

    sweep = list(itertools.product(architectures, learning_rates, batch_sizes, dropouts, batchnorms))
    print(f"Total experiments: {len(sweep)}")
    print("-" * 70)

    results = []
    for idx, (arch, lr, bs, dr, bn) in enumerate(sweep, start=1):
        config = {
            "hidden_units": arch,
            "lr": lr,
            "batch_size": bs,
            "dropout": dr,
            "batchnorm": bn,
        }
        print(f"[{idx:03d}/{len(sweep)}] arch={arch}, lr={lr}, bs={bs}, drop={dr}, bn={bn}")
        row = run_experiment(
            X_train, y_train, X_val, y_val, X_test, y_test,
            input_dim=input_dim,
            config=config,
            max_epochs=100
        )
        results.append(row)

    df = pd.DataFrame(results)

    # Select best: maximize F1 then Accuracy, tie-breaker by lower training time
    df_sorted = df.sort_values(
        by=["test_f1", "test_accuracy", "train_time_sec"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    print("\nTop 5 configurations (by Test F1, then Test Accuracy):")
    print(df_sorted.head(5).to_string(index=False))

    # Export
    df_sorted.to_csv("hyperparam_results.csv", index=False)
    with open("hyperparam_results.md", "w", encoding="utf-8") as f:
        f.write(df_sorted.to_markdown(index=False))

    print("\nSaved:")
    print(" - hyperparam_results.csv")
    print(" - hyperparam_results.md")

    best = df_sorted.iloc[0].to_dict()
    print("\nBEST CONFIG:")
    for k, v in best.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    main()
