"""Train the academic CNN-RNN-LSTM stroke-prediction model."""

import logging
import os
import pickle
from pathlib import Path

# Keep TensorFlow and joblib console output clean for dissertation screenshots.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from models.model import build_hybrid_model


LOGGER = logging.getLogger("hybrid_model_training")
PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DATASET_PATH = PROJECT_ROOT / "dataset" / "processed_dataset.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_hybrid_model.keras"
HISTORY_PATH = PROJECT_ROOT / "models" / "history.pkl"
MODEL_SUMMARY_PATH = PROJECT_ROOT / "reports" / "model_summary.txt"
TRAINING_METRICS_PATH = PROJECT_ROOT / "reports" / "training_metrics.txt"
PLOTS_DIR = PROJECT_ROOT / "reports" / "plots"
TARGET_COLUMN = "stroke"
SPLIT_COLUMN = "data_split"
RANDOM_STATE = 42
EPOCHS = 30
BATCH_SIZE = 32


def configure_training() -> None:
    """Configure reproducible seeds and readable logs."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    tf.keras.utils.set_random_seed(RANDOM_STATE)


def load_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Phase 3 data and reshape it for Conv1D/RNN/LSTM layers."""
    if not PROCESSED_DATASET_PATH.is_file():
        raise FileNotFoundError(
            "processed_dataset.csv was not found. Complete Phase 3 first."
        )

    dataframe = pd.read_csv(PROCESSED_DATASET_PATH)
    required_columns = {TARGET_COLUMN, SPLIT_COLUMN}
    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            "The processed dataset must contain stroke and data_split columns."
        )

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in {TARGET_COLUMN, SPLIT_COLUMN}
    ]
    train_data = dataframe[dataframe[SPLIT_COLUMN] == "train"]
    validation_data = dataframe[dataframe[SPLIT_COLUMN] == "test"]
    if train_data.empty or validation_data.empty:
        raise ValueError("Both train and test rows are required for training.")

    x_train = train_data[feature_columns].to_numpy(dtype=np.float32)
    y_train = train_data[TARGET_COLUMN].to_numpy(dtype=np.float32)
    x_validation = validation_data[feature_columns].to_numpy(dtype=np.float32)
    y_validation = validation_data[TARGET_COLUMN].to_numpy(dtype=np.float32)

    # Conv1D, RNN, and LSTM expect a 3D input: samples, steps, channels.
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_validation = x_validation.reshape(
        x_validation.shape[0],
        x_validation.shape[1],
        1,
    )

    LOGGER.info("Training samples: %d", len(x_train))
    LOGGER.info("Validation samples: %d", len(x_validation))
    LOGGER.info("Features per sample: %d", len(feature_columns))
    return x_train, y_train, x_validation, y_validation


def create_callbacks() -> list[keras.callbacks.Callback]:
    """Create the three callbacks required by the dissertation."""
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Stop when validation loss no longer improves and restore the best weights.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    # Save only the model with the lowest validation loss.
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # Lower the learning rate when validation progress becomes slow.
    reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1,
    )

    return [early_stopping, model_checkpoint, reduce_learning_rate]


def save_model_summary(model: keras.Model) -> None:
    """Print and save the architecture and trainable parameter count."""
    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    trainable_parameters = int(
        sum(np.prod(variable.shape) for variable in model.trainable_weights)
    )
    summary_text = "\n".join(summary_lines)

    print("\nMODEL SUMMARY")
    print("=" * 80)
    print(summary_text)
    print(f"\nTrainable Parameters: {trainable_parameters:,}")

    MODEL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_SUMMARY_PATH.write_text(
        f"{summary_text}\n\nTrainable Parameters: {trainable_parameters:,}\n",
        encoding="utf-8",
    )


def save_history(history: keras.callbacks.History) -> None:
    """Serialize the Keras training history for later academic analysis."""
    with HISTORY_PATH.open("wb") as history_file:
        pickle.dump(history.history, history_file)


def save_training_plots(history: keras.callbacks.History) -> None:
    """Save separate accuracy and loss graphs for dissertation screenshots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.history["loss"]) + 1)

    accuracy_figure, accuracy_axis = plt.subplots(figsize=(8, 5))
    accuracy_axis.plot(
        epochs,
        history.history["accuracy"],
        label="Training Accuracy",
        color="#1f5a91",
    )
    accuracy_axis.plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation Accuracy",
        color="#dc3545",
    )
    accuracy_axis.set_title("Hybrid Model Accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.legend()
    accuracy_axis.grid(alpha=0.3)
    accuracy_figure.tight_layout()
    accuracy_figure.savefig(
        PLOTS_DIR / "training_accuracy.png",
        dpi=150,
    )
    plt.close(accuracy_figure)

    loss_figure, loss_axis = plt.subplots(figsize=(8, 5))
    loss_axis.plot(
        epochs,
        history.history["loss"],
        label="Training Loss",
        color="#1f5a91",
    )
    loss_axis.plot(
        epochs,
        history.history["val_loss"],
        label="Validation Loss",
        color="#dc3545",
    )
    loss_axis.set_title("Hybrid Model Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Binary Cross-Entropy Loss")
    loss_axis.legend()
    loss_axis.grid(alpha=0.3)
    loss_figure.tight_layout()
    loss_figure.savefig(PLOTS_DIR / "training_loss.png", dpi=150)
    plt.close(loss_figure)


def print_and_save_metrics(history: keras.callbacks.History) -> None:
    """Print and save metrics from the best validation-loss epoch."""
    best_epoch_index = int(np.argmin(history.history["val_loss"]))
    final_metrics = {
        "Training Accuracy": history.history["accuracy"][best_epoch_index],
        "Validation Accuracy": history.history["val_accuracy"][best_epoch_index],
        "Training Loss": history.history["loss"][best_epoch_index],
        "Validation Loss": history.history["val_loss"][best_epoch_index],
        "Best Epoch": best_epoch_index + 1,
        "Epochs Completed": len(history.history["loss"]),
    }

    print("\nBEST CHECKPOINT METRICS")
    print("=" * 80)
    for name, value in final_metrics.items():
        if name in {"Best Epoch", "Epochs Completed"}:
            print(f"{name}: {value}")
        else:
            print(f"{name}: {value:.4f}")

    TRAINING_METRICS_PATH.write_text(
        "\n".join(
            f"{name}: "
            f"{value if name in {'Best Epoch', 'Epochs Completed'} else f'{value:.4f}'}"
            for name, value in final_metrics.items()
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Train and save the Phase 5 hybrid deep-learning model."""
    configure_training()
    try:
        x_train, y_train, x_validation, y_validation = load_training_data()
        model = build_hybrid_model(number_of_features=x_train.shape[1])
        save_model_summary(model)

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=create_callbacks(),
            verbose=2,
        )
        save_history(history)
        save_training_plots(history)
        print_and_save_metrics(history)
    except (FileNotFoundError, PermissionError, ValueError, OSError) as error:
        LOGGER.error("Training failed: %s", error)
        return 1

    LOGGER.info("Best model saved: %s", BEST_MODEL_PATH)
    LOGGER.info("Training history saved: %s", HISTORY_PATH)
    LOGGER.info("Phase 5 model training completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
