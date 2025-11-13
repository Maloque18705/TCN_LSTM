"""
Simple prediction visualization script based on trained LSTM model.

Usage:
    python visualize_predictions_lstm.py outputs/lstm/5p/run_LSTM_Model_...
    python visualize_predictions_lstm.py outputs/lstm/5p/run_LSTM_Model_... --num-samples 20
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from Data.dataloader import DataLoader, DataProcess
from Data import config
from Model.lstm import LSTM_Model


def parse_args():
    p = argparse.ArgumentParser(description="Visualize predictions from trained LSTM model")
    p.add_argument("run_dir", type=str, help="Path to run directory (e.g., outputs/lstm/5p/run_XXX)")
    p.add_argument("--num-samples", type=int, default=10, help="Number of samples to visualize")
    p.add_argument("--save", action="store_true", help="Save plots instead of displaying")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        sys.exit(1)

    print(f"📂 Using run directory: {run_dir}")

    # Load model
    model_path = run_dir / "model_saved.keras"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    print(f"📦 Loading model from: {model_path}")
    custom_objects = {'LSTM_Model': LSTM_Model}

    try:
        model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
        print(f"✅ Model loaded successfully with custom_objects!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Load scaler
    scaler_path = run_dir / "scaler_values.npy"
    if not scaler_path.exists():
        print(f"❌ Scaler not found: {scaler_path}")
        sys.exit(1)

    min_scaler, max_scaler = np.load(scaler_path)
    denom = max_scaler - min_scaler if max_scaler != min_scaler else 1
    print(f"📊 Scaler loaded: min={min_scaler:.6f}, max={max_scaler:.6f}")

    # Load data
    print(f"📂 Loading data from: {config.FOLDER_PATH}")
    dl = DataLoader(folder_path=config.FOLDER_PATH)
    final_array = dl.read_data()
    dp = DataProcess()
    Data1 = dp.extract_from_sensor(final_array, case_index=0)
    print(f"✅ Sensor data extracted: {Data1.shape}")

    # Sample random segments
    num_samples = args.num_samples
    time_steps = config.INPUT_STEPS + config.OUTPUT_STEPS
    start_indices = np.random.randint(0, len(Data1) - time_steps, num_samples)
    samples = np.array([Data1[i:i + time_steps] for i in start_indices])

    X = samples[:, :config.INPUT_STEPS]
    y_true = samples[:, config.INPUT_STEPS:]

    X_normalized = (X - min_scaler) / denom
    X_input = X_normalized.reshape((X_normalized.shape[0], X_normalized.shape[1], 1))

    print(f"🔮 Predicting...")
    y_pred = model.predict(X_input, verbose=0)
    y_pred_real = y_pred * denom + min_scaler
    print(f"✅ Predictions done and rescaled")

    # Prepare to save if needed
    if args.save:
        plot_dir = run_dir / "full_predictions"
        plot_dir.mkdir(exist_ok=True)
        print(f"💾 Saving plots to: {plot_dir}")

    # --- Visualization ---
    for i in range(num_samples):
        plt.figure(figsize=(12, 4))

        # Time axes
        time_past = np.arange(config.INPUT_STEPS)
        time_future = np.arange(config.INPUT_STEPS, config.INPUT_STEPS + config.OUTPUT_STEPS)

        # Past data (green)
        plt.plot(time_past, X[i], 's-', color='green', label="Past Data", markersize=5, linewidth=2)

        # Actual future (blue)
        plt.plot(time_future, y_true[i], 'o-', color='blue', label="Actual Future", markersize=5, linewidth=2)

        # Predicted future (red)
        plt.plot(time_future, y_pred_real[i], 'D--', color='red', label="Predicted Future", markersize=5, linewidth=2)

        # Baseline and labels
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.xlabel("Time Step", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title(f"Time Series Prediction for Sample {i} LSTM Network - Missing {config.OUTPUT_STEPS}%", fontsize=14)

        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if args.save:
            save_path = plot_dir / f"prediction_sample_{i+1:03d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"💾 Saved: {save_path.name}")
        else:
            plt.show()

    if args.save:
        print(f"✅ All {num_samples} plots saved to: {plot_dir}")
    else:
        print(f"✅ Displayed {num_samples} prediction plots")


if __name__ == "__main__":
    main()
