"""
Simple prediction visualization script based on trained model.

Usage:
    python visualize_predictions.py outputs/tcn_lstm/5p/run_TCN_LSTM_20251112_114530
    python visualize_predictions.py outputs/tcn_lstm/5p/run_TCN_LSTM_20251112_114530 --num-samples 20
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from Data.dataloader import DataLoader, DataProcess
from Data import config
from Model.tcn_lstm import TCN_LSTM, ResidualBlock


def parse_args():
    p = argparse.ArgumentParser(description="Visualize predictions from trained model")
    p.add_argument("run_dir", type=str, help="Path to run directory (e.g., outputs/tcn_lstm/5p/run_XXX)")
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

    # 1️⃣ Load model
    model_path = run_dir / "model_saved.keras"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    print(f"📦 Loading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(str(model_path))
        print(f"✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️  Standard loading failed, trying with custom_objects...")
        try:
            custom_objects = {'TCN_LSTM': TCN_LSTM, 'ResidualBlock': ResidualBlock}
            model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
            print(f"✅ Model loaded with custom_objects!")
        except Exception as e2:
            print(f"❌ Failed to load model: {e2}")
            sys.exit(1)

    # 2️⃣ Load scaler values
    scaler_path = run_dir / "scaler_values.npy"
    if not scaler_path.exists():
        print(f"❌ Scaler not found: {scaler_path}")
        sys.exit(1)

    min_scaler, max_scaler = np.load(scaler_path)
    print(f"📊 Scaler loaded: min={min_scaler:.6f}, max={max_scaler:.6f}")

    # 3️⃣ Load original data
    print(f"📂 Loading data from: {config.FOLDER_PATH}")
    dl = DataLoader(folder_path=config.FOLDER_PATH)
    try:
        final_array = dl.read_data()
        print(f"✅ Data loaded: {final_array.shape}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        sys.exit(1)

    dp = DataProcess()
    Data1 = dp.extract_from_sensor(final_array, case_index=config.CASE_INDEX)
    print(f"✅ Sensor data extracted: {Data1.shape}")

    # 4️⃣ Sample random segments from Data1
    num_samples = args.num_samples
    time_steps = config.INPUT_STEPS + config.OUTPUT_STEPS  # e.g., 100 + 5 = 105

    # Ensure we have enough data
    if len(Data1) < time_steps:
        print(f"❌ Not enough data: {len(Data1)} < {time_steps}")
        sys.exit(1)

    # Random sampling
    start_indices = np.random.randint(0, len(Data1) - time_steps, num_samples)
    samples = np.array([Data1[i:i + time_steps] for i in start_indices])
    print(f"✅ Sampled {num_samples} segments: {samples.shape}")

    # 5️⃣ Split into X (input) and y_true (ground truth)
    X = samples[:, :config.INPUT_STEPS]  # First 100 timesteps
    y_true = samples[:, config.INPUT_STEPS:]  # Last 5 timesteps

    print(f"📊 X (input): {X.shape}")
    print(f"📊 y_true (ground truth): {y_true.shape}")

    # 6️⃣ Normalize X
    denom = max_scaler - min_scaler
    if denom == 0:
        denom = 1  # Avoid division by zero

    X_normalized = (X - min_scaler) / denom
    X_input = X_normalized.reshape((X_normalized.shape[0], X_normalized.shape[1], 1))
    print(f"✅ Normalized X_input: {X_input.shape}")

    # 7️⃣ Predict
    print(f"🔮 Predicting...")
    y_pred = model.predict(X_input, verbose=0)
    print(f"✅ Predictions done: {y_pred.shape}")

    # 8️⃣ Inverse transform predictions to original scale
    y_pred_real = y_pred * denom + min_scaler
    print(f"✅ Predictions converted to original scale")

    # 9️⃣ Visualize
    n_steps = config.INPUT_STEPS
    time_future = np.arange(n_steps, n_steps + config.OUTPUT_STEPS)

    if args.save:
        plot_dir = run_dir / "predictions"
        plot_dir.mkdir(exist_ok=True)
        print(f"💾 Saving plots to: {plot_dir}")

    for i in range(num_samples):
        plt.figure(figsize=(16, 4))

        # Plot past data (input)
        time_input = np.arange(n_steps)
        plt.plot(time_input, X[i], 's-', label="Past Data (Input)", color='green',
                markersize=3, linewidth=1.5, alpha=0.7)

        # Plot actual future (ground truth)
        plt.plot(time_future, y_true[i], 'o-', label="Actual Future (Ground Truth)",
                color='blue', markersize=5, linewidth=2)

        # Plot predicted future
        plt.plot(time_future, y_pred_real[i], 'D--', label="Predicted Future",
                color='red', markersize=5, linewidth=2)

        # Connect last point of past to future
        plt.plot([n_steps - 1, n_steps], [X[i, -1], y_true[i, 0]],
                'b-', linewidth=1, alpha=0.5)
        plt.plot([n_steps - 1, n_steps], [X[i, -1], y_pred_real[i, 0]],
                'r--', linewidth=1, alpha=0.5)

        # Formatting
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.axvline(n_steps, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

        plt.xlabel("Time Step", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title(f"Time Series Prediction - Sample {i+1}/{num_samples} (Case {config.CASE_INDEX})", fontsize=14)
        plt.legend(loc='best', fontsize=10)
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
