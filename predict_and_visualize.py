"""
Visualize TCN-LSTM predictions on test data.

Usage:
    python predict_and_visualize.py --run-dir ./outputs/tcn_lstm/5p/run_TCN_LSTM_20251112_103856
    python predict_and_visualize.py  # Auto-detect latest run
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from Data.dataloader import DataLoader, DataProcess
from Data import config
from Model.tcn_lstm import TCN_LSTM, ResidualBlock  # Import custom classes for loading


def find_latest_run(base_dir: str = "./outputs/tcn_lstm/5p") -> Path:
    """Find the most recent run directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        raise RuntimeError(f"Base directory not found: {base_dir}")

    run_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        raise RuntimeError(f"No run directories found in {base_dir}")

    latest = run_dirs[-1]
    print(f"📂 Using latest run: {latest}")
    return latest


def parse_args():
    p = argparse.ArgumentParser(description="Visualize TCN-LSTM predictions")
    p.add_argument("--run-dir", type=str, default=None, help="Run directory (auto-detect if not specified)")
    p.add_argument("--num-samples", type=int, default=10, help="Number of samples to visualize")
    p.add_argument("--save-plots", action="store_true", help="Save plots to file instead of showing")
    return p.parse_args()


def main():
    args = parse_args()

    # 1️⃣ Find run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run()

    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        sys.exit(1)

    # 2️⃣ Load model
    model_path = run_dir / "model_saved.keras"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    print(f"📦 Loading model from: {model_path}")
    try:
        # Try loading normally (works if model was trained with decorator)
        model = tf.keras.models.load_model(model_path)
    except TypeError as e:
        # Fallback: Load with custom_objects (for old models without decorator)
        print(f"⚠️  Standard loading failed, trying with custom_objects...")
        custom_objects = {
            'TCN_LSTM': TCN_LSTM,
            'ResidualBlock': ResidualBlock
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"✅ Loaded with custom_objects (model trained before decorator was added)")

    # 3️⃣ Load scaler values
    scaler_path = run_dir / "scaler_values.npy"
    if not scaler_path.exists():
        print(f"❌ Scaler values not found: {scaler_path}")
        sys.exit(1)

    scaler_values = np.load(scaler_path)
    min_scaler, max_scaler = scaler_values[0], scaler_values[1]
    print(f"📊 Scaler: min={min_scaler:.6f}, max={max_scaler:.6f}")

    # 4️⃣ Recreate test data (same process as training)
    print(f"📂 Loading data from: {config.FOLDER_PATH}")
    dl = DataLoader(folder_path=config.FOLDER_PATH)
    try:
        final_array = dl.read_data()
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        sys.exit(1)

    dp = DataProcess()
    sensor_series = dp.extract_from_sensor(final_array, case_index=config.CASE_INDEX)

    # Create samples (same as training)
    X_train, X_val, X_test, y_train, y_val, y_test = dp.create_sample(
        data=sensor_series,
        input_steps=config.INPUT_STEPS,
        output_steps=config.OUTPUT_STEPS,
        limit_samples=config.DESIGN_SAMPLES,
    )

    # Scale data
    (X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s), _, _ = dp.minmax_scaler(
        X_train, y_train, X_val, y_val, X_test, y_test,
        save_path=str(run_dir / "scaler_values_temp.npy")
    )

    # Reshape for model
    n_features = 1
    X_test_scaled = X_test_s.reshape((X_test_s.shape[0], X_test_s.shape[1], n_features))
    y_test_scaled = y_test_s

    print(f"✅ Test data: X_test_scaled {X_test_scaled.shape}, y_test_scaled {y_test_scaled.shape}")

    # 5️⃣ Visualize predictions
    num_samples = min(args.num_samples, X_test_scaled.shape[0])
    n_steps = X_test_scaled.shape[1]  # Input timesteps

    # Create output directory for plots
    if args.save_plots:
        plot_dir = run_dir / "predictions"
        plot_dir.mkdir(exist_ok=True)
        print(f"💾 Saving plots to: {plot_dir}")

    for i in range(num_samples):
        # Get input sample
        sample_input = X_test_scaled[i].reshape(1, X_test_scaled.shape[1], X_test_scaled.shape[2])

        # Predict future values
        predicted_value = model.predict(sample_input, verbose=0)[0]

        # Create time axis
        time_input = np.arange(n_steps)  # Time axis for input
        time_future = np.arange(n_steps, n_steps + len(predicted_value))  # Time axis for prediction

        # Inverse transform (scaled -> original)
        past_data = X_test_scaled[i][:, 0] * (max_scaler - min_scaler) + min_scaler
        actual_future = y_test_scaled[i] * (max_scaler - min_scaler) + min_scaler
        predicted_future = predicted_value * (max_scaler - min_scaler) + min_scaler

        # Plot
        plt.figure(figsize=(16, 4))

        # Plot past data (input)
        plt.plot(time_input, past_data, marker='s', label="Past Data (Input)", color='green', markersize=3)

        # Plot actual future (ground truth)
        plt.plot(time_future, actual_future, marker='o', label="Actual Future (Ground Truth)", color='blue', markersize=5)

        # Plot predicted future
        plt.plot(time_future, predicted_future, marker='D', label="Predicted Future", color='red', linestyle='dashed', markersize=5)

        # Connect last point of past data to actual future
        plt.plot([n_steps - 1, n_steps],
                [past_data[-1], actual_future[0]],
                color='blue', linewidth=1)

        # Connect last point of past data to predicted future
        plt.plot([n_steps - 1, n_steps],
                [past_data[-1], predicted_future[0]],
                color='red', linestyle='dashed', linewidth=1)

        # Add horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Add vertical line to separate past and future
        plt.axvline(x=n_steps, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

        # Labels and formatting
        plt.xlabel("Time Step", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.title(f"Time Series Prediction for Test Sample {i} (Case {config.CASE_INDEX})", fontsize=14)
        plt.grid(which='both', alpha=0.3)
        plt.tight_layout()

        # Save or show
        if args.save_plots:
            save_path = plot_dir / f"prediction_sample_{i:03d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"💾 Saved: {save_path}")
        else:
            plt.show()

    if args.save_plots:
        print(f"✅ All plots saved to: {plot_dir}")
    else:
        print(f"✅ Displayed {num_samples} prediction plots")


if __name__ == "__main__":
    main()
