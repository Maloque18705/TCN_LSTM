"""
Quick visualization script - loads pre-saved test data if available.

Usage:
    python quick_visualize.py outputs/tcn_lstm/5p/run_TCN_LSTM_20251112_103856
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def parse_args():
    p = argparse.ArgumentParser(description="Quick visualize predictions")
    p.add_argument("run_dir", type=str, help="Run directory path")
    p.add_argument("--num-samples", type=int, default=10, help="Number of samples to plot")
    p.add_argument("--save", action="store_true", help="Save plots instead of showing")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    # Load model
    model_path = run_dir / "model_saved.keras"
    print(f"📦 Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Load scaler
    scaler = np.load(run_dir / "scaler_values.npy")
    min_val, max_val = scaler[0], scaler[1]
    print(f"📊 Scaler: [{min_val:.6f}, {max_val:.6f}]")

    # Try to load pre-saved test data
    test_data_path = run_dir / "test_data.npz"
    if test_data_path.exists():
        print(f"📂 Loading test data from: {test_data_path}")
        data = np.load(test_data_path)
        X_test_scaled = data['X_test_scaled']
        y_test_scaled = data['y_test_scaled']
    else:
        print(f"❌ Test data not found at: {test_data_path}")
        print(f"💡 Run 'python predict_and_visualize.py --run-dir {run_dir}' instead")
        sys.exit(1)

    print(f"✅ Data loaded: X_test {X_test_scaled.shape}, y_test {y_test_scaled.shape}")

    # Visualize
    num_samples = min(args.num_samples, X_test_scaled.shape[0])
    n_steps = X_test_scaled.shape[1]

    if args.save:
        plot_dir = run_dir / "predictions"
        plot_dir.mkdir(exist_ok=True)

    for i in range(num_samples):
        # Predict
        sample_input = X_test_scaled[i:i+1]  # Keep batch dimension
        predicted_scaled = model.predict(sample_input, verbose=0)[0]

        # Inverse transform
        past = X_test_scaled[i, :, 0] * (max_val - min_val) + min_val
        actual = y_test_scaled[i] * (max_val - min_val) + min_val
        predicted = predicted_scaled * (max_val - min_val) + min_val

        # Time axis
        t_input = np.arange(n_steps)
        t_future = np.arange(n_steps, n_steps + len(actual))

        # Plot
        plt.figure(figsize=(16, 4))
        plt.plot(t_input, past, 's-', label="Past Data", color='green', markersize=3, linewidth=1.5)
        plt.plot(t_future, actual, 'o-', label="Actual Future", color='blue', markersize=5, linewidth=2)
        plt.plot(t_future, predicted, 'D--', label="Predicted", color='red', markersize=5, linewidth=2)

        # Connect lines
        plt.plot([n_steps-1, n_steps], [past[-1], actual[0]], 'b-', linewidth=1)
        plt.plot([n_steps-1, n_steps], [past[-1], predicted[0]], 'r--', linewidth=1)

        plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.axvline(n_steps, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

        plt.xlabel("Time Step", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title(f"Prediction Sample {i}", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if args.save:
            plt.savefig(plot_dir / f"sample_{i:03d}.png", dpi=150)
            plt.close()
            print(f"💾 Saved sample {i}")
        else:
            plt.show()

    print(f"✅ Done!")


if __name__ == "__main__":
    main()
