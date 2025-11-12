"""
Entry point to train the TCN-LSTM model and save artifacts.

Usage:
    python run_tcn_lstm.py --epochs 100 --batch-size 64 --outdir ./outputs/tcn_lstm

This script builds data, creates sequences, trains a TCN-LSTM model,
saves artifacts into a timestamped run folder and writes a summary and plots (loss + MAE).
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
import pickle
import io
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Model.tcn_lstm import TCN_LSTM
from Data.dataloader import DataLoader, DataProcess
from Data import config
from tensorflow.keras.callbacks import EarlyStopping

early_stopper = EarlyStopping(
    monitor='val_mae',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

def parse_args():
    p = argparse.ArgumentParser(description="Train TCN-LSTM model and save artifacts")
    p.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    p.add_argument("--outdir", type=str, default=f"./outputs/tcn_lstm/{config.OUTPUT_STEPS}p", help="Base output directory")
    p.add_argument("--limit-samples", type=int, default=None, help="Limit total samples for faster runs")
    return p.parse_args()


def main():
    args = parse_args()

    base_out = Path(args.outdir).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_out / f"run_TCN_LSTM_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting TCN-LSTM training. Artifacts will be saved to: {run_dir}")

    # 1️⃣ Load data
    dl = DataLoader(folder_path=config.FOLDER_PATH)
    try:
        final_array = dl.read_data()
    except Exception as e:
        print(f"⚠️ Warning: failed to load data from {config.FOLDER_PATH}: {e}")
        # final_array = np.random.randn(1, getattr(config, 'SENSORS_EXPECTED', 27), 64000)

    dp = DataProcess()
    sensor_series = dp.extract_from_sensor(final_array, case_index=config.CASE_INDEX)

    # 2️⃣ Create samples
    X_train, X_val, X_test, y_train, y_val, y_test = dp.create_sample(
        data=sensor_series,
        input_steps=config.INPUT_STEPS,
        output_steps=config.OUTPUT_STEPS,
        limit_samples=args.limit_samples or config.DESIGN_SAMPLES,
    )

    print("✅ Created samples:")
    print("X_train", X_train.shape, "y_train", y_train.shape)

    if X_train.size == 0:
        print("❌ Not enough data to train. Exiting.")
        sys.exit(1)

    # 3️⃣ Scale and reshape


    (X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s), min_val, max_val = dp.minmax_scaler(
        X_train, y_train, X_val, y_val, X_test, y_test,
        save_path=str(run_dir / "scaler_values.npy")
    )
    n_features = 1
    X_train_s = X_train_s.reshape((X_train_s.shape[0], X_train_s.shape[1], n_features))
    X_val_s = X_val_s.reshape((X_val_s.shape[0], X_val_s.shape[1], n_features))
    # X_test_s = X_test_s.reshape((X_test_s.shape[0], X_test_s.shape[1], n_features))

    # 4️⃣ Build model
    model = TCN_LSTM(num_blocks=4, filters=64, kernel_size=3, target_len=config.OUTPUT_STEPS)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
    model.build(input_shape=(None, X_train_s.shape[1], X_train_s.shape[2]))
    model.summary()
 
    # 5️⃣ Train
    start_time = time.time()
    try:
        history = model.fit(
            X_train_s,
            y_train_s,
            validation_data=(X_val_s, y_val_s),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1
        )
    except Exception:
        print("❌ Training failed:")
        traceback.print_exc()
        sys.exit(1)
    training_time = round(time.time() - start_time, 2)

    # 6️⃣ Save artifacts
    try:
        model.save(str(run_dir / "model_saved.keras"), save_format="")
    except Exception as e:
        print(f"⚠️ Warning: failed to save model: {e}")

    try:
        with open(run_dir / "history_saved.pkl", "wb") as f:
            pickle.dump(history.history, f)
    except Exception as e:
        print(f"⚠️ Warning: failed to save history: {e}")

    pd.DataFrame({"Training Time (s)": [training_time]}).to_csv(run_dir / "training_time.csv", index=False)

    # 7️⃣ Evaluate metrics
    try:
        y_train_pred_scaled = model.predict(X_train_s)
        y_val_pred_scaled = model.predict(X_val_s)
        y_test_pred_scaled = model.predict(X_test_s)

        denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
        y_train_real = y_train_s.squeeze() * denom + min_val
        y_val_real = y_val_s.squeeze() * denom + min_val
        y_test_real = y_test_s.squeeze() * denom + min_val

        y_train_pred = y_train_pred_scaled.squeeze() * denom + min_val
        y_val_pred = y_val_pred_scaled.squeeze() * denom + min_val
        y_test_pred = y_test_pred_scaled.squeeze() * denom + min_val

        def calc_metrics(y_true, y_pred):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            return rmse, mae, r2

        rmse_train, mae_train, r2_train = calc_metrics(y_train_real, y_train_pred)
        rmse_val, mae_val, r2_val = calc_metrics(y_val_real, y_val_pred)
        rmse_test, mae_test, r2_test = calc_metrics(y_test_real, y_test_pred)

        metrics = {
            "Dataset": ["Train", "Validation", "Test"],
            "RMSE": [rmse_train, rmse_val, rmse_test],
            "MAE": [mae_train, mae_val, mae_test],
            "R2": [r2_train, r2_val, r2_test],
        }

        pd.DataFrame(metrics).to_csv(run_dir / "metrics.csv", index=False)
    except Exception as e:
        print(f"⚠️ Warning: evaluation failed: {e}")

    # 8️⃣ Plot training history
    hist = history.history if hasattr(history, 'history') else (history if isinstance(history, dict) else None)
    if hist and isinstance(hist, dict):
        try:
            if 'loss' in hist:
                plt.figure(figsize=(10, 5))
                plt.plot(hist.get('loss', []), label='Train Loss')
                plt.plot(hist.get('val_loss', []), label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss (MSE)')
                plt.title(f"Training and Validation MSE Loss - TCN-LSTM - Missing {config.OUTPUT_STEPS}% Data - Case {config.CASE_INDEX}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(run_dir / 'loss_mse.png')
                plt.close()

            mae_key = None
            for k in ('mean_absolute_error', 'mae'):
                if k in hist:
                    mae_key = k
                    break
            if mae_key is not None:
                plt.figure(figsize=(10, 5))
                plt.plot(hist.get(mae_key, []), label='Train MAE')
                plt.plot(hist.get(f'val_{mae_key}', []), label='Validation MAE')
                plt.xlabel('Epochs')
                plt.ylabel('MAE')
                plt.title(f"Training and Validation MAE - TCN-LSTM - Missing {config.OUTPUT_STEPS}% Data - Case {config.CASE_INDEX}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(run_dir / 'mae.png')
                plt.close()
        except Exception as e:
            print(f"⚠️ Warning: plotting failed: {e}")

    # 9️⃣ Save summary
    summary = {
        "timestamp": stamp,
        "out_dir": str(run_dir),
        "training_time_s": training_time,
        "metrics": metrics if 'metrics' in locals() else {},
        "case_index": config.CASE_INDEX
    }
    with open(run_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ TCN-LSTM run finished. Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
