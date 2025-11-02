"""Run LSTM training entrypoint similar to run_tcn_lstm.py.

This script builds data, creates sequences, trains an LSTM model, saves artifacts
into a timestamped run folder and writes a summary and plots (loss + MAE).
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
import traceback
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from Model.lstm import build_lstm_model
from Data.dataloader import DataLoader, DataProcess
from Data import config


def parse_args():
    p = argparse.ArgumentParser(description="Train LSTM model and save artifacts")
    p.add_argument("--epochs", type=int, default=20, help="Maximum number of epochs")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    p.add_argument("--outdir", type=str, default="./outputs", help="Base output directory")
    p.add_argument("--limit-samples", type=int, default=None, help="Limit total samples for faster runs")
    return p.parse_args()


def main():
    args = parse_args()

    base_out = Path(args.outdir)
    base_out.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_out / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting LSTM training. Artifacts will be saved to: {run_dir}")

    # 1) Load data
    dl = DataLoader(folder_path=config.FOLDER_PATH)
    try:
        final_array = dl.read_data()
    except Exception as e:
        print(f"Warning: failed to load data from {config.FOLDER_PATH}: {e}")
        # fallback synthetic
        final_array = np.random.randn(1, 27, 64000)

    dp = DataProcess()
    sensor_series = dp.extract_from_sensor(final_array, case_index=0)

    # 2) Create samples
    X_train, X_val, X_test, y_train, y_val, y_test = dp.create_sample(
        data=sensor_series,
        input_steps=config.INPUT_STEPS,
        output_steps=config.OUTPUT_STEPS,
        limit_samples=args.limit_samples or config.DESIGN_SAMPLES,
    )

    print("Created samples:")
    print("X_train", X_train.shape, "y_train", y_train.shape)

    if X_train.size == 0:
        print("Not enough data to train. Exiting.")
        sys.exit(1)

    # 3) Scale and reshape
    (X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s), min_val, max_val = dp.minmax_scaler(
        X_train, y_train, X_val, y_val, X_test, y_test, data_reference=sensor_series, save_path=str(run_dir / "scaler_values.npy")
    )

    n_features = 1
    X_train_s = X_train_s.reshape((X_train_s.shape[0], X_train_s.shape[1], n_features))
    X_val_s = X_val_s.reshape((X_val_s.shape[0], X_val_s.shape[1], n_features))
    X_test_s = X_test_s.reshape((X_test_s.shape[0], X_test_s.shape[1], n_features))

    # 4) Build model
    model = build_lstm_model(input_shape=(X_train_s.shape[1], X_train_s.shape[2]), num_layers=2, units=64, target_len=config.OUTPUT_STEPS)
    model.summary()

    # 5) Train
    start_time = time.time()
    try:
        history = model.fit(
            X_train_s,
            y_train_s,
            validation_data=(X_val_s, y_val_s),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
        )
    except Exception as e:
        print("Training failed:")
        traceback.print_exc()
        sys.exit(1)
    training_time = round(time.time() - start_time, 2)

    # 6) Save artifacts
    try:
        model.save(str(run_dir / "model_saved"), save_format="tf")
    except Exception as e:
        print(f"Warning: failed to save model: {e}")

    # save history
    try:
        with open(run_dir / "history_saved.pkl", "wb") as f:
            pickle.dump(history.history, f)
    except Exception as e:
        print(f"Warning: failed to save history: {e}")

    # save training time
    try:
        pd.DataFrame({"Training Time (s)": [training_time]}).to_csv(run_dir / "training_time.csv", index=False)
    except Exception:
        with open(run_dir / "training_time.txt", "w") as f:
            f.write(str(training_time))

    # 7) Evaluate and metrics
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
        try:
            pd.DataFrame(metrics).to_csv(run_dir / "metrics.csv", index=False)
        except Exception:
            with open(run_dir / "metrics.txt", "w") as f:
                f.write(str(metrics))
    except Exception as e:
        print(f"Warning: evaluation failed: {e}")

    # 8) Plot history
    hist = history.history if hasattr(history, 'history') else (history if isinstance(history, dict) else None)
    if hist and isinstance(hist, dict):
        try:
            if 'loss' in hist:
                plt.figure(figsize=(10, 5))
                plt.plot(hist.get('loss', []), label='Train Loss')
                plt.plot(hist.get('val_loss', []), label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss (MSE)')
                plt.title('Training and Validation MSE Loss')
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
                plt.title('Training and Validation MAE')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(run_dir / 'mae.png')
                plt.close()
        except Exception as e:
            print(f"Warning: plotting failed: {e}")

    # 9) Save summary
    summary = {
        "timestamp": stamp,
        "out_dir": str(run_dir),
        "training_time_s": training_time,
        "metrics": metrics if 'metrics' in locals() else {},
    }
    with open(run_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("LSTM run finished. Artifacts saved to:", run_dir)


if __name__ == '__main__':
    main()
