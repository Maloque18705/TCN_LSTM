import time
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Data.dataloader import DataLoader, DataProcess
from Data import config
from Model.model import TCN_Model
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger


def train(epochs: int = 100, batch_size: int = 64, save_dir: str = "."):
	"""End-to-end training routine based on the notebook.

	Steps:
	- Load .mat files via DataLoader
	- Extract sensor series according to config
	- Create sliding-window samples and split into train/val/test
	- Scale using min-max based on the sensor reference
	- Build and compile TCN_Model
	- Train with EarlyStopping and CSVLogger
	- Save model, history, scaler, training time and metrics

	Args:
		epochs: maximum number of epochs to train (EarlyStopping may stop earlier)
		batch_size: training batch size
		save_dir: directory to save artifacts (model, logs, csvs)
	"""

	save_path = Path(save_dir)
	save_path.mkdir(parents=True, exist_ok=True)

	# 1) Load data
	dl = DataLoader(folder_path=config.FOLDER_PATH)
	print("Loading .mat files from:", dl.folder_path)
	final_array = dl.read_data()  # shape (cases, sensors, timesteps)

	# 2) Extract sensor series
	dp = DataProcess()
	sensor_series = dp.extract_from_sensor(final_array, case_index=0)
	print("Extracted sensor series shape:", sensor_series.shape)

	# 3) Create samples
	X_train, X_val, X_test, y_train, y_val, y_test = dp.create_sample(
		data=sensor_series,
		input_steps=config.INPUT_STEPS,
		output_steps=config.OUTPUT_STEPS,
		limit_samples=config.DESIGN_SAMPLES,
	)
	print("Created samples:")
	print("X_train", X_train.shape, "y_train", y_train.shape)
	print("X_val", X_val.shape, "y_val", y_val.shape)
	print("X_test", X_test.shape, "y_test", y_test.shape)

	# 4) Scale
	(X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s), min_val, max_val = dp.minmax_scaler(
		X_train, y_train, X_val, y_val, X_test, y_test, data_reference=sensor_series, save_path=str(save_path / "scaler_values.npy")
	)

	# 5) Reshape to (samples, timesteps, features)
	n_features = 1
	X_train_s = X_train_s.reshape((X_train_s.shape[0], X_train_s.shape[1], n_features))
	X_val_s = X_val_s.reshape((X_val_s.shape[0], X_val_s.shape[1], n_features))
	X_test_s = X_test_s.reshape((X_test_s.shape[0], X_test_s.shape[1], n_features))

	# 6) Build model
	model = TCN_Model(num_blocks=4, filters=64, kernel_size=3, target_len=config.OUTPUT_STEPS)
	model.build(input_shape=(None, config.INPUT_STEPS, n_features))
	model.summary()

	model.compile(
		optimizer="adam",
		loss='mse',
		metrics=["mean_absolute_error"],
	)

	# Callbacks
	early_stop = EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss')
	csv_logger = CSVLogger(str(save_path / "training_log.csv"), append=False)

	# 7) Train
	start_time = time.time()
	history = model.fit(
		X_train_s,
		y_train_s,
		validation_data=(X_val_s, y_val_s),
		epochs=epochs,
		batch_size=batch_size,
		callbacks=[early_stop, csv_logger],
		verbose=1,
	)
	training_time = round(time.time() - start_time, 2)
	print(f"Training finished in {training_time} seconds")

	# Save training time
	try:
		pd.DataFrame({"Training Time (s)": [training_time]}).to_csv(str(save_path / "training_time.csv"), index=False)
	except Exception:
		# pandas optional
		with open(save_path / "training_time.txt", "w") as f:
			f.write(str(training_time))

	# 8) Save model and history
	model.save(str(save_path / "model_saved"), save_format="tf")
	with open(save_path / "history_saved.pkl", "wb") as f:
		pickle.dump(history.history, f)

	# 9) Evaluate and compute metrics

	# Predictions (on scaled data)
	y_train_pred_scaled = model.predict(X_train_s)
	y_val_pred_scaled = model.predict(X_val_s)
	y_test_pred_scaled = model.predict(X_test_s)

	# Convert back to original scale
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
		pd.DataFrame(metrics).to_csv(str(save_path / "metrics.csv"), index=False)
	except Exception:
		# Fallback to plain text
		with open(save_path / "metrics.txt", "w") as f:
			f.write(str(metrics))

	print("Training complete. Artifacts saved to:", save_path)

	return {
		"model": model,
		"history": history.history,
		"metrics": metrics,
		"training_time_s": training_time,
	}
