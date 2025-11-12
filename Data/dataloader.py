import os
from typing import List, Tuple, Optional

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from . import config


class DataLoader:
    """Load .mat files from a folder and provide simple plotting utilities.

    Methods mirror the operations used in the notebook: read MAT files containing
    key 'data', trim to `trim_rows`, and produce an array with shape
    (n_cases, n_sensors, timesteps).
    """

    def __init__(self, folder_path: Optional[str] = None, trim_rows: int = 64000, sensors_expected: int = 27):
        self.folder_path = folder_path or config.FOLDER_PATH
        self.trim_rows = trim_rows
        self.sensors_expected = sensors_expected
        self.data = None  # Will hold final_array: shape (cases, sensors, timesteps)

    def read_data(self) -> np.ndarray:
        """Read all .mat files in folder_path, collect 'data' matrices, trim rows,
        and return stacked array with shape (n_cases, sensors, timesteps).

        Skips files that don't contain key 'data' or have unexpected number of columns.
        """
        mat_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.mat')])
        all_data = []

        for file_name in mat_files:
            file_path = os.path.join(self.folder_path, file_name)
            try:
                mat = loadmat(file_path)
            except Exception as e:
                print(f"Warning: failed to load {file_name}: {e}")
                continue

            if 'data' not in mat:
                print(f"Warning: {file_name} does not contain key 'data'. Skipping.")
                continue

            data_matrix = mat['data']
            trimmed = data_matrix[: self.trim_rows, :]
            if trimmed.shape[1] != self.sensors_expected:
                print(f"Warning: unexpected column count in {file_name}: {trimmed.shape}. Skipping.")
                continue
            all_data.append(trimmed)

        if len(all_data) == 0:
            raise RuntimeError("No valid .mat data files found in folder: %s" % self.folder_path)

        final_array = np.stack(all_data, axis=0)  # (n_cases, timesteps, sensors)
        # Notebook expects shape (cases, sensors, timesteps)
        final_array = np.swapaxes(final_array, 1, 2)
        self.data = final_array
        return final_array

    def plot_data(self, case_index: int = 0, sensors: Optional[List[int]] = None, figsize: Tuple[int, int] = (12, 54)) -> None:
        """Plot sensors for a given case.

        Args:
            case_index: index of case to plot (default 0)
            sensors: list of sensor indices (0-based) to plot. If None plot all sensors.
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call read_data() first.")

        data = self.data
        if case_index < 0 or case_index >= data.shape[0]:
            raise IndexError("case_index out of bounds")

        case = data[case_index]  # shape (sensors, timesteps)
        n_sensors = case.shape[0]
        if sensors is None:
            sensors_to_plot = list(range(n_sensors))
        else:
            sensors_to_plot = sensors

        n = len(sensors_to_plot)
        fig, axes = plt.subplots(n, 1, figsize=figsize)
        if n == 1:
            axes = [axes]

        for i, s in enumerate(sensors_to_plot):
            axes[i].plot(case[s])
            axes[i].set_title(f"Sensor {s+1} (case {case_index})")
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Value')

        plt.tight_layout()
        plt.show()


class DataProcess:
    """Processing utilities used in the notebook: extract cases/sensors, slice a sensor,
    create sliding-window samples and apply min-max scaling.
    """

    def __init__(self):
        pass

    def extract(self, final_array: np.ndarray) -> List[List[np.ndarray]]:
        """Split final_array into a list of cases, each containing a list of sensors (1D arrays).

        final_array is expected shape (cases, sensors, timesteps).
        Returns: separated_cases where separated_cases[case_idx][sensor_idx] is 1D numpy array.
        """
        separated_cases = []
        for i in range(final_array.shape[0]):
            case_data = final_array[i]  # shape (sensors, timesteps)
            separated_sensors = [case_data[j] for j in range(case_data.shape[0])]
            separated_cases.append(separated_sensors)
        return separated_cases

    def extract_from_sensor(self, final_array: np.ndarray, case_index: int = 0, sensor_index: Optional[int] = None,
                            step_start: Optional[int] = None, step_finish: Optional[int] = None) -> np.ndarray:
        """Extract a single sensor time series and slice between step_start and step_finish.

        Defaults use values from `config`.
        """
        if sensor_index is None:
            sensor_index = config.SELECTED_SENSOR - 1
        if step_start is None:
            step_start = config.STEP_START
        if step_finish is None:
            step_finish = config.STEP_FINISH

        if final_array is None:
            raise RuntimeError("final_array is required")

        if case_index < 0 or case_index >= final_array.shape[0]:
            raise IndexError("case_index out of range")

        case = final_array[case_index]  # (sensors, timesteps)
        if sensor_index < 0 or sensor_index >= case.shape[0]:
            raise IndexError("sensor_index out of range")

        sensor_data = case[sensor_index]
        return sensor_data[step_start:step_finish]

    def create_sample(self, data: np.ndarray, input_steps: int, output_steps: int,
                      test_size: float = 0.3, val_ratio_within_temp: float = 0.5, random_state: int = 42,
                      limit_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create sliding-window samples (X,y) and split into train/val/test.

        First splits the raw time series into train/val/test sets to maintain temporal independence,
        then creates sequences within each set separately.

        Args:
            data: 1D array of time series data
            input_steps: number of input timesteps per sample
            output_steps: number of output timesteps to predict
            test_size: fraction of data to use for test+validation (default 0.3 for 70/15/15 split)
            val_ratio_within_temp: how to split test_size between val/test (default 0.5 for equal split)
            random_state: random seed for reproducibility
            limit_samples: if provided, limit total samples (split proportionally)

        Returns: X_train, X_val, X_test, y_train, y_val, y_test
        """
        def _create_sequences(arr, in_steps, out_steps, limit=None):
            """Create sliding window sequences from an array.
            
            Args:
                arr: 1D array of time series data
                in_steps: number of input timesteps
                out_steps: number of output timesteps
                limit: optional max number of sequences to return (random selection)
            """
            # Calculate valid sequences that can be created
            n_sequences = len(arr) - in_steps - out_steps + 1
            
            # If limit specified and smaller than possible sequences, randomly select positions
            if limit is not None and limit < n_sequences:
                # Generate all possible start positions
                all_positions = np.arange(n_sequences)
                # Randomly select limit positions
                selected = np.random.RandomState(seed=random_state).choice(
                    all_positions, size=limit, replace=False
                )
                selected.sort()  # Keep temporal order within the random selection
            else:
                selected = np.arange(n_sequences)
            
            X, y = [], []
            for i in selected:
                X.append(arr[i: i + in_steps])
                y.append(arr[i + in_steps: i + in_steps + out_steps])
            
            return np.array(X), np.array(y)
        
        # 1. First split the raw data into train/val/test
        # Calculate split points
        total_points = len(data)
        train_end = int(total_points * (1 - test_size))  # 70% for train
        val_size = int(total_points * test_size * (1 - val_ratio_within_temp))  # 15% for val
        
        # Split maintaining temporal order
        train_data = data[:train_end]
        val_data = data[train_end:train_end + val_size]
        test_data = data[train_end + val_size:]
        
        # 2. Create sequences for each set independently
        # Calculate samples per set if limit is specified (maintain 70/15/15 ratio)
        if limit_samples is not None:
            train_samples = int(limit_samples * 0.7)  # 70%
            val_samples = int(limit_samples * 0.15)   # 15%
            test_samples = limit_samples - train_samples - val_samples  # remaining ~15%
        else:
            train_samples = val_samples = test_samples = None
        
        # Create sequences for each set
        X_train, y_train = _create_sequences(train_data, input_steps, output_steps, limit=train_samples)
        X_val, y_val = _create_sequences(val_data, input_steps, output_steps, limit=val_samples)
        X_test, y_test = _create_sequences(test_data, input_steps, output_steps, limit=test_samples)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def minmax_scaler(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      save_path: str = "scaler_values.npy") -> Tuple[Tuple[np.ndarray, ...], float, float]:
        """Apply min-max scaling using min/max from TRAINING DATA ONLY to prevent data leakage.

        Important: We compute min/max statistics ONLY from training data (X_train, y_train)
        and apply the same transformation to validation and test sets. This ensures no
        information from val/test sets leaks into the training process.

        Returns: ((X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s), min_val, max_val)
        """
        # Compute min/max ONLY from training data to prevent data leakage
        train_all = np.concatenate([X_train.ravel(), y_train.ravel()])
        min_val = train_all.min()
        max_val = train_all.max()

        denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0

        # Apply the same transformation to all sets
        X_train_s = (X_train - min_val) / denom
        y_train_s = (y_train - min_val) / denom
        X_val_s = (X_val - min_val) / denom
        y_val_s = (y_val - min_val) / denom
        X_test_s = (X_test - min_val) / denom
        y_test_s = (y_test - min_val) / denom

        # Save scaler
        np.save(save_path, np.array([min_val, max_val]))

        return (X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s), min_val, max_val


__all__ = ["DataLoader", "DataProcess"]

