import matplotlib.pyplot as plt
import numpy as np
from Data.dataloader import DataLoader, DataProcess
from Data import config # Cần để lấy INPUT_STEPS, OUTPUT_STEPS

def plot_single_sample():
    """
    Tải dữ liệu, tạo mẫu (samples), và vẽ một mẫu (X, y)
    để minh họa nhiệm vụ dự đoán.
    """
    
    # --- 1. Tải Dữ liệu ---
    print("Đang tải dữ liệu...")
    dl = DataLoader(folder_path=config.FOLDER_PATH)
    try:
        final_array = dl.read_data()
    except Exception as e:
        print(f"Lỗi tải dữ liệu: {e}")
        return

    # --- 2. Trích xuất Chuỗi thời gian ---
    # (Mô phỏng lại logic của run_tcn_lstm.py)
    dp = DataProcess()
    # Lấy chuỗi thời gian của case 0, sensor 9 (mặc định từ config)
    sensor_series = dp.extract_from_sensor(final_array, case_index=0)
    
    # --- 3. Tạo Mẫu ---
    print("Đang tạo các mẫu (samples)...")
    # Sử dụng tỷ lệ 70:15:15 (test_size=0.3)
    X_train, X_val, X_test, y_train, y_val, y_test = dp.create_sample(
        data=sensor_series,
        input_steps=config.INPUT_STEPS,
        output_steps=config.OUTPUT_STEPS,
        test_size=0.9, # 30% cho val+test = 70% train
        val_ratio_within_temp=0.0, # 15% val, 15% test
        limit_samples=config.DESIGN_SAMPLES # Giới hạn số mẫu (nếu có)
    )

    if X_train.shape[0] == 0:
        print("Không tạo được mẫu huấn luyện nào để vẽ.")
        return

    print(f"Tạo xong: {X_train.shape[0]} mẫu huấn luyện.")

    # --- 4. Vẽ Mẫu ---
    
    # Chọn 1 mẫu để vẽ (ví dụ: mẫu đầu tiên)
    sample_index = 0
    input_sample = X_train[sample_index]
    target_sample = y_train[sample_index]

    # Lấy độ dài từ config
    input_steps = config.INPUT_STEPS
    output_steps = config.OUTPUT_STEPS

    # Tạo trục x (chỉ số thời gian)
    # Input: 0, 1, ..., (input_steps - 1)
    x_input = np.arange(0, input_steps)
    
    # Target: input_steps, ..., (input_steps + output_steps - 1)
    x_target = np.arange(input_steps, input_steps + output_steps)

    print(f"Đang vẽ mẫu số {sample_index}...")
    plt.figure(figsize=(15, 6))
    
    # Vẽ phần Input (X)
    plt.plot(x_input, input_sample, label='Dữ liệu Input (X)', color='blue', marker='.')
    
    # Vẽ phần Target (y)
    plt.plot(x_target, target_sample, label='Dữ liệu Target (y) - Cần dự đoán', color='orange', marker='.')
    
    # Vẽ đường ranh giới
    plt.axvline(x=input_steps - 0.5, color='red', linestyle='--', label='Điểm bắt đầu dự đoán')
    
    plt.title(f"Minh họa Mẫu Huấn luyện (Sample Index {sample_index})")
    plt.xlabel("Time Step (Bước thời gian)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show() # Hiển thị biểu đồ

if __name__ == "__main__":
    plot_single_sample()