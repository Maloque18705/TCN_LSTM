import matplotlib.pyplot as plt
import numpy as np
from Data.dataloader import DataLoader, DataProcess
from Data import config # Cần để lấy các giá trị mặc định

def plot_train_val_test_splits():
    """
    Tải dữ liệu, trích xuất 1 chuỗi sensor, và vẽ biểu đồ
    thể hiện rõ các phân vùng Train/Validation/Test.
    """
    
    # --- 1. Tải và Trích xuất Dữ liệu ---
    print("Đang tải dữ liệu...")
    dl = DataLoader(folder_path=config.FOLDER_PATH)
    try:
        final_array = dl.read_data()
    except Exception as e:
        print(f"Lỗi tải dữ liệu: {e}")
        return

    dp = DataProcess()
    # Lấy chuỗi thời gian của case 0 (mặc định)
    sensor_series = dp.extract_from_sensor(final_array, case_index=0)
    print(f"Đã trích xuất sensor. Tổng độ dài: {len(sensor_series)}")

    # --- 2. Tính toán các điểm chia (Replicate logic từ create_sample) ---
    # Sử dụng tỷ lệ 70:15:15
    test_size = 0.9  # 30% cho (Validation + Test)
    val_ratio = 0.0  # Chia 30% thành 15% Val và 15% Test
    
    total_points = len(sensor_series)
    
    # Điểm kết thúc của Train (70%)
    train_end_index = int(total_points * (1 - test_size))
    
    # Kích thước của Val (15%)
    val_size = int(total_points * test_size * (1 - val_ratio))
    
    # Điểm kết thúc của Val
    val_end_index = train_end_index + val_size
    
    print(f"Điểm chia: Train (0 -> {train_end_index}), "
          f"Val ({train_end_index} -> {val_end_index}), "
          f"Test ({val_end_index} -> {total_points})")

    # --- 3. Vẽ Biểu đồ ---
    plt.figure(figsize=(20, 7))
    
    # Vẽ toàn bộ chuỗi dữ liệu
    plt.plot(sensor_series, label='Dữ liệu gốc (Sensor)', color='black', linewidth=0.7)
    
    # Tô màu các vùng
    # Vùng Train (màu xanh)
    plt.axvspan(0, train_end_index, 
                color='blue', alpha=0.2, 
                label=f'Train Data {(1-test_size)*100}% - {train_end_index} điểm')
    
    # Vùng Validation (màu cam)
    plt.axvspan(train_end_index, val_end_index, 
                color='orange', alpha=0.3, 
                label=f'Validation Data {val_ratio*100}% - {val_size} điểm')
    
    # Vùng Test (màu xanh lá)
    plt.axvspan(val_end_index, total_points, 
                color='green', alpha=0.3, 
                label=f'Test Data {(1-val_ratio)*100}% - {total_points - val_end_index} điểm')

    plt.title(f"Phân chia Train/Validation/Test cho Dữ liệu Sensor {config.SELECTED_SENSOR} (Case 0)")
    plt.xlabel("Time Step (Bước thời gian)")
    plt.ylabel("Value")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show() # Hiển thị biểu đồ

if __name__ == "__main__":
    plot_train_val_test_splits()