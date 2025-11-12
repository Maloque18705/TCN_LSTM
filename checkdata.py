import matplotlib.pyplot as plt
from Data.dataloader import DataLoader, DataProcess
from Data import config # Cần để lấy các giá trị mặc định

# 1. Khởi tạo các lớp
dl = DataLoader(folder_path=config.FOLDER_PATH)
dp = DataProcess()

print("Đang tải dữ liệu (Loading data)...")
try:
    # 2. Tải dữ liệu thô (giống run_tcn_lstm.py)
    final_array = dl.read_data() # Shape (cases, sensors, timesteps)
    print(f"Tải xong, shape dữ liệu thô: {final_array.shape}")

    # 3. XỬ LÝ: Trích xuất 1 sensor (giống run_tcn_lstm.py)
    # Hàm này sẽ tự động lấy case 0, sensor 9, và cắt từ 5000-60000
    # dựa trên file config
    sensor_series = dp.extract_from_sensor(final_array, case_index=0)
    
    print(f"Đã trích xuất sensor, shape chuỗi thời gian: {sensor_series.shape}")

    # 4. PLOT: Vẽ chuỗi thời gian đã xử lý
    plt.figure(figsize=(15, 6))
    plt.plot(sensor_series)
    plt.title(f"Processed Data: Sensor {config.SELECTED_SENSOR} (Case 0)")
    plt.xlabel("Time Step (sau khi cắt)")
    plt.ylabel("Value")
    plt.grid(True)
    
    print("Đang hiển thị biểu đồ...")
    plt.show() # Mở cửa sổ hiển thị biểu đồ

except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")