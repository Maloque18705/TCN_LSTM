# 📊 Hướng Dẫn Visualize Predictions

Sau khi train model, bạn có thể sử dụng các script để visualize kết quả dự đoán.

---

## 🚀 **Cách 1: Sử dụng `quick_visualize.py` (Khuyến nghị)**

**Ưu điểm:** Nhanh, đơn giản, sử dụng test data đã được save sẵn

### **Bước 1: Train model (nếu chưa có)**

```bash
python run_tcn_lstm.py --epochs 50 --batch-size 64
```

→ Model và test data sẽ được save vào: `outputs/tcn_lstm/5p/run_TCN_LSTM_YYYYMMDD_HHMMSS/`

### **Bước 2: Visualize**

```bash
# Hiển thị 10 samples (default)
python quick_visualize.py outputs/tcn_lstm/5p/run_TCN_LSTM_20251112_103856

# Hiển thị 20 samples
python quick_visualize.py outputs/tcn_lstm/5p/run_TCN_LSTM_20251112_103856 --num-samples 20

# Lưu plots thay vì hiển thị
python quick_visualize.py outputs/tcn_lstm/5p/run_TCN_LSTM_20251112_103856 --save
```

**Output:** Hiển thị hoặc save các biểu đồ vào `outputs/.../predictions/`

---

## 🔄 **Cách 2: Sử dụng `predict_and_visualize.py`**

**Ưu điểm:** Tự động tìm run mới nhất, recreate data (không cần test_data.npz)

### **Sử dụng:**

```bash
# Auto-detect run mới nhất
python predict_and_visualize.py

# Chỉ định run cụ thể
python predict_and_visualize.py --run-dir outputs/tcn_lstm/5p/run_TCN_LSTM_20251112_103856

# Visualize 15 samples
python predict_and_visualize.py --num-samples 15

# Lưu plots
python predict_and_visualize.py --save-plots
```

**Lưu ý:** Script này sẽ load lại data từ folder gốc và recreate test set, có thể chậm hơn.

---

## 📈 **Biểu Đồ Sẽ Hiển Thị:**

Mỗi plot sẽ có:

```
┌─────────────────────────────────────────────────┐
│  🟢 Past Data (Input - 100 timesteps)          │
│  └─ Dữ liệu quá khứ (green squares)           │
│                                                 │
│  🔵 Actual Future (Ground Truth - 5 timesteps) │
│  └─ Giá trị thực tế (blue circles)            │
│                                                 │
│  🔴 Predicted Future (Model prediction)        │
│  └─ Dự đoán của model (red diamonds)          │
│                                                 │
│  📏 Vertical line: Phân cách quá khứ/tương lai │
│  ─── Horizontal line: y = 0                    │
└─────────────────────────────────────────────────┘
```

### **Ví dụ output:**

![Time Series Prediction](docs/example_prediction.png)

---

## 📂 **Cấu Trúc Output Directory**

Sau khi train, mỗi run sẽ có:

```
outputs/tcn_lstm/5p/run_TCN_LSTM_20251112_103856/
├── model_saved.keras          # ← Trained model
├── scaler_values.npy          # ← Min/max values
├── test_data.npz              # ← Saved test data (X_test, y_test)
├── metrics.csv                # ← RMSE, MAE, R² scores
├── summary.json               # ← Training summary
├── training_time.csv          # ← Training duration
├── history_saved.pkl          # ← Training history
├── loss_mse.png               # ← Loss plot
├── mae.png                    # ← MAE plot
└── predictions/               # ← Saved plots (if --save used)
    ├── sample_000.png
    ├── sample_001.png
    └── ...
```

---

## 🎯 **So Sánh 2 Script**

| Feature | quick_visualize.py | predict_and_visualize.py |
|---------|-------------------|-------------------------|
| **Tốc độ** | ⚡ Rất nhanh | 🐢 Chậm hơn (recreate data) |
| **Yêu cầu** | test_data.npz | Data folder gốc |
| **Auto-detect run** | ❌ Không | ✅ Có |
| **Dễ sử dụng** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Khuyến nghị** | Dùng hàng ngày | Dùng khi cần recreate |

---

## 💡 **Tips**

### **1. Lưu tất cả plots:**
```bash
python quick_visualize.py outputs/.../run_XXX --save --num-samples 50
```

### **2. Kiểm tra model performance:**
```bash
# Xem metrics
cat outputs/.../run_XXX/metrics.csv

# Xem summary
cat outputs/.../run_XXX/summary.json
```

### **3. So sánh nhiều runs:**
```python
# Script tự viết
import pandas as pd
import glob

runs = glob.glob("outputs/tcn_lstm/5p/run_*/metrics.csv")
for run in runs:
    df = pd.read_csv(run)
    print(f"\n{run}")
    print(df)
```

---

## ❓ **Troubleshooting**

### **Lỗi: "Test data not found"**
```bash
# Giải pháp 1: Retrain model (sẽ tự động save test_data.npz)
python run_tcn_lstm.py --epochs 50

# Giải pháp 2: Dùng predict_and_visualize.py thay vì quick_visualize.py
python predict_and_visualize.py
```

### **Lỗi: "Model not found"**
```bash
# Kiểm tra run directory có đúng không
ls outputs/tcn_lstm/5p/run_TCN_LSTM_YYYYMMDD_HHMMSS/

# Nếu không có file model_saved.keras, cần retrain
```

### **Lỗi: "Data folder not found"**
```bash
# Kiểm tra config
cat Data/config.py | grep FOLDER_PATH

# Đảm bảo folder data tồn tại
ls -la <FOLDER_PATH>
```

---

## 📚 **Tham Khảo**

- **Training guide:** `README.md`
- **Code mẫu:** Xem `predict_and_visualize.py`
- **Model architecture:** `Model/tcn_lstm.py`

---

## 🎨 **Customization**

Bạn có thể tùy chỉnh plots bằng cách sửa script:

```python
# Trong predict_and_visualize.py hoặc quick_visualize.py

# Đổi màu sắc
plt.plot(..., color='purple')  # Thay vì 'red'

# Đổi kích thước figure
plt.figure(figsize=(20, 6))  # Thay vì (16, 4)

# Đổi marker style
plt.plot(..., marker='^')  # Triangle thay vì diamond

# Thêm title tùy chỉnh
plt.title(f"My Custom Title - Sample {i}")
```

---

**Chúc bạn visualize thành công! 🎉**
