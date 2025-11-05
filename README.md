# TCN_LSTM
### Dự án Đào tạo Mô hình Chuỗi thời gian (TCN, LSTM, TCN-LSTM)
Dự án này chứa các script để đào tạo và đánh giá ba mô hình học sâu khác nhau (TCN, LSTM, và TCN-LSTM) cho các tác vụ dự báo chuỗi thời gian. Các script này được thiết kế để xử lý dữ liệu cảm biến, tạo mẫu, huấn luyện mô hình và lưu lại các kết quả, chỉ số đánh giá, và biểu đồ.

## 🗂️ Cấu trúc thư mục (Yêu cầu)
Để các script này hoạt động chính xác, cấu trúc dự án của bạn cần phải tuân theo các đường dẫn import được sử dụng trong tệp:

```
.
├── Data/
│   ├── Z24/
│   │   └── ... (Tệp dữ liệu của bạn ở đây)
│   ├── __init__.py
│   ├── config.py      (Tệp cấu hình chính)
│   ├── dataloader.py  (Chứa các lớp DataLoader, DataProcess)
│   └── ...
├── Model/
│   ├── __init__.py
│   ├── tcn.py         (Chứa TCN_Model)
│   ├── lstm.py        (Chứa build_lstm_model)
│   ├── tcn_lstm.py    (Chứa TCN_LSTM)
│   └── ...
├── outputs/           (Thư mục này sẽ được tạo tự động)
│   ├── tcn/
│   ├── lstm/
│   └── tcn_lstm/
├── run_tcn.py         (Script chạy mô hình TCN)
├── run_lstm.py        (Script chạy mô hình LSTM)
├── run_tcn_lstm.py    (Script chạy mô hình TCN-LSTM)
├── requirements.txt
└── README.md          (Tệp này)
```

## ⚙️ Cài đặt
Tạo môi trường ảo (Khuyến nghị):

```bash
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
```

Cài đặt các gói phụ thuộc: Cài đặt tất cả các thư viện cần thiết bằng tệp requirements.txt.

```bash
pip install -r requirements.txt
```

Tệp này bao gồm các thư viện chính như tensorflow==2.10.1, pandas, scikit-learn, và matplotlib.

## 🔧 Cấu hình
Tệp Data/config.py chứa các cài đặt quan trọng cho quá trình xử lý dữ liệu.

- `INPUT_STEPS = 100`: Số bước thời gian trong chuỗi đầu vào.
- `FOLDER_PATH = r"./Data/Z24"`: Đường dẫn đến thư mục chứa dữ liệu của bạn.
- `DESIGN_SAMPLES = 10000`: Số lượng mẫu được thiết kế để sử dụng.

## ⚠️ Lưu ý quan trọng
Tệp Data/config.py chứa một lệnh "Missing Data Percentage: "  
Hãy nhập phần trăm dữ liệu bị mất vào và nhấn ENTER, ví dụ: 5, 10, 15,...

Điều này có nghĩa là mỗi khi bạn chạy bất kỳ script đào tạo nào, terminal sẽ tạm dừng và yêu cầu bạn nhập "Missing Data Percentage".  
Giá trị này được sử dụng làm số bước dự đoán đầu ra (OUTPUT_STEPS) và cũng được dùng để đặt tên cho thư mục đầu ra (ví dụ: `outputs/tcn/10p` nếu bạn nhập 10).

## 🚀 Cách chạy các Script Đào tạo
Tất cả các script đào tạo đều chấp nhận các đối số dòng lệnh để tùy chỉnh quá trình chạy.

### Các đối số dòng lệnh (Tùy chọn)
- `--epochs`: Số lượng epoch để đào tạo (mặc định: 100).
- `--batch-size`: Kích thước lô đào tạo (mặc định: 64).
- `--outdir`: Thư mục cơ sở để lưu kết quả (mặc định thay đổi theo mô hình, ví dụ: ./outputs/tcn/{OUTPUT_STEPS}p).
- `--limit-samples`: Giới hạn tổng số mẫu để chạy thử nghiệm nhanh (mặc định: None).

### 1️⃣ Đào tạo Mô hình TCN (Temporal Convolutional Network)
Script này sẽ đào tạo mô hình TCN.

```bash
python run_tcn.py
```
Hoặc chạy với các đối số tùy chỉnh:
```bash
python run_tcn.py --epochs 50 --batch-size 32
```

### 2️⃣ Đào tạo Mô hình LSTM (Long Short-Term Memory)
Script này sẽ đào tạo mô hình LSTM.

```bash
python run_lstm.py
```
Hoặc chạy với các đối số tùy chỉnh:
```bash
python run_lstm.py --epochs 50 --batch-size 32
```

### 3️⃣ Đào tạo Mô hình TCN-LSTM
Script này sẽ đào tạo mô hình TCN-LSTM lai.

```bash
python run_tcn_lstm.py
```
Hoặc chạy với các đối số tùy chỉnh:
```bash
python run_tcn_lstm.py --epochs 50 --batch-size 32
```

## 📊 Đầu ra
Sau khi chạy thành công, mỗi script sẽ tạo một thư mục con mới bên trong thư mục `--outdir` được đặt tên theo dấu thời gian (timestamp) của lần chạy đó (ví dụ: `outputs/tcn/10p/run_TCN_20251105_183000/`).

Thư mục kết quả này sẽ chứa:

- `model_saved.keras`: Tệp mô hình đã đào tạo.
- `history_saved.pkl`: Tệp pickle chứa lịch sử đào tạo (loss, mae).
- `metrics.csv`: Bảng CSV chứa các chỉ số RMSE, MAE, và R2 cho các tập train, validation, và test.
- `loss_mse.png`: Biểu đồ của MSE loss (train vs. validation).
- `mae.png`: Biểu đồ của MAE (train vs. validation).
- `training_time.csv`: Tệp CSV ghi lại tổng thời gian đào tạo.
- `scaler_values.npy`: Giá trị min/max được sử dụng bởi bộ scaler.
- `summary.json`: Tệp JSON tóm tắt các đường dẫn, thời gian và chỉ số chính.
