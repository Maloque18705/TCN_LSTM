import os
import numpy as np
import scipy.io

class DataLoader:
    def __init__(self, folder_path, start_index=0, end_index=None, cut_range=None):
        """
        folder_path: thư mục chứa các file .mat
        start_index, end_index: chỉ số file cần đọc (ví dụ: SETUP4 -> start=4, end=4)
        cut_range: tuple (start_sample, end_sample) để cắt dữ liệu, ví dụ (300000, 320000)
        """
        self.folder_path = folder_path
        self.start_index = start_index
        self.end_index = end_index
        self.cut_range = cut_range

    # ============================================================
    # 1️⃣ Đọc toàn bộ file .mat hợp lệ trong folder
    # ============================================================
    def read_data(self):
        all_data = []

        for file_name in sorted(os.listdir(self.folder_path)):
            if not file_name.endswith(".mat"):
                continue

            # Lọc theo chỉ số file nếu cần (ví dụ: SETUP4.mat)
            try:
                idx = int(''.join(filter(str.isdigit, file_name)))
            except ValueError:
                idx = None

            if self.start_index and idx is not None and idx < self.start_index:
                continue
            if self.end_index and idx is not None and idx > self.end_index:
                continue

            file_path = os.path.join(self.folder_path, file_name)
            print(f"🔄 Đang đọc: {file_path}")

            mat = scipy.io.loadmat(file_path)

            # Tìm các key dạng Untitled...Z
            keys_to_extract = [key for key in mat.keys() if key.startswith("Untitled") and key.endswith("Z")]
            if not keys_to_extract:
                print(f"⚠️ {file_name} không có key Untitled...Z, bỏ qua.")
                continue

            extracted_arrays = []
            for key in keys_to_extract:
                raw_value = mat[key][0, 0]
                if isinstance(raw_value, np.void) and "Data" in raw_value.dtype.names:
                    numerical_array = raw_value["Data"]
                    if isinstance(numerical_array, np.ndarray):
                        # Flatten thành 1D để ghép
                        extracted_arrays.append(numerical_array.flatten())
            
            if not extracted_arrays:
                print(f"⚠️ {file_name} không có trường 'Data' hợp lệ, bỏ qua.")
                continue

            # Ghép các cột sensor lại
            file_data = np.column_stack(extracted_arrays)
            print(f"📊 {file_name}: {file_data.shape}")

            # Cắt dữ liệu nếu có yêu cầu
            if self.cut_range:
                start, end = self.cut_range
                file_data = file_data[start:end, :]
                print(f"✂️ Cắt dữ liệu: {file_data.shape}")

            all_data.append(file_data)

        # Gộp tất cả các file thành một ma trận duy nhất
        if not all_data:
            raise RuntimeError(f"❌ Không tìm thấy dữ liệu hợp lệ trong thư mục: {self.folder_path}")

        final_array = np.concatenate(all_data, axis=0)
        print(f"✅ Dữ liệu cuối cùng: {final_array.shape} (samples × sensors)")
        return final_array
