# src/train.py

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load file data
FILE_PATH = 'data/processed/ssense_clean.csv'
try:
    df = pd.read_csv(FILE_PATH)
    print(f"Đã load file {FILE_PATH} thành công.")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file {FILE_PATH}. Hãy đảm bảo file đã tồn tại.")
    exit()

# Kiểm tra sự tồn tại của các cột cần thiết
REQUIRED_COLUMNS = ['brand', 'description', 'price_usd', 'type']
if not all(col in df.columns for col in REQUIRED_COLUMNS):
    print("Lỗi: File CSV thiếu một hoặc nhiều cột cần thiết (brand, description, price_usd, type).")
    exit()

# X: Đặc trưng (Features), y: Biến mục tiêu (Target - price_usd)
# SỬ DỤNG CỘT price_usd LÀM MỤC TIÊU
X = df.drop('price_usd', axis=1)
y = df['price_usd']

# Xử lý đơn giản các giá trị NaN trên các cột đặc trưng (rất quan trọng cho Pipeline)
X['description'] = X['description'].fillna('')
X['brand'] = X['brand'].astype(str).fillna('Unknown')
X['type'] = X['type'].astype(str).fillna('Unknown')

# Chia tập dữ liệu (60/40)
# test_size=0.4 (40%), train_size=0.6 (60%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print(f"Kích thước tập Train: {len(X_train)} | Kích thước tập Test: {len(X_test)}")

# ----------------------------------------------------
# 2. Tiền xử lý (Preprocessing)
# ----------------------------------------------------

# Định nghĩa các cột MỚI
text_features = 'description'
categorical_features = ['brand', 'type']

# Tạo Transformer cho Text (TF-IDF) và Categorical (One-Hot Encoding)
preprocessor = ColumnTransformer(
    transformers=[
        # Vectorize cột text (description) → TF-IDF
        ('text_vec', TfidfVectorizer(max_features=5000), text_features), 
        # Encode cột category (brand, type)
        ('cat_ohe', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop' # Chỉ giữ lại các cột đã được định nghĩa
)

# ----------------------------------------------------
# 3. Huấn luyện (Training)
# ----------------------------------------------------

# Tạo Pipeline bao gồm Preprocessor và Model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression()) # Train Linear Regression
])

print("\nBắt đầu huấn luyện mô hình Linear Regression...")
model_pipeline.fit(X_train, y_train)
print("Huấn luyện hoàn tất.")

# Dự đoán trên tập Test
y_pred = model_pipeline.predict(X_test)

# Tính R², MAE, MSE
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n--- Kết quả Đánh giá Mô hình (trên tập Test) ---")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f} USD")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} USD")

# So sánh với baseline (dự đoán toàn bộ bằng giá trung bình)
mean_price = y_train.mean()
y_baseline = np.full_like(y_test, mean_price)

r2_baseline = r2_score(y_test, y_baseline)
mae_baseline = mean_absolute_error(y_test, y_baseline)

print("\n--- Kết quả Đánh giá Baseline (Giá trung bình) ---")
print(f"Giá trung bình tập Train: {mean_price:.2f} USD")
print(f"R-squared (R²) Baseline: {r2_baseline:.4f}") 
print(f"Mean Absolute Error (MAE) Baseline: {mae_baseline:.2f} USD")

# ----------------------------------------------------
# 4. Lưu model
# ----------------------------------------------------

# Đảm bảo thư mục 'models' tồn tại
if not os.path.exists('models'):
    os.makedirs('models')

# Dùng joblib để lưu model
model_filename = "models/pipeline_lr.pkl"
joblib.dump(model_pipeline, model_filename)
print(f"\nĐã lưu mô hình Pipeline vào: {model_filename}")

# ----------------------------------------------------
# 5. Kết quả (Vẽ biểu đồ)
# ----------------------------------------------------

# Vẽ biểu đồ Actual vs Predicted
plt.figure(figsize=(10, 6))
# Lấy mẫu để biểu đồ không bị quá tải
sample_size = min(500, len(y_test))
sample_indices = np.random.choice(len(y_test), size=sample_size, replace=False)
y_test_sample = y_test.iloc[sample_indices]
y_pred_sample = y_pred[sample_indices]

plt.scatter(y_test_sample, y_pred_sample, alpha=0.5)
plt.plot([y_test_sample.min(), y_test_sample.max()], [y_test_sample.min(), y_test_sample.max()], 'r--', lw=2, label='Đường lý tưởng (y=x)')
plt.xlabel("Giá thực tế (Actual Price - USD)")
plt.ylabel("Giá dự đoán (Predicted Price - USD)")
plt.title("Biểu đồ Giá thực tế so với Giá dự đoán (Tập Test)")
plt.legend()

# Lưu biểu đồ vào thư mục 'results'
if not os.path.exists('results'):
    os.makedirs('results')
plot_filename = 'results/actual_vs_predicted_usd.png'
plt.savefig(plot_filename)
plt.close()
print(f"Đã lưu biểu đồ Actual vs Predicted vào: {plot_filename}")

# Số liệu kết quả dự đoán
results = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "MAE_Baseline": mae_baseline,
    "R2_Baseline": r2_baseline,
    "Model_File": model_filename,
    "Plot_File": plot_filename
}

print("\n--- Số liệu kết quả dự đoán ---")
for k, v in results.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")