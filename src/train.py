import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ===============================
# 1️⃣ Đọc dữ liệu
# ===============================
df = pd.read_csv("./data/processed/ssense_clean.csv")

# Xóa dòng có giá trị thiếu
df = df.dropna(subset=["price_usd", "season"])

# ===============================
# 2️⃣ Chuẩn bị dữ liệu huấn luyện
# ===============================
# Chỉ dùng các đặc trưng phù hợp
features = ["brand", "type", "season"]
target = "price_usd"

X = df[features]
y = df[target]

# Phân chia train/test theo mùa (đảm bảo không trộn lẫn mùa)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=df["season"])

# ===============================
# 3️⃣ Pipeline xử lý + Mô hình Linear Regression
# ===============================
categorical_features = ["brand", "type", "season"]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('regressor', LinearRegression())
])

# ===============================
# 4️⃣ Huấn luyện mô hình
# ===============================
model.fit(X_train, y_train)

# ===============================
# 5️⃣ Đánh giá mô hình
# ===============================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("===== ĐÁNH GIÁ MÔ HÌNH =====")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.3f}")

# ===============================
# 6️⃣ Dự đoán giá mùa xuân chỉ với sản phẩm mùa đông
# ===============================
spring_df = df[df["season"] == "winter"].copy()  # Chỉ lấy sản phẩm mùa đông
spring_df["season"] = "spring"                   # Gán mùa dự đoán là mùa xuân

spring_pred = model.predict(spring_df[features])
spring_df["spring_price"] = spring_pred

# ===============================
# 7️⃣ Tính % thay đổi và gợi ý mua hàng
# ===============================
spring_df["current_price"] = df.loc[spring_df.index, "price_usd"]  # Giá hiện tại mùa đông
spring_df["change_rate"] = (spring_df["spring_price"] - spring_df["current_price"]) / spring_df["current_price"] * 100

# > 0	Giá dự đoán mùa xuân cao hơn mùa đông → giá tăng → có thể gợi ý “nên mua ngay mùa đông” trước khi giá tăng.
# < 0	Giá dự đoán mùa xuân thấp hơn mùa đông → giá giảm → có thể gợi ý “không nên mua ngay, chờ giảm giá”.
spring_df["recommendation"] = spring_df["change_rate"].apply(
    lambda x: "Không nên mua" if x < 0 else "Nên mua ngay"
)

# ===============================
# 8️⃣ Xuất kết quả
# ===============================
result = spring_df[["brand", "type", "description", "current_price", "spring_price", "change_rate", "recommendation"]]
result.to_csv("./results/price_forecast.csv", index=False)

joblib.dump(model, "./models/season_price_predict.pkl")

print("\n✅ File kết quả: price_forecast.csv")
print("✅ File model: season_price_predict.pkl")
