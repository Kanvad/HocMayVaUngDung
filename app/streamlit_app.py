import streamlit as st
import pandas as pd
import joblib
import numpy as np
# ==============================
#  1. Cấu hình giao diện
# ==============================
st.set_page_config(
    page_title="Dự Đoán Giá Thời Trang Cao Cấp",
    page_icon="💎",
    layout="wide"
)

# ==============================
#  CSS Style
# ==============================
st.markdown("""
<style>
.main { background-color: #f8fdfd; font-family: 'Poppins', sans-serif; color: #222; }
.stButton>button {
    background: linear-gradient(90deg, #81d8d0, #0abab5);
    color: white; font-weight: 600; border-radius: 10px; padding: 10px 20px;
    border: none; transition: 0.3s; letter-spacing: 0.3px;
}
.stButton>button:hover { transform: scale(1.05); background: linear-gradient(90deg, #0abab5, #089c95); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #81d8d0, #0abab5);
    color: white !important; border-radius: 8px; font-weight: 600;
}
[data-testid="stMetricValue"] { color: #0abab5; font-weight: bold; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ==============================
#  Sidebar
# ==============================
st.sidebar.image("images/logo.png", width=180)
st.sidebar.title(" Dự Đoán Giá Thời Trang Cao Cấp")
st.sidebar.write("""
**ĐỀ TÀI:**  
ỨNG DỤNG MACHINE LEARNING DỰ ĐOÁN GIÁ THỜI TRANG THEO MÙA  

**NHÓM THỰC HIỆN:**  
- Tuấn – Data Engineer  
- Minh – EDA  
- Phát – Modeling  
- Đức – Streamlit & Báo cáo  
""")

# ==============================
# 🔧 Load Model & Data
# ==============================
@st.cache_resource
def load_model_and_data():
    model = joblib.load("models/season_price_predict.pkl")
    df = pd.read_csv("data/processed/ssense_clean.csv")
    return model, df

model, df = load_model_and_data()

# ==============================
# 📈 Tabs
# ==============================
tab1, tab2 = st.tabs([" Dự Đoán & Gợi Ý Mùa Xuân", " Phân Tích Giá"])

# ==============================
# TAB 1 – Prediction + Recommendation
# ==============================
with tab1:
    st.header(" Dự đoán & Gợi ý sản phẩm nên mua mùa Xuân")

    col1, col2 = st.columns(2)

    brand = col1.selectbox(
    "Chọn thương hiệu:",
    sorted(df["brand"].unique()),
    key="brand_tab1"
    )

    prod_type = col2.selectbox(
    "Giới tính:",
    sorted(df["type"].unique()),
    key="type_tab1"
    )



    if st.button("Phân Tích & Gợi Ý", use_container_width=True):
        try:
            result_df = pd.read_csv("results/price_forecast.csv")


            # ---- Tính % thay đổi nếu chưa có ----
            if "change_rate" not in result_df.columns:
                result_df["change_rate"] = ((result_df["spring_price"] - result_df["current_price"]) / result_df["current_price"]) * 100

            # ---- Recommendation Logic ----
            def classify(change):
                if change < -30:
                    return " Nên mua"
                elif -30 <= change < -10:
                    return " Cân nhắc"
                elif -10 <= change <= 0:
                    return " Theo dõi thêm"
                else:
                    return " Không nên mua"

            def reason(change):
                if change < -30:
                    return "Giảm sâu"
                elif -30 <= change < -10:
                    return "Giảm nhiều, mua nếu thích."
                elif -10 <= change <= 0:
                    return "Giảm nhẹ, chưa hấp dẫn."
                else:
                    return "Giá tăng, tránh mua."

            result_df["recommendation_level"] = result_df["change_rate"].apply(classify)
            result_df["reason"] = result_df["change_rate"].apply(reason)
            result_df["change_rate_display"] = result_df["change_rate"].map(lambda x: f"{x:.2f}%")

            # ---- Lọc theo Brand + Type ----
            filtered = result_df[(result_df["brand"] == brand) & (result_df["type"] == prod_type)]

            if filtered.empty:
                st.warning("Không có dữ liệu phù hợp.")
            else:
                # ---- Metrics Summary ----
                avg_cur = filtered["current_price"].mean()
                avg_spr = filtered["spring_price"].mean()
                avg_change = ((avg_spr - avg_cur) / avg_cur) * 100
                
                st.metric("Giá hiện tại TB", f"{avg_cur:,.2f} USD")
                st.metric("Giá dự đoán mùa Xuân TB", f"{avg_spr:,.2f} USD")
                st.metric("Mức thay đổi TB", f"{avg_change:.2f}%")

                st.subheader(" Gợi ý mua mùa Xuân")

                display_cols = [
                    "description", "current_price", "spring_price",
                    "change_rate_display", "recommendation_level", "reason"
                ]
                rename_cols = {
                    "description": "Mô tả",
                    "current_price": "Giá hiện tại (USD)",
                    "spring_price": "Giá mùa Xuân (USD)",
                    "change_rate_display": "% thay đổi",
                    "recommendation_level": "Khuyến nghị",
                    "reason": "Giải thích"
                }
                
                st.dataframe(
                    filtered[display_cols].rename(columns=rename_cols),
                    use_container_width=True
                )

        except FileNotFoundError:
            st.error(" Chưa có `price_forecast.csv`. Vui lòng chạy train trước.")

# ==============================
# TAB 2 – Brand Price Similarity
# ==============================
with tab2:
    st.header(" So sánh giá thương hiệu")

    brand_sel = st.selectbox(
    "Chọn thương hiệu:",
    sorted(df["brand"].unique()),
    key="brand_tab2"
    )


    avg_prices = df.groupby("brand")["price_usd"].mean().sort_values(ascending=False)
    
    st.metric(f"Giá trung bình {brand_sel}", f"{avg_prices[brand_sel]:,.2f} USD")

    similar = avg_prices[(avg_prices > avg_prices[brand_sel]*0.7) & (avg_prices < avg_prices[brand_sel]*1.3)].head(5)

    st.subheader("Thương hiệu giá tương đương")
    st.dataframe(similar.reset_index().rename(columns={"brand":"Thương hiệu","price_usd":"Giá trung bình (USD)"}))

# ==============================
# Footer
# ==============================
st.markdown("---")
st.caption("© 2025 | Đồ án Machine Learning – Văn Lang University")
