import os
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

st.set_page_config(
    page_title="AI Customer Listening Dashboard",
    page_icon="📊",
    layout="wide"
)

DATA_PATH = "/content/customer_reviews_analyzed.csv"
EVAL_PATH = "/content/customer_reviews_labeled.csv"

st.title("📊 AI Customer Listening Dashboard")
st.caption("Phân tích cảm xúc và chủ đề trong đánh giá của khách hàng")

# Cho phép tải file khác lên hoặc tự dùng file có sẵn
uploaded_file = st.file_uploader(
    "Tải lên file customer_reviews_analyzed.csv (không bắt buộc)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = None

# ==========================================
# 1. DASHBOARD PHÂN TÍCH KHÁCH HÀNG
# ==========================================
if df is not None:
    required_columns = ["review", "sentiment", "topic"]

    if not all(col in df.columns for col in required_columns):
        st.error(
            "File thiếu cột bắt buộc: review, sentiment hoặc topic."
        )
    else:
        df = df.dropna(
            subset=["review", "sentiment", "topic"]
        ).copy()

        df["review"] = df["review"].astype(str)
        df["sentiment"] = df["sentiment"].astype(str)
        df["topic"] = df["topic"].astype(str)

        # Nếu dữ liệu chưa có confidence thì vẫn hiển thị được
        if "confidence" not in df.columns:
            df["confidence"] = None

        total = len(df)
        positive = (df["sentiment"] == "Positive").sum()
        neutral = (df["sentiment"] == "Neutral").sum()
        negative = (df["sentiment"] == "Negative").sum()

        st.success(f"Đã nạp {total:,} bình luận.")

        st.subheader("1. Tổng quan phản hồi")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng bình luận", f"{total:,}")
        c2.metric("Tích cực", f"{positive:,}")
        c3.metric("Trung tính", f"{neutral:,}")
        c4.metric("Tiêu cực", f"{negative:,}")

        st.subheader("2. Phân bố cảm xúc")
        sentiment_counts = (
            df["sentiment"].value_counts()
            .rename_axis("Cảm xúc")
            .reset_index(name="Số bình luận")
        )

        fig1 = px.pie(
            sentiment_counts,
            names="Cảm xúc",
            values="Số bình luận",
            hole=0.4
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("3. Các chủ đề khách hàng đề cập")
        topic_counts = (
            df["topic"].value_counts()
            .rename_axis("Chủ đề")
            .reset_index(name="Số bình luận")
        )

        fig2 = px.bar(
            topic_counts,
            x="Chủ đề",
            y="Số bình luận",
            text="Số bình luận"
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("4. Tra cứu bình luận")
        col1, col2 = st.columns(2)

        with col1:
            sentiments = ["Tất cả"] + sorted(
                df["sentiment"].unique().tolist()
            )
            selected_sentiment = st.selectbox(
                "Lọc theo cảm xúc", sentiments
            )

        with col2:
            topics = ["Tất cả"] + sorted(
                df["topic"].unique().tolist()
            )
            selected_topic = st.selectbox(
                "Lọc theo chủ đề", topics
            )

        filtered = df.copy()

        if selected_sentiment != "Tất cả":
            filtered = filtered[
                filtered["sentiment"] == selected_sentiment
            ]

        if selected_topic != "Tất cả":
            filtered = filtered[
                filtered["topic"] == selected_topic
            ]

        st.write(f"Tìm thấy **{len(filtered):,}** bình luận.")

        st.dataframe(
            filtered[
                ["review", "sentiment", "confidence", "topic"]
            ],
            use_container_width=True,
            hide_index=True
        )

        csv_data = filtered.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "⬇️ Tải kết quả CSV",
            data=csv_data,
            file_name="customer_reviews_filtered.csv",
            mime="text/csv"
        )

else:
    st.warning(
        "Chưa tìm thấy file customer_reviews_analyzed.csv. "
        "Hãy chạy bước phân tích dữ liệu trước hoặc tải file CSV lên."
    )

# ==========================================
# 2. ĐÁNH GIÁ HIỆU QUẢ MÔ HÌNH AI
# ==========================================
st.divider()

st.subheader("5. Đánh giá hiệu quả mô hình AI")

st.write(
    "Đánh giá dựa trên tập bình luận có nhãn thực tế "
    "do con người xác định độc lập."
)

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

evaluation_file = st.file_uploader(
    "Tải file customer_reviews_labeled.csv",
    type=["csv"],
    key="evaluation_file"
)

if evaluation_file is not None:
    eval_df = pd.read_csv(evaluation_file)

    required_columns = ["sentiment", "true_sentiment"]

    if all(col in eval_df.columns for col in required_columns):
        y_pred = eval_df["sentiment"].astype(str).str.strip()
        y_true = eval_df["true_sentiment"].astype(str).str.strip()

        valid = y_true.notna() & y_pred.notna()
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        if len(y_true) > 0:
            st.metric(
                "Accuracy",
                f"{accuracy_score(y_true, y_pred):.1%}"
            )

            st.write("### Báo cáo phân loại")
            report = classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0
            )
            st.dataframe(pd.DataFrame(report).transpose())

            st.write("### Ma trận nhầm lẫn")
            labels = sorted(set(y_true) | set(y_pred))
            matrix = confusion_matrix(
                y_true,
                y_pred,
                labels=labels
            )
            st.dataframe(
                pd.DataFrame(
                    matrix,
                    index=[f"Thực tế: {x}" for x in labels],
                    columns=[f"Dự đoán: {x}" for x in labels]
                )
            )
        else:
            st.warning("File chưa có dữ liệu đánh giá hợp lệ.")
    else:
        st.error(
            "File cần có hai cột sentiment và true_sentiment."
        )
else:
    st.info("Hãy tải file customer_reviews_labeled.csv để xem kết quả đánh giá.")
