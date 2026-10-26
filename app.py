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
st.caption(
    "Đánh giá dựa trên tập bình luận có nhãn thực tế do con người xác định độc lập."
)

if os.path.exists(EVAL_PATH):
    try:
        eval_df = pd.read_csv(EVAL_PATH)

        required_eval = ["sentiment", "true_sentiment"]
        if not all(col in eval_df.columns for col in required_eval):
            st.error(
                "File đánh giá cần có hai cột: sentiment và true_sentiment."
            )
        else:
            eval_df = eval_df.dropna(
                subset=["sentiment", "true_sentiment"]
            ).copy()

            y_pred = eval_df["sentiment"].astype(str).str.strip()
            y_true = eval_df["true_sentiment"].astype(str).str.strip()

            if len(eval_df) == 0:
                st.warning("File đánh giá chưa có dữ liệu hợp lệ.")
            else:
                accuracy = accuracy_score(y_true, y_pred)

                m1, m2 = st.columns(2)
                m1.metric("Số bình luận đánh giá", f"{len(eval_df):,}")
                m2.metric("Accuracy", f"{accuracy:.1%}")

                st.markdown("**Báo cáo phân loại**")
                report = classification_report(
                    y_true,
                    y_pred,
                    output_dict=True,
                    zero_division=0
                )

                report_df = pd.DataFrame(report).transpose()
                report_df.index.name = "Nhãn"
                report_df = report_df.reset_index()

                st.dataframe(
                    report_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("**Ma trận nhầm lẫn**")
                labels = ["Negative", "Neutral", "Positive"]
                labels = [
                    label for label in labels
                    if label in set(y_true) | set(y_pred)
                ]

                cm = confusion_matrix(
                    y_true,
                    y_pred,
                    labels=labels
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=[f"Thực tế: {x}" for x in labels],
                    columns=[f"AI dự đoán: {x}" for x in labels]
                )

                st.dataframe(
                    cm_df,
                    use_container_width=True
                )

                wrong = eval_df[y_true != y_pred]
                st.markdown(
                    f"**Số bình luận dự đoán sai: {len(wrong)}**"
                )

                if len(wrong) > 0:
                    display_cols = [
                        col for col in [
                            "review_no", "review",
                            "true_sentiment", "sentiment"
                        ]
                        if col in wrong.columns
                    ]
                    st.dataframe(
                        wrong[display_cols],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.success(
                        "Mô hình không dự đoán sai bình luận nào trong tập đánh giá."
                    )

    except Exception as e:
        st.error(f"Lỗi khi đọc hoặc đánh giá dữ liệu: {e}")

else:
    st.info(
        "Chưa có file customer_reviews_labeled.csv tại "
        "/content/customer_reviews_labeled.csv. "
        "Phần đánh giá sẽ xuất hiện khi bạn lưu file này vào đúng đường dẫn."
    )

st.caption(
    "Lưu ý: kết quả đánh giá chỉ đại diện cho tập dữ liệu có nhãn thực tế. "
    "Chủ đề hiện được phân loại bằng từ khóa nên có thể chưa chính xác với mọi bình luận."
)

