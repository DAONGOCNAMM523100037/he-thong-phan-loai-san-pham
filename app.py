
from pathlib import Path
from io import BytesIO
import json

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from rembg import remove, new_session


# =====================================================
# 1. CAU HINH
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "models" / "product_classifier.keras"
LABEL_PATH = PROJECT_DIR / "models" / "class_names.json"

IMG_SIZE = (224, 224)

CLASS_DISPLAY_NAMES = {
    "nuoc_uong": "Nước uống đóng chai",
    "ca_phe": "Cà phê đóng hộp",
    "thuc_an": "Thức ăn đóng hộp",
    "mi_an_lien": "Mỳ ăn liền"
}


# =====================================================
# 2. CAU HINH GIAO DIEN
# =====================================================

st.set_page_config(
    page_title="Hệ thống phân loại sản phẩm",
    page_icon="📦",
    layout="wide"
)

st.title("📦 HỆ THỐNG PHÂN LOẠI SẢN PHẨM")
st.write(
    "Ứng dụng sử dụng CNN MobileNetV2 để nhận diện "
    "4 nhóm sản phẩm sau khi tự động tách nền ảnh."
)

st.divider()


# =====================================================
# 3. KIEM TRA FILE MO HINH
# =====================================================

if not MODEL_PATH.exists():
    st.error(
        "Không tìm thấy mô hình! "
        "Vui lòng kiểm tra file models/product_classifier.keras."
    )
    st.stop()

if not LABEL_PATH.exists():
    st.error(
        "Không tìm thấy file class_names.json trong thư mục models."
    )
    st.stop()


# =====================================================
# 4. TAI MO HINH
# =====================================================

@st.cache_resource
def load_product_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_background_session():
    return new_session("u2net")


try:
    with st.spinner("Đang tải mô hình nhận diện..."):
        model = load_product_model()

    with st.spinner("Đang khởi tạo hệ thống tách nền..."):
        background_session = load_background_session()

    with open(LABEL_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

except Exception as error:
    st.error(f"Không thể khởi tạo hệ thống: {error}")
    st.info(
        "Nếu đây là lần đầu chạy, hãy kiểm tra kết nối Internet "
        "để tải mô hình tách nền."
    )
    st.stop()


# =====================================================
# 5. HAM TACH NEN
# =====================================================

def remove_background(image):
    """
    Nhan anh PIL, tra ve anh PNG da tach nen.
    """

    image_buffer = BytesIO()

    image.save(
        image_buffer,
        format="PNG"
    )

    input_bytes = image_buffer.getvalue()

    output_bytes = remove(
        input_bytes,
        session=background_session
    )

    output_image = Image.open(
        BytesIO(output_bytes)
    ).convert("RGBA")

    return output_image


# =====================================================
# 6. CHUAN BI ANH CHO MOBILENETV2
# =====================================================


def prepare_image(image):
    """
    Chuyen anh sang RGB, resize 224x224.
    Giu nguyen pixel 0-255 vi mo hinh da co lop Rescaling.
    """

    image = image.convert("RGBA")

    # Tao nen trang de thay the phan trong suot
    white_background = Image.new(
        "RGBA",
        image.size,
        (255, 255, 255, 255)
    )

    white_background.alpha_composite(image)

    image = white_background.convert("RGB")

    image = image.resize(
        IMG_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # KHONG chuan hoa tai day.
    # Mo hinh se tu chuan hoa bang lop Rescaling.

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array



# =====================================================
# 7. DU DOAN SAN PHAM
# =====================================================

def predict_product(image):
    image_array = prepare_image(image)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_names[predicted_index]

    confidence = float(
        predictions[predicted_index]
    )

    return predicted_class, confidence, predictions


# =====================================================
# 8. GIAO DIEN TAI ANH
# =====================================================

st.subheader("📸 Tải ảnh sản phẩm")

uploaded_file = st.file_uploader(
    "Chọn ảnh sản phẩm từ máy tính",
    type=["jpg", "jpeg", "png", "webp"],
    help="Hỗ trợ các định dạng JPG, JPEG, PNG và WEBP."
)

if uploaded_file is not None:

    try:
        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.subheader("Ảnh sản phẩm ban đầu")

        st.image(
            original_image,
            caption="Ảnh gốc",
            use_container_width=True
        )

        if st.button(
            "🔍 TÁCH NỀN VÀ PHÂN LOẠI",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Đang tách nền ảnh..."):
                processed_image = remove_background(
                    original_image
                )

            with st.spinner("Đang nhận diện sản phẩm..."):
                predicted_class, confidence, predictions = (
                    predict_product(processed_image)
                )

            st.divider()
            st.subheader("📷 Kết quả xử lý ảnh")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    original_image,
                    caption="Ảnh ban đầu",
                    use_container_width=True
                )

            with col2:
                st.image(
                    processed_image,
                    caption="Ảnh đã tách nền",
                    use_container_width=True
                )

            # Hien thi ket qua
            display_name = CLASS_DISPLAY_NAMES.get(
                predicted_class,
                predicted_class
            )

            st.divider()
            st.subheader("🎯 KẾT QUẢ PHÂN LOẠI")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.success(
                    f"Nhóm sản phẩm: {display_name}"
                )

            with result_col2:
                st.metric(
                    "Điểm tin cậy dự đoán",
                    f"{confidence * 100:.2f}%"
                )

            if confidence < 0.60:
                st.warning(
                    "Mô hình chưa đủ tự tin với kết quả này. "
                    "Em nên thử ảnh rõ nét hơn, chụp sản phẩm "
                    "ở góc khác hoặc kiểm tra lại nhóm sản phẩm."
                )

            st.subheader("📊 Điểm dự đoán của từng nhóm")

            score_data = {
                CLASS_DISPLAY_NAMES.get(name, name): float(score)
                for name, score in zip(class_names, predictions)
            }

            st.bar_chart(score_data)

            # Tai anh da tach nen
            output_buffer = BytesIO()

            processed_image.save(
                output_buffer,
                format="PNG"
            )

            st.download_button(
                label="⬇️ Tải ảnh đã tách nền",
                data=output_buffer.getvalue(),
                file_name="anh_da_tach_nen.png",
                mime="image/png"
            )

    except Exception as error:
        st.error(
            f"Đã xảy ra lỗi khi xử lý ảnh: {error}"
        )

else:
    st.info(
        "👆 Hãy tải một ảnh sản phẩm lên để bắt đầu nhận diện."
    )


# =====================================================
# 9. THONG TIN HE THONG
# =====================================================

st.divider()

with st.expander("ℹ️ Thông tin về hệ thống"):

    st.write("**Mô hình phân loại:** CNN MobileNetV2")

    st.write("**Phương pháp tách nền:** rembg (U²-Net)")

    st.write("**Kích thước ảnh đầu vào:** 224 × 224 pixel")

    st.write("**Các nhóm sản phẩm:**")

    for index, name in enumerate(class_names, start=1):
        st.write(
            f"{index}. {CLASS_DISPLAY_NAMES.get(name, name)}"
        )

st.caption(
    "Đề tài: Hệ thống phân loại sản phẩm bằng học sâu CNN"
)
