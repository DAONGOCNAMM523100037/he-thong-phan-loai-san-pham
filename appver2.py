
from pathlib import Path
from io import BytesIO
import json

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from rembg import remove, new_session


# =====================================================
# 1. CAU HINH DU AN
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
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1565C0;
        font-size: 34px;
        font-weight: bold;
        margin-bottom: 8px;
    }

    .sub-title {
        text-align: center;
        color: #666666;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #E8F5E9;
        margin-top: 10px;
    }

    .stButton button {
        border-radius: 10px;
        font-weight: bold;
        min-height: 45px;
    }
</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="main-title">📦 HỆ THỐNG PHÂN LOẠI SẢN PHẨM</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Ứng dụng nhận diện sản phẩm bằng CNN MobileNetV2 '
    'kết hợp công nghệ tự động tách nền'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# 3. KIEM TRA MO HINH
# =====================================================

if not MODEL_PATH.exists():
    st.error(
        "Không tìm thấy mô hình product_classifier.keras "
        "trong thư mục models."
    )
    st.stop()

if not LABEL_PATH.exists():
    st.error(
        "Không tìm thấy file class_names.json "
        "trong thư mục models."
    )
    st.stop()


# =====================================================
# 4. TAI MO HINH CNN
# =====================================================

@st.cache_resource
def load_product_model():
    return tf.keras.models.load_model(MODEL_PATH)


# =====================================================
# 5. TAI MO HINH TACH NEN
# =====================================================

@st.cache_resource
def load_background_session():
    return new_session("u2net")


try:
    with st.spinner("Đang khởi tạo hệ thống AI..."):
        model = load_product_model()
        background_session = load_background_session()

    with open(LABEL_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

except Exception as error:
    st.error(f"Không thể khởi tạo hệ thống: {error}")
    st.info(
        "Nếu đây là lần đầu chạy, hãy kiểm tra kết nối Internet "
        "để tải mô hình tách nền U2-Net."
    )
    st.stop()


# =====================================================
# 6. HAM TACH NEN
# =====================================================

def remove_background(image):

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
# 7. CHUAN BI ANH
# =====================================================

def prepare_image(image):

    image = image.convert("RGBA")

    # Thay nen trong suot bang nen trang
    white_background = Image.new(
        "RGBA",
        image.size,
        (255, 255, 255, 255)
    )

    white_background.alpha_composite(image)

    image = white_background.convert("RGB")

    # Resize anh
    image = image.resize(
        IMG_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Khong chuan hoa tai day.
    # Lop Rescaling trong mo hinh se tu xu ly.

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =====================================================
# 8. DU DOAN SAN PHAM
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
# 9. HAM HIEN THI KET QUA
# =====================================================

def display_result(original_image):

    try:

        # Tach nen
        with st.spinner("✂️ Đang tự động tách nền..."):

            processed_image = remove_background(
                original_image
            )

        # Phan loai
        with st.spinner("🧠 AI đang nhận diện sản phẩm..."):

            predicted_class, confidence, predictions = (
                predict_product(processed_image)
            )

        display_name = CLASS_DISPLAY_NAMES.get(
            predicted_class,
            predicted_class
        )

        st.divider()

        st.subheader("📷 KẾT QUẢ XỬ LÝ ẢNH")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                original_image,
                caption="Ảnh sản phẩm ban đầu",
                use_container_width=True
            )

        with col2:
            st.image(
                processed_image,
                caption="Ảnh sản phẩm đã tách nền",
                use_container_width=True
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
                "Độ tin cậy dự đoán",
                f"{confidence * 100:.2f}%"
            )

        if confidence < 0.60:
            st.warning(
                "Mô hình chưa đủ tự tin với kết quả này. "
                "Hãy thử chụp ảnh rõ hơn, đủ sáng và "
                "để sản phẩm nằm trọn trong khung hình."
            )

        st.subheader("📊 ĐIỂM DỰ ĐOÁN CỦA TỪNG NHÓM")

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
            label="⬇️ TẢI ẢNH ĐÃ TÁCH NỀN",
            data=output_buffer.getvalue(),
            file_name="anh_da_tach_nen.png",
            mime="image/png"
        )

    except Exception as error:

        st.error(
            f"Đã xảy ra lỗi khi xử lý ảnh: {error}"
        )


# =====================================================
# 10. SIDEBAR
# =====================================================

with st.sidebar:

    st.title("📦 MENU HỆ THỐNG")

    st.write("**Công nghệ sử dụng:**")

    st.write("🧠 CNN MobileNetV2")

    st.write("✂️ Tách nền U²-Net")

    st.write("🖥️ Giao diện Streamlit")

    st.divider()

    st.subheader("📋 Các nhóm sản phẩm")

    for index, name in enumerate(class_names, start=1):

        display_name = CLASS_DISPLAY_NAMES.get(
            name,
            name
        )

        st.write(f"{index}. {display_name}")

    st.divider()

    st.caption(
        "Đề tài: Hệ thống phân loại sản phẩm "
        "bằng học sâu CNN"
    )


# =====================================================
# 11. GIAO DIEN NHAN DIEN
# =====================================================

tab_upload, tab_camera = st.tabs([
    "📁 TẢI ẢNH TỪ MÁY TÍNH",
    "📷 NHẬN DIỆN BẰNG CAMERA"
])


# -----------------------------------------------------
# TAB 1: UPLOAD ANH
# -----------------------------------------------------

with tab_upload:

    st.subheader("📁 Tải ảnh sản phẩm")

    st.write(
        "Chọn ảnh sản phẩm có sẵn trong máy tính "
        "để hệ thống phân loại."
    )

    uploaded_file = st.file_uploader(
        "Chọn ảnh sản phẩm",
        type=["jpg", "jpeg", "png", "webp"],
        key="upload_ver2"
    )

    if uploaded_file is not None:

        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            original_image,
            caption="Ảnh đã chọn",
            use_container_width=True
        )

        if st.button(
            "🔍 TÁCH NỀN VÀ PHÂN LOẠI",
            key="upload_button_ver2",
            type="primary",
            use_container_width=True
        ):

            display_result(original_image)


# -----------------------------------------------------
# TAB 2: CAMERA
# -----------------------------------------------------

with tab_camera:

    st.subheader("📷 Nhận diện sản phẩm bằng camera")

    st.info(
        "Hãy đưa sản phẩm vào giữa khung hình, "
        "đảm bảo đủ ánh sáng và nhấn nút chụp."
    )

    camera_image = st.camera_input(
        "📸 CAMERA SẢN PHẨM",
        key="camera_ver2"
    )

    if camera_image is not None:

        original_image = Image.open(
            camera_image
        ).convert("RGB")

        if st.button(
            "🤖 NHẬN DIỆN SẢN PHẨM",
            key="camera_button_ver2",
            type="primary",
            use_container_width=True
        ):

            display_result(original_image)


# =====================================================
# 12. THONG TIN CUOI TRANG
# =====================================================

st.divider()

st.markdown(
    "<center>📦 Hệ thống phân loại sản phẩm bằng CNN "
    "| MobileNetV2 + U²-Net</center>",
    unsafe_allow_html=True
)
