
from pathlib import Path
from io import BytesIO
import json

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from rembg import remove, new_session
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment


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

st.set_page_config(
    page_title="Phân loại sản phẩm AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. GIAO DIEN
# =====================================================

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: bold;
    color: #1565C0;
    padding: 10px;
}

.sub-title {
    text-align: center;
    color: #666;
    font-size: 16px;
    margin-bottom: 20px;
}

.result-card {
    padding: 15px;
    border-radius: 12px;
    background-color: #E8F5E9;
}

.stButton button {
    border-radius: 10px;
    font-weight: bold;
    min-height: 42px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">📦 HỆ THỐNG PHÂN LOẠI SẢN PHẨM</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'CNN MobileNetV2 | Tách nền U²-Net | Streamlit'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# 3. KIEM TRA FILE
# =====================================================

if not MODEL_PATH.exists():
    st.error("Không tìm thấy models/product_classifier.keras")
    st.stop()

if not LABEL_PATH.exists():
    st.error("Không tìm thấy models/class_names.json")
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
    with st.spinner("Đang khởi tạo hệ thống AI..."):
        model = load_product_model()
        background_session = load_background_session()

    with open(LABEL_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

except Exception as error:
    st.error(f"Không thể tải hệ thống: {error}")
    st.stop()


# =====================================================
# 5. CAC HAM XU LY ANH
# =====================================================

def remove_background(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    output_bytes = remove(
        buffer.getvalue(),
        session=background_session
    )

    return Image.open(
        BytesIO(output_bytes)
    ).convert("RGBA")


def prepare_image(image):

    image = image.convert("RGBA")

    # Thay phan trong suot bang nen trang
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

    # Khong chuan hoa tai day.
    # Lop Rescaling trong mo hinh se tu xu ly.

    return np.expand_dims(image_array, axis=0)


def predict_product(image):

    image_array = prepare_image(image)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    index = int(np.argmax(predictions))

    return (
        class_names[index],
        float(predictions[index]),
        predictions
    )


def process_image(image, filename):

    # Tach nen
    processed_image = remove_background(image)

    # Phan loai
    predicted_class, confidence, predictions = predict_product(
        processed_image
    )

    display_name = CLASS_DISPLAY_NAMES.get(
        predicted_class,
        predicted_class
    )

    score_data = {
        CLASS_DISPLAY_NAMES.get(name, name): float(score)
        for name, score in zip(class_names, predictions)
    }

    return {
        "filename": filename,
        "original": image.copy(),
        "processed": processed_image,
        "class": predicted_class,
        "display_name": display_name,
        "confidence": confidence,
        "scores": score_data
    }


# =====================================================
# 6. HIEN THI KET QUA MOT ANH
# =====================================================

def show_result(result):

    st.markdown(f"### 🖼️ {result['filename']}")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            result["original"],
            caption="Ảnh ban đầu",
            use_container_width=True
        )

    with col2:
        st.image(
            result["processed"],
            caption="Ảnh đã tách nền",
            use_container_width=True
        )

    c1, c2 = st.columns(2)

    with c1:
        st.success(
            f"Nhóm sản phẩm: {result['display_name']}"
        )

    with c2:
        st.metric(
            "Độ tin cậy dự đoán",
            f"{result['confidence'] * 100:.2f}%"
        )

    if result["confidence"] < 0.60:
        st.warning(
            "Mô hình chưa đủ tự tin với kết quả này. "
            "Hãy thử chụp ảnh rõ nét hơn."
        )

    st.bar_chart(result["scores"])

    # Tai anh da tach nen
    image_buffer = BytesIO()

    result["processed"].save(
        image_buffer,
        format="PNG"
    )

    st.download_button(
        "⬇️ Tải ảnh đã tách nền",
        data=image_buffer.getvalue(),
        file_name=f"tach_nen_{result['filename']}.png",
        mime="image/png",
        key=f"download_{result['filename']}_{id(result)}"
    )

    st.divider()


# =====================================================
# 7. XUAT BAO CAO EXCEL
# =====================================================

def create_excel_report(results):

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Ket qua phan loai"

    headers = [
        "STT",
        "Ten anh",
        "Nhom san pham",
        "Do tin cay (%)",
        "Trang thai"
    ]

    sheet.append(headers)

    # Dinh dang tieu de
    for cell in sheet[1]:
        cell.font = Font(
            bold=True,
            color="FFFFFF"
        )

        cell.fill = PatternFill(
            start_color="1565C0",
            end_color="1565C0",
            fill_type="solid"
        )

        cell.alignment = Alignment(
            horizontal="center"
        )

    # Them du lieu
    for index, result in enumerate(results, start=1):

        sheet.append([
            index,
            result["filename"],
            result["display_name"],
            round(result["confidence"] * 100, 2),
            "Thành công"
        ])

    # Dieu chinh do rong cot
    widths = {
        "A": 8,
        "B": 35,
        "C": 30,
        "D": 20,
        "E": 20
    }

    for column, width in widths.items():
        sheet.column_dimensions[column].width = width

    # Can giua du lieu
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center"
            )

    # Them thong ke
    summary = workbook.create_sheet("Thong ke")

    summary.append(["NHÓM SẢN PHẨM", "SỐ LƯỢNG"])

    for name in CLASS_DISPLAY_NAMES.values():

        count = sum(
            1 for result in results
            if result["display_name"] == name
        )

        summary.append([name, count])

    summary.append([])
    summary.append(["TỔNG SỐ ẢNH", len(results)])

    summary.column_dimensions["A"].width = 30
    summary.column_dimensions["B"].width = 20

    for cell in summary[1]:
        cell.font = Font(
            bold=True,
            color="FFFFFF"
        )
        cell.fill = PatternFill(
            start_color="1565C0",
            end_color="1565C0",
            fill_type="solid"
        )

    output = BytesIO()
    workbook.save(output)
    output.seek(0)

    return output.getvalue()


# =====================================================
# 8. SIDEBAR
# =====================================================

with st.sidebar:

    st.title("📦 MENU HỆ THỐNG")

    st.write("**Công nghệ sử dụng**")
    st.write("🧠 CNN MobileNetV2")
    st.write("✂️ U²-Net")
    st.write("🖥️ Streamlit")

    st.divider()

    st.subheader("📋 Các nhóm sản phẩm")

    for index, name in enumerate(class_names, start=1):

        st.write(
            f"{index}. {CLASS_DISPLAY_NAMES.get(name, name)}"
        )

    st.divider()

    st.caption("Phiên bản 3.0")


# =====================================================
# 9. KHO LUU KET QUA
# =====================================================

if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

if "camera_result" not in st.session_state:
    st.session_state.camera_result = None


# =====================================================
# 10. CAC TAB CHINH
# =====================================================

tab_upload, tab_camera, tab_report = st.tabs([
    "📁 TẢI NHIỀU ẢNH",
    "📷 CAMERA",
    "📊 BÁO CÁO"
])


# =====================================================
# TAB 1: NHAN DIEN NHIEU ANH
# =====================================================

with tab_upload:

    st.subheader("📁 Tải nhiều ảnh cùng lúc")

    uploaded_files = st.file_uploader(
        "Chọn ảnh sản phẩm",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="upload_ver3"
    )

    if uploaded_files:

        st.success(
            f"Đã chọn {len(uploaded_files)} ảnh."
        )

        preview_cols = st.columns(4)

        for index, uploaded_file in enumerate(uploaded_files):

            try:
                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                with preview_cols[index % 4]:
                    st.image(
                        image,
                        use_container_width=True
                    )

                    st.caption(uploaded_file.name)

            except Exception:
                st.warning(
                    f"Không đọc được ảnh {uploaded_file.name}"
                )

        if st.button(
            "🚀 BẮT ĐẦU PHÂN LOẠI TẤT CẢ",
            type="primary",
            use_container_width=True,
            key="process_batch"
        ):

            st.session_state.batch_results = []

            progress = st.progress(0)

            for index, uploaded_file in enumerate(uploaded_files):

                st.write(
                    f"Đang xử lý {index + 1}/{len(uploaded_files)}: "
                    f"{uploaded_file.name}"
                )

                try:

                    image = Image.open(
                        uploaded_file
                    ).convert("RGB")

                    result = process_image(
                        image,
                        uploaded_file.name
                    )

                    st.session_state.batch_results.append(result)

                except Exception as error:

                    st.error(
                        f"Lỗi ảnh {uploaded_file.name}: {error}"
                    )

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            st.success("Đã hoàn thành quá trình xử lý!")

    # Hien thi ket qua da luu
    if st.session_state.batch_results:

        st.divider()
        st.subheader("📊 KẾT QUẢ CÁC ẢNH")

        for result in st.session_state.batch_results:
            show_result(result)

        st.success(
            f"Đã phân loại thành công "
            f"{len(st.session_state.batch_results)} ảnh."
        )


# =====================================================
# TAB 2: CAMERA
# =====================================================

with tab_camera:

    st.subheader("📷 Nhận diện bằng camera")

    st.info(
        "Đưa sản phẩm vào giữa khung hình, "
        "đảm bảo đủ ánh sáng rồi chụp ảnh."
    )

    camera_image = st.camera_input(
        "Camera sản phẩm",
        key="camera_ver3"
    )

    if camera_image is not None:

        if st.button(
            "🤖 NHẬN DIỆN ẢNH VỪA CHỤP",
            type="primary",
            use_container_width=True,
            key="process_camera"
        ):

            try:

                image = Image.open(
                    camera_image
                ).convert("RGB")

                with st.spinner("Đang xử lý ảnh camera..."):

                    result = process_image(
                        image,
                        "anh_chup_camera"
                    )

                st.session_state.camera_result = result

                st.success("Nhận diện thành công!")

            except Exception as error:

                st.error(
                    f"Lỗi xử lý camera: {error}"
                )

    if st.session_state.camera_result:

        show_result(
            st.session_state.camera_result
        )


# =====================================================
# TAB 3: BAO CAO
# =====================================================

with tab_report:

    st.subheader("📊 BÁO CÁO KẾT QUẢ")

    results = st.session_state.batch_results

    if results:

        total = len(results)

        st.metric(
            "Tổng số ảnh đã phân loại",
            total
        )

        # Thong ke tung nhom
        summary_data = {}

        for name in CLASS_DISPLAY_NAMES.values():

            count = sum(
                1 for result in results
                if result["display_name"] == name
            )

            summary_data[name] = count

        st.bar_chart(summary_data)

        # Bang ket qua
        table_data = []

        for result in results:

            table_data.append({
                "Tên ảnh": result["filename"],
                "Nhóm sản phẩm": result["display_name"],
                "Độ tin cậy (%)": round(
                    result["confidence"] * 100,
                    2
                )
            })

        st.dataframe(
            table_data,
            use_container_width=True
        )

        # Xuat Excel
        excel_data = create_excel_report(results)

        st.download_button(
            "📥 XUẤT BÁO CÁO EXCEL",
            data=excel_data,
            file_name="bao_cao_phan_loai_san_pham.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )

        if st.button(
            "🗑️ XÓA KẾT QUẢ ĐỢT PHÂN LOẠI",
            use_container_width=True
        ):

            st.session_state.batch_results = []
            st.rerun()

    else:

        st.info(
            "Chưa có kết quả phân loại hàng loạt. "
            "Hãy sang tab Tải nhiều ảnh để bắt đầu."
        )


# =====================================================
# 11. CHAN TRANG
# =====================================================

st.divider()

st.markdown(
    "<center>HỆ THỐNG PHÂN LOẠI SẢN PHẨM "
    "| CNN MobileNetV2 + U²-Net</center>",
    unsafe_allow_html=True
)
