
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
    page_title="Product AI | Phân loại sản phẩm",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# 2. GIAO DIEN
# =====================================================

st.markdown("""
<style>

/* An cac thanh khong can thiet */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

[data-testid="stSidebar"] {
    display: none;
}

/* Nen tong the */
.stApp {
    background: #F5F8FC;
}

/* Gioi han chieu rong noi dung */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Tieu de */
.hero {
    background: linear-gradient(120deg, #155EEF, #3787F5);
    padding: 35px 40px;
    border-radius: 20px;
    margin-bottom: 25px;
    color: white;
}

.hero-title {
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 8px;
    color: white;
}

.hero-subtitle {
    font-size: 15px;
    color: #E5EEFF;
    margin: 0;
}

/* Tieu de khu vuc */
.section-title {
    font-size: 22px;
    font-weight: 750;
    color: #172B4D;
    margin-top: 10px;
    margin-bottom: 5px;
}

.section-description {
    color: #718096;
    font-size: 14px;
    margin-bottom: 20px;
}

/* Tab */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    border-bottom: 1px solid #E2E8F0;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding: 0 22px;
    border-radius: 10px 10px 0 0;
    font-weight: 650;
    color: #52627A;
}

.stTabs [aria-selected="true"] {
    color: #155EEF !important;
}

/* Nut bam */
.stButton > button {
    border-radius: 10px;
    min-height: 44px;
    font-weight: 650;
    border: 1px solid #DCE5F0;
    transition: 0.2s;
}

.stButton > button[kind="primary"] {
    background: #155EEF;
    border: none;
    color: white;
}

.stButton > button:hover {
    border-color: #155EEF;
    color: #155EEF;
}

/* Nut primary khi hover */
.stButton > button[kind="primary"]:hover {
    background: #1049C7;
    color: white;
}

/* Upload */
[data-testid="stFileUploader"] {
    background: white;
    border: 1.5px dashed #B9CBE5;
    border-radius: 15px;
    padding: 15px;
}

/* Khung ket qua */
.result-heading {
    color: #172B4D;
    font-size: 19px;
    font-weight: 750;
    margin-bottom: 15px;
}

.result-label {
    color: #64748B;
    font-size: 13px;
    margin-bottom: 5px;
}

.result-name {
    color: #155EEF;
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 8px;
}

.confidence-text {
    color: #16845B;
    font-size: 15px;
    font-weight: 700;
}

/* Metric */
[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 13px;
    border: 1px solid #E6EDF5;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #E6EDF5;
    border-radius: 12px;
}

/* Divider */
hr {
    border-color: #E6EDF5;
}

/* Chan trang */
.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 13px;
    padding: 25px 0 5px 0;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">HỆ THỐNG PHÂN LOẠI SẢN PHẨM</div>
    <p class="hero-subtitle">
        Thị giác máy tính
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 4. KIEM TRA FILE
# =====================================================

if not MODEL_PATH.exists():
    st.error("Không tìm thấy file mô hình: models/product_classifier.keras")
    st.stop()

if not LABEL_PATH.exists():
    st.error("Không tìm thấy file nhãn: models/class_names.json")
    st.stop()


# =====================================================
# 5. TAI MO HINH
# =====================================================

@st.cache_resource
def load_product_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_background_session():
    return new_session("u2net")


try:
    with st.spinner("Đang khởi tạo hệ thống..."):
        model = load_product_model()
        background_session = load_background_session()

    with open(LABEL_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

except Exception as error:
    st.error(f"Không thể khởi tạo hệ thống: {error}")
    st.stop()


# =====================================================
# 6. KHO LUU KET QUA
# =====================================================

if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

if "camera_result" not in st.session_state:
    st.session_state.camera_result = None


# =====================================================
# 7. XU LY ANH
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

    # Model da co lop Rescaling, khong chuan hoa lan nua.

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

    processed_image = remove_background(image)

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
# 8. HIEN THI KET QUA
# =====================================================

def show_result(result):

    with st.container(border=True):

        st.markdown(
            f'<div class="result-heading">🖼️ {result["filename"]}</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                result["original"],
                caption="Ảnh gốc",
                use_container_width=True
            )

        with col2:
            st.image(
                result["processed"],
                caption="Ảnh sau khi tách nền",
                use_container_width=True
            )

        st.markdown(
            '<div class="result-label">KẾT QUẢ NHẬN DIỆN</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-name">{result["display_name"]}</div>',
            unsafe_allow_html=True
        )

        confidence_percent = result["confidence"] * 100

        st.markdown(
            f'<div class="confidence-text">Độ tin cậy: {confidence_percent:.2f}%</div>',
            unsafe_allow_html=True
        )

        st.progress(
            result["confidence"]
        )

        if result["confidence"] < 0.60:
            st.warning(
                "Độ tin cậy còn thấp. Em nên thử ảnh rõ nét hơn, "
                "đủ sáng và tập trung vào sản phẩm."
            )

        with st.expander("Xem xác suất của các nhóm sản phẩm"):

            st.bar_chart(
                result["scores"],
                horizontal=True
            )

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
            key=f"download_{id(result)}"
        )


# =====================================================
# 9. XUAT BAO CAO EXCEL
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

    for cell in sheet[1]:

        cell.font = Font(
            bold=True,
            color="FFFFFF"
        )

        cell.fill = PatternFill(
            start_color="155EEF",
            end_color="155EEF",
            fill_type="solid"
        )

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center"
        )

    for index, result in enumerate(results, start=1):

        sheet.append([
            index,
            result["filename"],
            result["display_name"],
            round(result["confidence"] * 100, 2),
            "Thành công"
        ])

    widths = {
        "A": 8,
        "B": 35,
        "C": 30,
        "D": 20,
        "E": 20
    }

    for column, width in widths.items():
        sheet.column_dimensions[column].width = width

    for row in sheet.iter_rows(min_row=2):

        for cell in row:
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center"
            )

    summary = workbook.create_sheet("Thong ke")

    summary.append([
        "NHÓM SẢN PHẨM",
        "SỐ LƯỢNG"
    ])

    for name in CLASS_DISPLAY_NAMES.values():

        count = sum(
            1 for result in results
            if result["display_name"] == name
        )

        summary.append([name, count])

    summary.append([])
    summary.append(["TỔNG SỐ ẢNH", len(results)])

    summary.column_dimensions["A"].width = 32
    summary.column_dimensions["B"].width = 20

    for cell in summary[1]:

        cell.font = Font(
            bold=True,
            color="FFFFFF"
        )

        cell.fill = PatternFill(
            start_color="155EEF",
            end_color="155EEF",
            fill_type="solid"
        )

    output = BytesIO()

    workbook.save(output)
    output.seek(0)

    return output.getvalue()


# =====================================================
# 10. CAC TAB CHINH
# =====================================================

tab_upload, tab_camera, tab_report = st.tabs([
    "📁  Tải ảnh",
    "📷  Camera",
    "📊  Báo cáo"
])


# =====================================================
# TAB 1: TAI NHIEU ANH
# =====================================================

with tab_upload:

    st.markdown(
        '<div class="section-title">Tải lên một hoặc nhiều ảnh để hệ thống tự động nhận diện</div>',
        unsafe_allow_html=True
    )

   

    uploaded_files = st.file_uploader(
        "Chọn ảnh sản phẩm",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="upload_ver3"
    )

    if uploaded_files:

        st.info(
            f"Đã chọn {len(uploaded_files)} ảnh."
        )

        if st.button(
            "🚀 BẮT ĐẦU PHÂN LOẠI",
            type="primary",
            use_container_width=True,
            key="process_batch"
        ):

            st.session_state.batch_results = []

            progress = st.progress(0)
            status = st.empty()

            for index, uploaded_file in enumerate(uploaded_files):

                status.write(
                    f"Đang xử lý ảnh {index + 1}/{len(uploaded_files)}..."
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
                        f"Không thể xử lý {uploaded_file.name}: {error}"
                    )

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            status.empty()

            st.success(
                f"Hoàn thành! Đã phân loại thành công "
                f"{len(st.session_state.batch_results)}/{len(uploaded_files)} ảnh."
            )

    if st.session_state.batch_results:

        st.divider()

        st.markdown(
            '<div class="section-title">Kết quả phân loại</div>',
            unsafe_allow_html=True
        )

        st.caption(
            f"{len(st.session_state.batch_results)} ảnh đã được xử lý thành công."
        )

        for result in st.session_state.batch_results:
            show_result(result)

        if st.button(
            "🗑️ Xóa kết quả",
            key="clear_batch"
        ):

            st.session_state.batch_results = []
            st.rerun()


# =====================================================
# TAB 2: CAMERA
# =====================================================

with tab_camera:

    st.markdown(
        '<div class="section-title">Nhận diện sản phẩm bằng camera</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Chụp ảnh sản phẩm trực tiếp bằng camera của thiết bị.'
        '</div>',
        unsafe_allow_html=True
    )

    camera_col1, camera_col2 = st.columns([1, 1])

    with camera_col1:

        camera_image = st.camera_input(
            "Chụp ảnh sản phẩm",
            key="camera_ver3"
        )

    with camera_col2:

        if camera_image is None:

            st.markdown("""
            <div style="
                background: white;
                border: 1px solid #E6EDF5;
                border-radius: 15px;
                padding: 25px;
                margin-top: 10px;
            ">
                <h3 style="color:#172B4D;">📷 Hướng dẫn chụp ảnh</h3>
                <p style="color:#64748B;">• Đặt sản phẩm ở giữa khung hình.</p>
                <p style="color:#64748B;">• Đảm bảo ánh sáng đầy đủ.</p>
                <p style="color:#64748B;">• Tránh để sản phẩm bị che khuất.</p>
                <p style="color:#64748B;">• Chụp rõ nhãn và hình dáng sản phẩm.</p>
            </div>
            """, unsafe_allow_html=True)

    if camera_image is not None:

        if st.button(
            "🤖 NHẬN DIỆN SẢN PHẨM",
            type="primary",
            use_container_width=True,
            key="process_camera"
        ):

            try:

                image = Image.open(
                    camera_image
                ).convert("RGB")

                with st.spinner("AI đang phân tích ảnh..."):

                    result = process_image(
                        image,
                        "anh_chup_camera"
                    )

                st.session_state.camera_result = result

                st.success("Nhận diện thành công!")

            except Exception as error:

                st.error(
                    f"Lỗi xử lý ảnh camera: {error}"
                )

    if st.session_state.camera_result:

        st.divider()

        st.markdown(
            '<div class="section-title">Kết quả nhận diện</div>',
            unsafe_allow_html=True
        )

        show_result(
            st.session_state.camera_result
        )


# =====================================================
# TAB 3: BAO CAO
# =====================================================

with tab_report:

    st.markdown(
        '<div class="section-title">Báo cáo phân loại sản phẩm</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Tổng hợp kết quả từ các ảnh đã phân loại hàng loạt.'
        '</div>',
        unsafe_allow_html=True
    )

    results = st.session_state.batch_results

    if results:

        total = len(results)

        summary_data = {}

        for name in CLASS_DISPLAY_NAMES.values():

            count = sum(
                1 for result in results
                if result["display_name"] == name
            )

            summary_data[name] = count

        classified_groups = sum(
            1 for count in summary_data.values()
            if count > 0
        )

        average_confidence = (
            sum(result["confidence"] for result in results)
            / total
        ) * 100

        # Cac chi so tong quan

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Tổng số ảnh",
                total
            )

        with col2:
            st.metric(
                "Nhóm sản phẩm xuất hiện",
                f"{classified_groups}/4"
            )

        with col3:
            st.metric(
                "Độ tin cậy trung bình",
                f"{average_confidence:.2f}%"
            )

        st.divider()

        st.markdown(
            "#### Thống kê theo nhóm sản phẩm"
        )

        st.bar_chart(
            summary_data,
            horizontal=True
        )

        st.divider()

        st.markdown(
            "#### Danh sách kết quả"
        )

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
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        excel_data = create_excel_report(results)

        st.download_button(
            "📥 TẢI BÁO CÁO EXCEL",
            data=excel_data,
            file_name="bao_cao_phan_loai_san_pham.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )

        if st.button(
            "🗑️ Xóa toàn bộ kết quả phân loại",
            use_container_width=True,
            key="clear_report"
        ):

            st.session_state.batch_results = []
            st.rerun()

    else:

        st.info(
            "Chưa có dữ liệu báo cáo. "
            "Em hãy tải ảnh và thực hiện phân loại trước nhé."
        )


# =====================================================
# 11. CHAN TRANG
# =====================================================

st.markdown("""
<div class="footer">
    HỆ THỐNG PHÂN LOẠI SẢN PHẨM
</div>
""", unsafe_allow_html=True)
