import os
import io
import tempfile

import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="BloodCell AI",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("File style.css tidak ditemukan. Tampilan akan menggunakan gaya default Streamlit.")


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = None
model_error = None

try:
    model = load_model()
except Exception as e:
    model_error = str(e)


# ----------------------------------------------------------------------
# Small render helpers (avoid repeating the same markdown block 4x)
# ----------------------------------------------------------------------

def stat_card(label, value, desc, variant=""):
    st.markdown(
        f"""
        <div class="stat-card {variant}">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(label, value, unit, desc):
    unit_html = f"<span>{unit}</span>" if unit else ""
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">{label}</div>
            <div class="info-value">{value}{unit_html}</div>
            <div class="info-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(number, title, subtitle, extra_class=""):
    st.markdown(
        f"""
        <div class="section-heading {extra_class}">
            <div class="section-number">{number}</div>
            <div>
                <div class="section-title">{title}</div>
                <div class="section-subtitle">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">BC</div>
            <div>
                <div class="brand-name">BloodCell AI</div>
                <div class="brand-subtitle">Detection System</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-line'></div>", unsafe_allow_html=True)
    st.markdown("<div class='menu-label'>MAIN MENU</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="side-menu active">
            <span class="menu-icon">01</span>
            <span>Dashboard</span>
        </div>
        <div class="side-menu">
            <span class="menu-icon">02</span>
            <span>Blood Cell Detection</span>
        </div>
        <div class="side-menu">
            <span class="menu-icon">03</span>
            <span>Detection History</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='menu-label menu-space'>MODEL</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="model-info">
            <div class="model-title">YOLOv8</div>
            <div class="model-row"><span>Framework</span><strong>Ultralytics</strong></div>
            <div class="model-row"><span>Classes</span><strong>3</strong></div>
            <div class="model-row"><span>Task</span><strong>Object Detection</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-bottom'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-footer">
            <div class="footer-title">Blood Cell Detection</div>
            <div class="footer-text">YOLOv8 Object Detection System</div>
            <div class="footer-text">Teknik Informatika</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Topbar
# ----------------------------------------------------------------------

status_label = "Model Ready" if model is not None else "Model Error"
st.markdown(
    f"""
    <div class="topbar">
        <div>
            <div class="topbar-small">AI OBJECT DETECTION</div>
            <div class="topbar-title">Blood Cell Analysis</div>
        </div>
        <div class="topbar-status">
            <span class="status-dot"></span>
            {status_label}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model_error:
    st.error(
        f"Model **best.pt** gagal dimuat. Pastikan file `best.pt` berada di folder yang sama "
        f"dengan `app.py`.\n\nDetail error: `{model_error}`"
    )
    st.stop()


# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-left">
            <div class="hero-number">01</div>
            <div class="hero-title">Deteksi Sel Darah</div>
            <div class="hero-description">
                Analisis citra mikroskopis darah menggunakan model YOLOv8
                untuk mendeteksi WBC, RBC, dan Platelets secara otomatis.
            </div>
            <div class="hero-tags">
                <span>YOLOv8</span>
                <span>Object Detection</span>
                <span>3 Classes</span>
            </div>
        </div>
        <div class="hero-visual">
            <div class="visual-circle circle-one"></div>
            <div class="visual-circle circle-two"></div>
            <div class="visual-center"></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Upload
# ----------------------------------------------------------------------

section_heading("02", "Unggah Gambar", "Masukkan citra mikroskopis untuk memulai proses deteksi")

uploaded_file = st.file_uploader(
    "Pilih gambar sel darah",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="upload-info">
            <div class="upload-symbol">+</div>
            <div class="upload-title">Unggah Gambar Sel Darah</div>
            <div class="upload-description">
                Seret dan lepas gambar di area upload atau klik tombol Browse files
            </div>
            <div class="upload-format">Format yang didukung: JPG · JPEG · PNG</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Gagal membaca gambar. Coba unggah file JPG/PNG lain.")
        st.stop()

    st.markdown(
        """
        <div class="preview-heading">
            <div class="preview-title">Preview Gambar</div>
            <div class="preview-subtitle">Gambar berhasil dimuat dan siap dianalisis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="image-label">ORIGINAL IMAGE</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(
            f"""
            <div class="image-meta">
                <span>File</span>
                <strong>{uploaded_file.name}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="detection-panel">
                <div class="detection-panel-title">Ready for Detection</div>
                <div class="detection-panel-text">
                    Tekan tombol di bawah untuk menjalankan model YOLOv8
                    pada gambar yang telah diunggah.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        detect = st.button("MULAI DETEKSI", use_container_width=True)

        if not detect:
            st.markdown(
                """
                <div class="detection-note">
                    Model akan mendeteksi tiga jenis objek: WBC, RBC, dan Platelets.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------
    # Run detection
    # ------------------------------------------------------------------

    if detect:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                temp_path = tmp.name
            image.save(temp_path)

            with st.spinner("Sedang melakukan analisis citra..."):
                results = model(temp_path)

            result = results[0]
            result_image_rgb = Image.fromarray(result.plot()[:, :, ::-1])

            # ---- results section ----
            section_heading("03", "Hasil Deteksi", "Objek yang berhasil ditemukan oleh model")
            st.image(result_image_rgb, use_container_width=True)

            counts = {"wbc": 0, "rbc": 0, "plt": 0}
            rows = []

            for i, box in enumerate(result.boxes, start=1):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]

                if cls == 0:
                    counts["wbc"] += 1
                elif cls == 1:
                    counts["rbc"] += 1
                elif cls == 2:
                    counts["plt"] += 1

                rows.append({"No": i, "Jenis Sel": class_name, "Confidence": f"{conf:.2f}"})

            total_obj = len(rows)
            avg_conf = (
                sum(float(box.conf[0]) for box in result.boxes) / total_obj
                if total_obj > 0 else 0
            )

            # ---- summary ----
            section_heading(
                "04", "Ringkasan Deteksi", "Statistik objek yang ditemukan",
                extra_class="result-summary-heading",
            )

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                stat_card("WBC", counts["wbc"], "White Blood Cell", "wbc")
            with c2:
                stat_card("RBC", counts["rbc"], "Red Blood Cell", "rbc")
            with c3:
                stat_card("PLT", counts["plt"], "Platelets", "plt")
            with c4:
                stat_card("TOTAL", total_obj, "Detected Objects", "total")

            # ---- analysis info ----
            section_heading(
                "05", "Informasi Analisis", "Performa model pada proses inferensi",
                extra_class="detail-heading",
            )

            i1, i2, i3 = st.columns(3)
            with i1:
                info_card("CONFIDENCE RATA-RATA", f"{avg_conf:.2f}", "", "Tingkat keyakinan model")
            with i2:
                info_card(
                    "INFERENCE TIME", f"{result.speed['inference']:.2f}", "ms",
                    "Waktu proses deteksi",
                )
            with i3:
                info_card(
                    "IMAGE RESOLUTION",
                    f"{result.orig_shape[1]} × {result.orig_shape[0]}", "",
                    "Resolusi gambar input",
                )

            # ---- detail table ----
            section_heading(
                "06", "Detail Objek", "Daftar seluruh objek yang berhasil terdeteksi",
                extra_class="detail-heading",
            )

            df = pd.DataFrame(rows)
            if not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada objek yang terdeteksi.")

            # ---- distribution chart ----
            section_heading(
                "07", "Distribusi Sel", "Perbandingan jumlah objek berdasarkan jenis sel",
                extra_class="detail-heading",
            )

            chart_data = pd.DataFrame(
                {
                    "Jenis Sel": ["WBC", "RBC", "Platelets"],
                    "Jumlah": [counts["wbc"], counts["rbc"], counts["plt"]],
                }
            )
            st.bar_chart(chart_data.set_index("Jenis Sel"), use_container_width=True)

            # ---- download ----
            buffer = io.BytesIO()
            result_image_rgb.save(buffer, format="JPEG", quality=95)

            st.markdown(
                """
                <div class="download-section">
                    <div class="download-title">Simpan Hasil Analisis</div>
                    <div class="download-text">
                        Download gambar yang sudah dilengkapi bounding box
                        dan label hasil deteksi.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.download_button(
                label="DOWNLOAD HASIL DETEKSI",
                data=buffer.getvalue(),
                file_name="hasil_deteksi.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)


# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------

st.markdown(
    """
    <div class="page-footer">
        <div class="footer-line"></div>
        <div class="footer-content">
            <span>BloodCell AI</span>
            <span>YOLOv8 Object Detection</span>
            <span>Teknik Informatika</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
