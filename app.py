import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import pandas as pd

st.set_page_config(
    page_title="Deteksi Sel Darah",
    layout="wide"
)

model = YOLO("best.pt")

st.title("Deteksi Sel Darah Menggunakan YOLOv8")

st.write(
    "Website ini digunakan untuk mendeteksi White Blood Cell (WBC), Red Blood Cell (RBC), dan Platelets menggunakan model YOLOv8."
)

uploaded_file = st.file_uploader(
    "Upload Gambar Sel Darah",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)

    if st.button("Deteksi"):

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_file.name)

        results = model(temp_file.name)

        result = results[0]

        result_image = result.plot()
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Hasil Deteksi")
            st.image(result_image, use_container_width=True)

        wbc = 0
        rbc = 0
        platelets = 0

        total_conf = 0
        total_obj = 0

        data = []

        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            total_conf += conf
            total_obj += 1

            if cls == 0:
                wbc += 1

            elif cls == 1:
                rbc += 1

            elif cls == 2:
                platelets += 1

            data.append(
                {
                    "Kelas": model.names[cls],
                    "Confidence": round(conf, 2)
                }
            )

        avg_conf = 0

        if total_obj > 0:
            avg_conf = total_conf / total_obj

        st.divider()

        st.subheader("Jumlah Objek Terdeteksi")

        c1, c2, c3 = st.columns(3)

        c1.metric("White Blood Cell", wbc)
        c2.metric("Red Blood Cell", rbc)
        c3.metric("Platelets", platelets)

        st.divider()

        st.subheader("Informasi Deteksi")

        c4, c5, c6 = st.columns(3)

        c4.metric("Total Objek", total_obj)
        c5.metric("Confidence Rata-rata", f"{avg_conf:.2f}")
        c6.metric("Waktu Inferensi", f"{result.speed['inference']:.2f} ms")

        st.divider()

        st.subheader("Detail Hasil Deteksi")

        df = pd.DataFrame(data)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        _, buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        )

        st.download_button(
            label="Download Hasil Deteksi",
            data=buffer.tobytes(),
            file_name="hasil_deteksi.jpg",
            mime="image/jpeg"
        )