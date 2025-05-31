import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Modeli yükle
model = YOLO("yolov8n.pt")  # küçük ve hızlı model

# Sayılacak sınıflar
target_classes = ["person", "car"]

st.title("🚦 Nesne Sayma Uygulaması (YOLOv8 ile)")
st.markdown("Bir görsel yükleyin. Uygulama insan ve araba gibi nesneleri sayacaktır.")

# Görsel yükleyici
uploaded_file = st.file_uploader("Bir resim dosyası yükleyin (.jpg, .png)", type=["jpg", "png"])

if uploaded_file is not None:
    # Görseli göster
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # Görseli OpenCV formatına çevir
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # YOLO ile tahmin yap
    results = model(img)

    # Sayım
    count = {cls: 0 for cls in target_classes}
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name in count:
                count[cls_name] += 1

    # Sonuçları göster
    st.subheader("🔢 Nesne Sayımı:")
    for cls, c in count.items():
        st.write(f"- {cls}: **{c} adet**")

    # Annotated görüntüyü göster
    result_img = results[0].plot()
    st.image(result_img, caption="📍 Tespit Sonuçları", use_column_width=True)
