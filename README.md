# YOLOv8 ile Nesne Sayma (Object Counting)

Bu proje, Ultralytics YOLOv8 modeli kullanılarak görsellerdeki insan ve araba gibi nesneleri saymak için geliştirilmiştir.

## 🔧 Kurulum

pip install -r requirements.txt



Kullanım

streamlit run app.py


🧠 Model
Kullanılan model: yolov8n.pt
Alternatif olarak Hugging Face üzerinden indirilebilir:
https://huggingface.co/yazodi/yolov8-object-counting



Proje Yapısı
app.py: Streamlit arayüzü

object_counter.py: Temel sayma scripti

test_images/: Örnek görseller

yolov8n.pt: Model dosyası