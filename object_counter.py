from ultralytics import YOLO
import cv2

# YOLOv8 modelini yükle (hazır model)
model = YOLO("yolov8n.pt")  # küçük, hızlı model

# Görseli oku
img = cv2.imread(r"C:\Users\LGR\Documents\Yapay-Zeka-Kursu\14 - Specialize in Data Science-27proje\Deep Learning\DeepLearn-3\test_images\test1.jpg")


# Tahmin yap
results = model(img)

# İnsan ve araba sınıflarını say
count = {"person": 0, "car": 0}

for r in results:
    boxes = r.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        if cls_name in count:
            count[cls_name] += 1

print("Nesne Sayımı:", count)

# Kutuları çiz (isteğe bağlı)
annotated_img = results[0].plot()
cv2.imwrite("result.jpg", annotated_img)
