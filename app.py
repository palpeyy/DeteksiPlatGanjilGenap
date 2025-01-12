import cv2
import numpy as np
from ultralytics import YOLO
import easyocr  # Library untuk OCR

# Load YOLOv8 model yang sudah dilatih
model = YOLO("runs/train/Ganjil_Genap_model/weights/best.pt")  # Pastikan path model benar

# Inisialisasi EasyOCR Reader untuk membaca teks pada plat nomor
reader = easyocr.Reader(['en'], gpu=False)  # Gunakan GPU=True jika memiliki GPU

def extract_plate_number_easyocr(plate_image):
    """Ekstrak nomor plat menggunakan EasyOCR."""
    # Preprocessing gambar untuk hasil OCR yang lebih baik
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Gunakan EasyOCR untuk membaca teks dari gambar
    results = reader.readtext(binary, detail=0)  # detail=0 hanya mengembalikan teks
    return " ".join(results).strip() if results else "Tidak Terbaca"

def process_frame(frame):
    """Proses satu frame untuk mendeteksi plat dan mengklasifikasikan ganjil/genap."""
    results = model(frame)  # Deteksi menggunakan YOLOv8
    annotated_frame = frame.copy()

    for result in results:  # Iterasi melalui deteksi
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
            conf = box.conf[0].item()
            cls = box.cls[0].item()  # Kelas yang diprediksi oleh YOLO

            if conf > 0.5:  # Threshold confidence
                # Crop area plat
                plate_image = frame[y1:y2, x1:x2]

                # Ekstrak nomor plat menggunakan OCR
                plate_number = extract_plate_number_easyocr(plate_image)

                # Gunakan klasifikasi yang didapat dari model (Ganjil/Genap)
                classification = result.names[int(cls)]  # Mengambil nama kelas (Ganjil/Genap)

                # Buat label untuk ditampilkan
                label = f"{plate_number}: {classification}"

                # Gambar bounding box pada plat kendaraan
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Tampilkan teks di atas bounding box
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return annotated_frame

def main():
    """Fungsi utama untuk menangkap video dan memproses frame."""
    cap = cv2.VideoCapture(0)  # Gunakan webcam (atau ganti dengan path file video)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Proses frame untuk mendeteksi dan membaca plat nomor
        processed_frame = process_frame(frame)

        # Tampilkan frame yang telah diproses
        cv2.imshow("Plat Nomor Detection", processed_frame)

        # Keluar saat menekan tombol 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()