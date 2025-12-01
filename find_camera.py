import cv2

print("🔍 Kamera portları taranıyor...")

# İlk 5 portu tara
for index in range(5):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Index {index}: Kamera ÇALIŞIYOR! (Çözünürlük: {frame.shape[1]}x{frame.shape[0]})")
        else:
            print(f"⚠️ Index {index}: Kamera var ama görüntü vermiyor (OBS/Sanal Kamera olabilir).")
        cap.release()
    else:
        print(f"❌ Index {index}: Kamera Yok.")