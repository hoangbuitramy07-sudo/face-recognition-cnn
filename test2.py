import cv2
import numpy as np
import tensorflow as tf
import os

image_path = r"C:\Users\ADMIN\Desktop\AI\face\face_dataset\29.png"

model_path = "multi_face_model_cnn.h5"
classes_path = "classes.npy"

if not os.path.exists(model_path) or not os.path.exists(classes_path):
    print("[ERROR] Thiếu file mô hình! Hãy chạy file face_cnn_multi.py trước.")
    exit()

model = tf.keras.models.load_model(model_path)
classes = np.load(classes_path)

if not os.path.exists(image_path):
    print(f"[ERROR] Không tìm thấy ảnh test tại: {image_path}")
    exit()

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(image_path) 

IMG_SIZE = 128
resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
normalized = resized / 255.0

img_input = np.expand_dims(normalized, axis=-1)
input_data = np.expand_dims(img_input, axis=0)

predictions = model.predict(input_data)[0]
predicted_label_index = np.argmax(predictions)

predicted_name = classes[predicted_label_index]
confidence = predictions[predicted_label_index] * 100

result_text = f"{predicted_name} - {confidence:.2f}%"
print("KẾT QUẢ DỰ ĐOÁN CNN CƠ BẢN")
print(result_text)

color = (0, 255, 0) if "TRA MY" in predicted_name else (0, 165, 255)
cv2.putText(img_color, result_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
cv2.imshow("He thong Nhan dien Khuon mat CNN", img_color)

print("\n[INFO] Nhấn một phím bất kỳ tại cửa sổ ảnh để THOÁT.")
cv2.waitKey(0)
cv2.destroyAllWindows()