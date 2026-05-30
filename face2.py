import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os

dataset_path = r"C:\Users\ADMIN\Desktop\AI\face\face_dataset\other"

classes = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
num_classes = len(classes)

data = []
labels = []
IMG_SIZE = 128  

print("[INFO] Đang nạp dataset với kích thước 128x128...")
for label, person in enumerate(classes):
    person_path = os.path.join(dataset_path, person)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        img_input = np.expand_dims(normalized, axis=-1)

        data.append(img_input)
        labels.append(label)

print(f"Tổng số ảnh nạp thành công: {len(data)}")

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([

    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    GlobalAveragePooling2D(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=80,       
    batch_size=16,    
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"ĐỘ CHÍNH XÁC CNN CƠ BẢN: {accuracy*100:.2f}%")

model.save("multi_face_model_cnn.h5")
np.save("classes.npy", classes)
print(" Đã lưu mô hình nâng cấp thành công!")