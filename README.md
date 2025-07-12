# 🧠 Brain Tumor Detection using EfficientNetV2B0

This project builds a binary image classification model using **EfficientNetV2B0** to detect the presence of a brain tumor from MRI images.

## 📌 Features

- ✅ EfficientNetV2B0 with transfer learning (ImageNet weights)
- 🧠 Detects Tumor / No Tumor from MRI scans
- 🗃️ Custom dataset of brain MRI images (e.g., Kaggle)
- 📉 Fine-tuned using GlobalAveragePooling + Dropout
- 💾 Trained model weights saved as `.weights.h5`

---

## 🧠 Model Architecture

- Backbone: `EfficientNetV2B0` (without top layer)
- Add-on Layers:
  - `GlobalAveragePooling2D`
  - `Dropout(0.3)`
  - `Dense(1, activation="sigmoid")`

Compiled with:

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
