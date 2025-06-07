# Brain Tumor Classification using Transfer Learning (EfficientNet / DenseNet / ResNet)

## 📌 Deskripsi Singkat

Project ini bertujuan untuk membangun model deep learning untuk **klasifikasi gambar MRI otak**, apakah terdapat **tumor** atau **tidak ada tumor**.

Model yang digunakan berbasis **Transfer Learning** dari pretrained model seperti:

- EfficientNet-B0
- DenseNet-121
- ResNet-18

Tujuan akhir: **mendeteksi keberadaan tumor otak** secara otomatis dari gambar MRI.

---

## 📂 Dataset

Dataset yang digunakan:

👉 [Brain MRI Images for Brain Tumor Detection - Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data)

### Struktur Dataset:

```
brain_tumor_dataset/
├── yes/ # Gambar MRI dengan tumor
└── no/ # Gambar MRI tanpa tumor
```

- Ukuran gambar bervariasi
- Format: JPG/PNG

### Distribusi Label:

| Label | Keterangan           |
|-------|----------------------|
| Yes   | MRI terdapat tumor    |
| No    | MRI tidak ada tumor   |

---

## 🏗️ Metodologi

### 1️⃣ Preprocessing:

- Resize gambar ke **224x224 px**
- Normalisasi sesuai ImageNet stats
- Augmentasi sederhana (horizontal flip)

### 2️⃣ Split Data:

- Train: **80%**
- Validation: **20%**
- Split dilakukan otomatis via `train_test_split`

### 3️⃣ Model Architecture:

- Menggunakan model pretrained:
    - EfficientNet-B0 / DenseNet-121 / ResNet-18
- Hanya bagian **classifier (fully connected layer)** yang diubah agar sesuai jumlah kelas (2 kelas)

### 4️⃣ Training:

- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Epochs: **5**
- Batch size: **32**

### 5️⃣ Evaluasi:

- Metrik utama: **Accuracy**
- Visualisasi Loss & Accuracy per epoch

---

## 🚀 Langkah-langkah Pengerjaan

### 1️⃣ Setup Environment

```bash
pip install torch torchvision scikit-learn
```

### 2️⃣ Import Library & Setup Device
```python
import torch
from torchvision import models, transforms, datasets
```

### 3️⃣ Split Dataset

### 4️⃣ DataLoader & Augmentasi

### 5️⃣ Transfer Learning

### 6️⃣ Training Loop

```Accuracy per Epoch
Epoch 1: Train Acc: 77.23% | Val Acc: 86.27%
Epoch 2: Train Acc: 97.52% | Val Acc: 92.16%
Epoch 3: Train Acc: 94.55% | Val Acc: 90.20%
Epoch 4: Train Acc: 94.06% | Val Acc: 92.16%
Epoch 5: Train Acc: 98.51% | Val Acc: 94.12% (Best)
```

## 📊 Visualisasi Hasil
![visualisasi_model_tumor](https://github.com/user-attachments/assets/570f95b8-1dd5-4c74-bf48-12e54a42e97b)

## 📝 Kesimpulan
✅ Model transfer learning mampu mencapai akurasi validasi >94% hanya dengan 5 epoch.
✅ Dataset sederhana, namun pretrained model mampu generalisasi dengan baik.
✅ Pipeline ini bisa diperluas ke klasifikasi gambar medis lain.

## 📌 Future Work
- Tambah augmentasi data
- Fine-tuning seluruh layer (bukan hanya classifier)
- Evaluasi di dataset MRI eksternal
- Deploy model sebagai aplikasi (streamlit / flask)

## 🙏 Credits
Project ini dibuat sebagai latihan deep learning dan transfer learning untuk klasifikasi citra medis.

## 🚀 Cara Menjalankan
```bash
git clone https://github.com/username/repo-name.git
cd repo-name
python train.py
```

---
