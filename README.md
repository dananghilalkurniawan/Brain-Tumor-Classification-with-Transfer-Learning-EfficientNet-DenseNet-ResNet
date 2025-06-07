# Brain Tumor Classification with Transfer Learning (EfficientNet, DenseNet, ResNet)

Sip! Ini contoh lengkap training model **EfficientNet (bisa juga diganti DenseNet atau ResNet)** dengan transfer learning di PyTorch, lengkap dengan split data otomatis dan semua tahapannya, dibagi per bagian agar gampang diikuti.

---

## Bagian 1: Import Library dan Setup Device

```python
import os
import shutil
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import time
import copy

# Cek device GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device yang digunakan: {device}")
```
