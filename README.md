# MNIST Pattern Recognition – Exercise 2

## 👥 Team & Roles
| Role | Members | Task |
|------|----------|------|
| **A&E (Lead / Integration)** | Lishangxu, liusongzhi | Project setup, data loader, report & integration |
| **B (SVM)** | Bole Yi | SVM with linear & RBF kernels |
| **C (MLP)** | Yuting Zhu | MLP model & hyperparameter tuning |
| **D (CNN)** | Jules | CNN design & training |

---

## 📁 Project Structure
```
mnist-pattern-recognition-2025/
├── cnn/                # CNN model (Jules)
├── mlp/                # MLP model (Yuting)
├── svm/                # SVM model (Bole)
├── utils/              # Shared tools (data loader)
├── report/             # Results & report
├── data/               # Local dataset (ignored by git)
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate    # (Windows)
pip install -r requirements.txt
```

---

## 💾 Data
Place the full MNIST dataset under `data/`:
```
data/
  ├─ train/0/ ... train/9/
  ├─ test/0/ ... test/9/
  ├─ gt-train.tsv
  └─ gt-test.tsv
```

---

## 🧰 Data Loader
All models share the same interface in `utils/dataset_loader.py`.

### For PyTorch (MLP / CNN)
```python
from utils.dataset_loader import get_loaders
train_loader, val_loader, test_loader = get_loaders(data_root="data")
```

### For sklearn (SVM)
```python
from utils.dataset_loader import get_numpy_data
(Xtr, ytr), (Xval, yval), (Xte, yte) = get_numpy_data(data_root="data")
```

---

## 🚀 Running Models
```bash
# SVM
python svm/train_svm.py

# MLP
python mlp/train_mlp.py

# CNN
python cnn/train_cnn.py
```

Each model should save results under its own `results/` folder:
- `train_curve.png` – training & validation curves  
- `cv_results.csv` – (SVM only) cross-validation summary  
- `test_accuracy.json` – final test accuracy  

---

## 📊 Report
Final report stored in `report/`:
- Accuracy comparison (SVM vs MLP vs CNN)  
- Plots of training curves  
- Short discussion of results  

---

© 2025 · University of Fribourg · Pattern Recognition
