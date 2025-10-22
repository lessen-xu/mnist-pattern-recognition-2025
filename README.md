# MNIST Pattern Recognition

## 👥 Team & Roles

| Role | Members | Task |
|------|----------|------|
| **A\&E (Lead / Integration)** | Lishang Xu, Songzhi Liu| Project setup, data loader, report & integration |
| **B (SVM)** | Bole Yi | SVM with linear & RBF kernels |
| **C (MLP)** | Yuting Zhu | MLP model & hyperparameter tuning |
| **D (CNN)** | Jules | CNN design & training |

-----

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

-----

## ⚙️ Environment Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate    # (Windows)
# source .venv/bin/activate  # (Linux/macOS)

# Install required packages
pip install -r requirements.txt
```

### 💾 Data

Place the full MNIST dataset under the `data/` directory:

```
data/
  ├─ train/0/ ... train/9/
  ├─ test/0/ ... test/9/
  ├─ gt-train.tsv
  └─ gt-test.tsv
```

### 🧰 Data Loader

All models share the same data loader interface provided in `utils/dataset_loader.py`.

**For PyTorch (MLP / CNN)**

```python
from utils.dataset_loader import get_loaders
train_loader, val_loader, test_loader = get_loaders(data_root="data")
```

**For sklearn (SVM)**

```python
from utils.dataset_loader import get_numpy_data
(Xtr, ytr), (Xval, yval), (Xte, yte) = get_numpy_data(data_root="data")
```

-----

## 🚀 Running Models

Each script should be run from the root of the project.

```bash
# SVM
python svm/train_svm.py

# MLP
python mlp/train_mlp.py

# CNN
python cnn/train_cnn.py
```

Each model will save its results under its own `results/` folder (e.g., `svm/results/`):

  * **train\_curve.png** – Training & validation curves
  * **cv\_results.csv** – (SVM only) Cross-validation summary
  * **test\_accuracy.json** – Final test accuracy

-----

## 📊 Report

The final report is stored in `report/` and includes:

  * Accuracy comparison (SVM vs MLP vs CNN)
  * Plots of training curves
  * A short discussion of the results

-----

© 2025 · University of Fribourg · Pattern Recognition
