# MNIST Pattern Recognition – Exercise 2

## 👥 Team & Roles
| Role | Members | Description |
|------|----------|--------------|
| **A&E (Lead / Integration)** | Lishang Xu, Songzhi Liu | Project structure, dataset loader, final report & merging |
| **B (SVM)** | Bole Yi | SVM with linear & RBF kernels, hyperparameter tuning via CV |
| **C (MLP)** | Yuting Zhu | MLP with 1–2 hidden layers, tuning hidden size & learning rate |
| **D (CNN)** | Jules | CNN with different kernel sizes & layers, tuning learning rate |

---

## 🧭 Project Structure

```
mnist-pattern-recognition-2025/
├── cnn/                # Jules
│   └── train_cnn.py
├── mlp/                # Yuting Zhu
│   └── train_mlp.py
├── svm/                # Bole Yi
│   └── train_svm.py
├── utils/              # Shared utilities
│   └── dataset_loader.py
├── report/             # Final report (Markdown or PDF)
│   └── report.md
├── data/               # Local dataset (ignored by git)
│   ├── train/
│   ├── test/
│   ├── gt-train.tsv
│   └── gt-test.tsv
├── requirements.txt
└── README.md
```

---

## 🔒 Branch Policy

- **main**: Protected. Only merged via Pull Request (PR).  
- **feature branches**:  
  - `feature/svm-bole-yi`  
  - `feature/mlp-yuting-zhu`  
  - `feature/cnn-jules`  
- **infra/**: AE setup & integration branches.

---

## ⚙️ Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💾 Data Preparation

Download and unzip the **Full MNIST Dataset** (provided by course).  
Place it under the `data/` folder like this:

```
data/
  ├─ train/0/*.png ... train/9/*.png
  ├─ test/0/*.png  ... test/9/*.png
  ├─ gt-train.tsv
  └─ gt-test.tsv
```

This folder is **ignored by Git** (listed in `.gitignore`).

---

## 📦 Unified Dataset Loader

All models use the same dataset interface in `utils/dataset_loader.py`.

### 🔹 For PyTorch (MLP / CNN)
```python
from utils.dataset_loader import get_loaders

train_loader, val_loader, test_loader = get_loaders(data_root="data", val_ratio=0.1)
```

### 🔹 For sklearn (SVM)
```python
from utils.dataset_loader import get_numpy_data

(Xtr, ytr), (Xval, yval), (Xte, yte) = get_numpy_data(data_root="data", val_ratio=0.1)
```

---

## 🚀 Running Each Model

### 🧠 SVM (Bole Yi)
```bash
python svm/train_svm.py --val-ratio 0.1 --results-dir svm/results
```

### 🔢 MLP (Yuting Zhu)
```bash
python mlp/train_mlp.py --epochs 20 --batch-size 128 --results-dir mlp/results
```

### 🧩 CNN (Jules)
```bash
python cnn/train_cnn.py --epochs 20 --batch-size 128 --results-dir cnn/results
```

---

## 📊 Expected Outputs

Each model should generate results in its own `results/` folder:

| File | Description |
|------|--------------|
| `train_curve.png` | Loss/accuracy curves on train & validation sets |
| `cv_results.csv` | (SVM only) Hyperparameter cross-validation table |
| `test_accuracy.json` | Final accuracy and parameters on the test set |

---

## 🧾 Final Report (A&E)

The final report should include:

1. Overview of dataset & preprocessing  
2. Method description for SVM, MLP, CNN  
3. Training/validation curves  
4. Cross-validation results  
5. Final test accuracies comparison  
6. Discussion on model performance & optimization

Report stored in: `report/report.md` or `report/report.pdf`

---

## 📧 Submission

- Repository: https://github.com/lessen-xu/mnist-pattern-recognition-2025  
- Send link to **michael.jungo@unifr.ch**  
- If repo is private: add user **jungomi** with *Read* access.  
- Deadline: **Nov 5, 2025 (end of day)**

---

## 🧩 Notes
- Do **not** touch the test set until final evaluation.
- Use the validation split to tune hyperparameters.
- Keep code modular; each model runs independently.
- Only A&E handles final merging & report formatting.

---
© University of Fribourg · Department of Informatics · Autumn Semester 2025
