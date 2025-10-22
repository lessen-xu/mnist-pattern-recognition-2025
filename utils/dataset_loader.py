# D（CNN）and C（MLP）:
# from utils.dataset_loader import get_loaders
# train_loader, val_loader, test_loader = get_loaders(data_root="data")


# B（SVM）
# from utils.dataset_loader import get_numpy_data
# (Xtr, ytr), (Xval, yval), (Xte, yte) = get_numpy_data(data_root="data")


from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class MNISTDataset(Dataset):
    def __init__(self, root, tsv, transform=None):
        df = pd.read_csv(Path(root)/tsv, sep="\t", header=None, names=["relpath","label"])
        self.root = Path(root)
        self.paths = df["relpath"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.root/self.paths[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def get_loaders(data_root="data", val_ratio=0.1, batch_size=128, num_workers=2, seed=42):
    """return train_loader, val_loader, test_loader（PyTorch DataLoader）"""
    df = pd.read_csv(Path(data_root)/"gt-train.tsv", sep="\t", header=None, names=["relpath","label"])
    y = df["label"].values
    idx = np.arange(len(df))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(idx, y))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])

    # Definite Dataset again
    class _SubDataset(Dataset):
        def __init__(self, df, root, transform):
            self.df = df
            self.root = Path(root)
            self.transform = transform
        def __len__(self):
            return len(self.df)
        def __getitem__(self, i):
            p = self.df.iloc[i,0]; y = int(self.df.iloc[i,1])
            img = Image.open(self.root/p).convert("L")
            return self.transform(img), y

    train_ds = _SubDataset(train_df, data_root, transform)
    val_ds   = _SubDataset(val_df, data_root, transform)
    test_ds  = MNISTDataset(data_root, "gt-test.tsv", transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def get_numpy_data(data_root="data", val_ratio=0.1, seed=42):
    """For SVM, returns numpy arrays (X_train, y_train), (X_val, y_val), (X_test, y_test)"""
    df = pd.read_csv(Path(data_root)/"gt-train.tsv", sep="\t", header=None, names=["relpath","label"])
    y = df["label"].values
    idx = np.arange(len(df))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(idx, y))
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = pd.read_csv(Path(data_root)/"gt-test.tsv", sep="\t", header=None, names=["relpath","label"])

    def load_images(df):
        X = []
        for p in df["relpath"]:
            arr = np.array(Image.open(Path(data_root)/p).convert("L"), dtype=np.float32) / 255.0
            X.append(arr.reshape(-1))
        return np.stack(X), df["label"].astype(int).values

    X_train, y_train = load_images(train_df)
    X_val, y_val = load_images(val_df)
    X_test, y_test = load_images(test_df)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
