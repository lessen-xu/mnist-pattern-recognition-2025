from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os, torch

 
torch.manual_seed(42)

 
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "MNIST-full")

 
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

 
train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
test_data  = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)


total_len = len(train_data)
train_len = int(0.8 * total_len)
val_len   = total_len - train_len
train_set, val_set = random_split(train_data, [train_len, val_len])

 
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_set, batch_size=64, num_workers=2)
test_loader  = DataLoader(test_data, batch_size=64, num_workers=2)


