import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden1=128, hidden2=64, num_classes=10, dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.LeakyReLU(0.01)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.LeakyReLU(0.01)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x); x = self.relu1(x); x = self.drop1(x)
        x = self.fc2(x); x = self.relu2(x); x = self.drop2(x)
        x = self.fc3(x)
        return x
