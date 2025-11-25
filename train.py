import torch
import torch.nn as nn
import torch.optim as optim

from model.py import Yolov1
from loss.py import YOLOV1Loss


model = Yolov1()
criterion = YOLOV1Loss
optimizer = optim.AdamW(lr=1e-5)

num_epochs = 30
best_val_loss = float("inf")
no_improve=0 #implementing early stopping

for idx_epoch in range(num_epochs):
    model.train()
