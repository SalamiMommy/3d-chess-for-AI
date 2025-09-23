"""Optimized training loop for 24GB VRAM, 64GB RAM."""

import torch
from torch.utils.data import DataLoader, Dataset
import time
from models.resnet3d import ResNet3D
from training.checkpoint import save_checkpoint, load_latest_checkpoint

BATCH_SIZE = 256       # 24GB VRAM can handle large batches (adjust as needed)
NUM_WORKERS = 8        # 64GB RAM, so plenty for DataLoader prefetch
PIN_MEMORY = True      # Faster CPU-to-GPU transfer if using CUDA
EPOCHS = 1000          # Train for many epochs

class ChessDataset(Dataset):
    """Zero-copy dataset from examples list."""
    def __init__(self, examples):
        super().__init__()
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, pi, z = self.examples[idx]
        # x: torch.Tensor (C,9,9,9), pi: torch.Tensor, z: int/float
        return x, pi, torch.tensor(z, dtype=torch.float32)

def train_model(net, optimizer, dataset, start_epoch=0, start_step=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    loader = DataLoader(
        ChessDataset(dataset),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    criterion_policy = torch.nn.CrossEntropyLoss()
    criterion_value = torch.nn.MSELoss()

    last_save_time = time.time()
    step = start_step
    for epoch in range(start_epoch, EPOCHS):
        for batch in loader:
            x, pi, z = batch
            x = x.to(device, non_blocking=True)
            pi = pi.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)

            out_pi, out_value = net(x)
            loss_policy = criterion_policy(out_pi, pi)
            loss_value = criterion_value(out_value.squeeze(), z)
            loss = loss_policy + loss_value

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # stability
            optimizer.step()
            step += 1

            if time.time() - last_save_time > 3600:  # 1 hour
                save_checkpoint(net, optimizer, epoch, step)
                last_save_time = time.time()
        # Save at end of each epoch just in case
        save_checkpoint(net, optimizer, epoch, step)
    print("Training complete.")

def load_or_init_model():
    net = ResNet3D(blocks=15, n_moves=10_000)
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-5)
    checkpoint = load_latest_checkpoint(net, optimizer)
    if checkpoint is None:
        return net, optimizer, 0, 0
    return net, optimizer, checkpoint["epoch"], checkpoint["step"]
