import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch: {torch.__version__}, ROCm: {torch.version.hip}, Device: {device}, GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'N/A'}")
tensor = torch.randn(100, device=device)  # Small tensor
model = torch.nn.Linear(100, 1).to(device)  # Simple model
out = model(tensor)
print(out.sum())  # Force compute
print("Test passed")
