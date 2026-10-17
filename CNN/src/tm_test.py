import os
import time
import torch

print("TrainManager test started", flush=True)
print("Working directory:", os.getcwd(), flush=True)
print("PyTorch:", torch.__version__, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("CUDA matrix test passed:", tuple(y.shape), flush=True)

for i in range(30):
    print(f"Heartbeat {i + 1}/30", flush=True)
    time.sleep(2)

print("TrainManager test completed", flush=True)
