import torch
from model import VisionTransformer
from config import config

def check():
    print("Checking model structure...")
    model = VisionTransformer()
    x = torch.randn(2, 3, config.img_size, config.img_size) 
    
    out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape} (Expected: [2, {config.num_classes}])")
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {params / 1e6:.2f} M")

if __name__ == '__main__':
    check()