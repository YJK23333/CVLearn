import os
import torch
from torch.utils.data import DataLoader
from config import config
from dataset import OxfordPetDataset
from model import VisionTransformer
from torchvision import transforms

@torch.no_grad()
def test():
    device = config.device
    model = VisionTransformer().to(device)
    
    best_ckpt_path = os.path.join(config.checkpoint_dir, 'best.pth')
    if not os.path.exists(best_ckpt_path):
        print(f"Checkpoint {best_ckpt_path} not found. Please train the model first.")
        return
        
    checkpoint = torch.load(best_ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Successfully loaded weights (Epoch {checkpoint['epoch']}, Best Val Acc: {checkpoint['best_acc']:.4f})")
    
    model.eval()
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(int(config.img_size * 256 / 224)),
        transforms.CenterCrop(config.img_size),
        transforms.ToTensor(),
        normalize
    ])
    test_dataset = OxfordPetDataset(config.data_dir, split="test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    correct = 0
    total = 0
    
    print("Evaluating on test set...")
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    print(f"Test Top-1 Accuracy: {100. * correct / total:.2f}%")

if __name__ == '__main__':
    test()