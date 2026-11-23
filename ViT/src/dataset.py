import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import config

class OxfordPetDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None, val_split=0.15):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 兼容处理：检查是否存在 annotations 文件夹
        anno_dir = os.path.join(data_dir, "annotations")
        if not os.path.exists(anno_dir):
            anno_dir = data_dir
            
        if split == "test":
            txt_path = os.path.join(anno_dir, "test.txt")
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2: continue
                    self.images.append(parts[0] + ".jpg")
                    self.labels.append(int(parts[1]) - 1) 
        else:
            txt_path = os.path.join(anno_dir, "trainval.txt")
            all_images, all_labels = [], []
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2: continue
                    all_images.append(parts[0] + ".jpg")
                    all_labels.append(int(parts[1]) - 1)
            
            split_idx = int(len(all_images) * (1 - val_split))
            if split == "train":
                self.images = all_images[:split_idx]
                self.labels = all_labels[:split_idx]
            elif split == "val":
                self.images = all_images[split_idx:]
                self.labels = all_labels[split_idx:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 兼容处理：检查是否存在 images 文件夹
        img_dir = os.path.join(self.data_dir, "images")
        if not os.path.exists(img_dir):
            img_dir = self.data_dir
            
        img_path = os.path.join(img_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        return image, label

def build_dataloaders():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 强化数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(), 
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25, inplace=True)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(int(config.img_size * 256 / 224)),
        transforms.CenterCrop(config.img_size),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = OxfordPetDataset(config.data_dir, split="train", transform=train_transform, val_split=config.val_split)
    val_dataset = OxfordPetDataset(config.data_dir, split="val", transform=val_transform, val_split=config.val_split)
    test_dataset = OxfordPetDataset(config.data_dir, split="test", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader