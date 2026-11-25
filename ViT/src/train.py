import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from config import config
from dataset import build_dataloaders
from model import VisionTransformer
from utils import set_seed, get_logger, AverageMeter, save_checkpoint, load_checkpoint

def train_one_epoch(train_loader, model, criterion, optimizer, scaler, device):
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device=="cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        losses.update(loss.item(), images.size(0))
        accs.update(acc.item(), images.size(0))
        
    return losses.avg, accs.avg

@torch.no_grad()
def evaluate(val_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        losses.update(loss.item(), images.size(0))
        accs.update(acc.item(), images.size(0))
        
    return losses.avg, accs.avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    args = parser.parse_args()
    
    set_seed(config.seed)
    logger = get_logger(os.path.join(config.output_dir, 'train.log'))
    device = config.device
    
    train_loader, val_loader, _ = build_dataloaders()
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    model = VisionTransformer().to(device)
    
    # === 加载预训练权重 ===
    if not args.resume:
        model.load_pretrained()
        
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # === Scheduler ===
    scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=config.warmup_epochs)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=config.epochs - config.warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[config.warmup_epochs])
    
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None
    
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        ckpt_path = os.path.join(config.checkpoint_dir, 'last.pth')
        if os.path.exists(ckpt_path):
            try:
                start_epoch, best_acc = load_checkpoint(ckpt_path, model, optimizer, scheduler)
            except RuntimeError as e:
                logger.info(f"Failed to load checkpoint due to shape mismatch (likely model structure changed): {e}")
                logger.info("Starting training from scratch instead.")
                start_epoch = 0
        else:
            logger.info("No checkpoint found. Training from scratch.")
    
    logger.info(f"Start training from Epoch {start_epoch} to {config.epochs}")
    
    for epoch in range(start_epoch, config.epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate(val_loader, model, criterion, device)
        
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch [{epoch+1}/{config.epochs}] "
                    f"LR: {lr:.6f} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | Time: {time.time()-start_time:.1f}s")
        
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            
        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc
        }
        
        save_checkpoint(state, os.path.join(config.checkpoint_dir, 'last.pth'))
        
        if is_best:
            save_checkpoint(state, os.path.join(config.checkpoint_dir, 'best.pth'))
            logger.info(f"==> Best model found and saved (Val Acc: {best_acc:.4f})")
            
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Warning] Training interrupted (Ctrl+C).")
        print("The progress from the last completed epoch is safely saved in 'last.pth'.")
        print("Use `python train.py --resume` to continue training next time.")c