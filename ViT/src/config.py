import os
import torch

class Config:
    # --- Path Config ---
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(project_dir, '../../data/oxford-iiit-pet'))
    checkpoint_dir = os.path.abspath(os.path.join(project_dir, '../checkpoints'))
    output_dir = os.path.abspath(os.path.join(project_dir, '../outputs'))
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Config ---
    img_size = 224
    patch_size = 16
    num_classes = 37 
    batch_size = 32
    num_workers = 4
    val_split = 0.15

    # --- Model Config ---
    # 修改为 12 层，与官方 vit_small 完全对齐，完整加载所有预训练权重
    embed_dim = 384
    depth = 12       
    num_heads = 6
    mlp_ratio = 4.0
    drop_rate = 0.1
    attn_drop_rate = 0.0
    drop_path_rate = 0.1

    # --- Training Config (Fine-tuning) ---
    # 增加训练轮次，让余弦退火真正走完
    epochs = 50          
    lr = 1e-4            # 稍微调高一点，因为现在是完整的 12 层微调
    weight_decay = 0.01
    warmup_epochs = 3    
    label_smoothing = 0.1
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 预训练模型名称
    pretrained_model_name = 'vit_small_patch16_224'

config = Config()