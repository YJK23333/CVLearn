
import torch
import torch.nn as nn

from torchvision.models import resnet18

from config import (
    IMAGE_SIZE,
    PATCH_SIZE,
    EMBED_DIM,
    DEPTH,
    NUM_HEADS,
    MLP_RATIO,
    NUM_CLASSES,
    DROPOUT,
    ATTENTION_DROPOUT
)


# ============================================================
# 1. Patch Embedding
# ============================================================

class PatchEmbedding(nn.Module):

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=3,
        embed_dim=EMBED_DIM
    ):

        super().__init__()

        self.image_size = image_size

        self.patch_size = patch_size

        self.num_patches = (
            image_size // patch_size
        ) ** 2

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        # [B, 3, 224, 224]
        x = self.projection(x)

        # [B, 384, 14, 14]
        x = x.flatten(2)

        # [B, 384, 196]
        x = x.transpose(1, 2)

        # [B, 196, 384]
        return x


# ============================================================
# 2. Multi-Head Self-Attention
# ============================================================

class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        attention_dropout=ATTENTION_DROPOUT,
        dropout=DROPOUT
    ):

        super().__init__()

        assert embed_dim % num_heads == 0

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(
            embed_dim,
            embed_dim * 3
        )

        self.attention_dropout = nn.Dropout(
            attention_dropout
        )

        self.projection = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.projection_dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):

        batch_size, num_tokens, embed_dim = x.shape

        # ----------------------------------------------------
        # Q, K, V
        # ----------------------------------------------------

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim
        )

        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )

        q, k, v = qkv.unbind(0)

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        attention = (
            q @ k.transpose(-2, -1)
        ) * self.scale

        attention = attention.softmax(
            dim=-1
        )

        attention = self.attention_dropout(
            attention
        )

        # ----------------------------------------------------
        # Weighted Sum
        # ----------------------------------------------------

        x = attention @ v

        x = x.transpose(1, 2)

        x = x.reshape(
            batch_size,
            num_tokens,
            embed_dim
        )

        # ----------------------------------------------------
        # Output Projection
        # ----------------------------------------------------

        x = self.projection(x)

        x = self.projection_dropout(x)

        return x


# ============================================================
# 3. MLP
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT
    ):

        super().__init__()

        hidden_dim = int(
            embed_dim * mlp_ratio
        )

        self.fc1 = nn.Linear(
            embed_dim,
            hidden_dim
        )

        self.activation = nn.GELU()

        self.fc2 = nn.Linear(
            hidden_dim,
            embed_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):

        x = self.fc1(x)

        x = self.activation(x)

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.dropout(x)

        return x


# ============================================================
# 4. Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        attention_dropout=ATTENTION_DROPOUT
    ):

        super().__init__()

        # ----------------------------------------------------
        # First Pre-Norm
        # ----------------------------------------------------

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout
        )

        # ----------------------------------------------------
        # Second Pre-Norm
        # ----------------------------------------------------

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

    def forward(self, x):

        # Attention + Residual
        x = x + self.attention(
            self.norm1(x)
        )

        # MLP + Residual
        x = x + self.mlp(
            self.norm2(x)
        )

        return x


# ============================================================
# 5. DeiT
# ============================================================

class DeiT(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES
    ):

        super().__init__()

        # ----------------------------------------------------
        # Patch Embedding
        # ----------------------------------------------------

        self.patch_embed = PatchEmbedding()

        num_patches = self.patch_embed.num_patches

        # ----------------------------------------------------
        # CLS Token
        # ----------------------------------------------------

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, EMBED_DIM)
        )

        # ----------------------------------------------------
        # Distillation Token
        # ----------------------------------------------------

        self.dist_token = nn.Parameter(
            torch.zeros(1, 1, EMBED_DIM)
        )

        # ----------------------------------------------------
        # Position Embedding
        # ----------------------------------------------------

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 2,
                EMBED_DIM
            )
        )

        self.pos_dropout = nn.Dropout(
            DROPOUT
        )

        # ----------------------------------------------------
        # Transformer Encoder
        # ----------------------------------------------------

        self.blocks = nn.Sequential(

            *[
                TransformerBlock()
                for _ in range(DEPTH)
            ]

        )

        self.norm = nn.LayerNorm(
            EMBED_DIM
        )

        # ----------------------------------------------------
        # Classification Heads
        # ----------------------------------------------------

        self.head = nn.Linear(
            EMBED_DIM,
            num_classes
        )

        self.head_dist = nn.Linear(
            EMBED_DIM,
            num_classes
        )

        # ----------------------------------------------------
        # Initialization
        # ----------------------------------------------------

        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02
        )

        nn.init.trunc_normal_(
            self.dist_token,
            std=0.02
        )

        nn.init.trunc_normal_(
            self.pos_embed,
            std=0.02
        )

        self.apply(
            self._init_weights
        )

    # ========================================================
    # Weight Initialization
    # ========================================================

    @staticmethod
    def _init_weights(module):

        if isinstance(module, nn.Linear):

            nn.init.trunc_normal_(
                module.weight,
                std=0.02
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.LayerNorm):

            nn.init.ones_(
                module.weight
            )

            nn.init.zeros_(
                module.bias
            )

        elif isinstance(module, nn.Conv2d):

            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu"
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        batch_size = x.shape[0]

        # ----------------------------------------------------
        # Patch Embedding
        # ----------------------------------------------------

        x = self.patch_embed(x)

        # [B, 196, 384]

        # ----------------------------------------------------
        # Expand CLS / Dist Tokens
        # ----------------------------------------------------

        cls_token = self.cls_token.expand(
            batch_size, -1, -1
        )

        dist_token = self.dist_token.expand(
            batch_size, -1, -1
        )

        # ----------------------------------------------------
        # Concatenate Tokens
        # ----------------------------------------------------

        x = torch.cat(
            [
                cls_token,
                dist_token,
                x
            ],
            dim=1
        )

        # [B, 198, 384]

        # ----------------------------------------------------
        # Position Embedding
        # ----------------------------------------------------

        x = x + self.pos_embed

        x = self.pos_dropout(x)

        # ----------------------------------------------------
        # Transformer Encoder
        # ----------------------------------------------------

        x = self.blocks(x)

        x = self.norm(x)

        # ----------------------------------------------------
        # Extract CLS / Dist Tokens
        # ----------------------------------------------------

        cls_output = x[:, 0]

        dist_output = x[:, 1]

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        cls_logits = self.head(
            cls_output
        )

        dist_logits = self.head_dist(
            dist_output
        )

        return cls_logits, dist_logits


# ============================================================
# 6. Teacher ResNet-18
# ============================================================

class TeacherResNet18(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):

        super().__init__()

        self.backbone = resnet18(
            weights=None
        )

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):

        return self.backbone(x)


# ============================================================
# 7. Build Student
# ============================================================

def build_model():

    return DeiT()


# ============================================================
# 8. Build Teacher
# ============================================================

def build_teacher(checkpoint_path, device):

    import os

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            f"Teacher checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    teacher = TeacherResNet18()

    teacher.load_state_dict(
        checkpoint["model_state_dict"]
    )

    teacher = teacher.to(device)

    teacher.eval()

    # Freeze all parameters
    for parameter in teacher.parameters():

        parameter.requires_grad = False

    return teacher, checkpoint["class_to_idx"]
