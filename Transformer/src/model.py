import copy
import math

import torch
import torch.nn as nn

import config


# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len,
        dropout
    ):

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        sequence_len = x.size(1)

        x = x + self.pe[:, :sequence_len, :]

        return self.dropout(x)


# ============================================================
# Scaled Dot-Product Attention
# ============================================================

class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout):

        super().__init__()

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key,
        value,
        mask=None
    ):

        d_k = query.size(-1)

        scores = torch.matmul(
            query,
            key.transpose(-2, -1)
        )

        scores = scores / math.sqrt(d_k)

        if mask is not None:

            scores = scores.masked_fill(
                ~mask,
                torch.finfo(scores.dtype).min
            )

        attention = torch.softmax(
            scores,
            dim=-1
        )

        attention = self.dropout(attention)

        output = torch.matmul(
            attention,
            value
        )

        return output, attention


# ============================================================
# Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        dropout
    ):

        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(
            d_model,
            d_model
        )

        self.w_k = nn.Linear(
            d_model,
            d_model
        )

        self.w_v = nn.Linear(
            d_model,
            d_model
        )

        self.w_o = nn.Linear(
            d_model,
            d_model
        )

        self.attention = ScaledDotProductAttention(
            dropout
        )

    def split_heads(self, x):

        batch_size = x.size(0)
        sequence_len = x.size(1)

        x = x.view(
            batch_size,
            sequence_len,
            self.num_heads,
            self.d_k
        )

        return x.transpose(1, 2)

    def combine_heads(self, x):

        batch_size = x.size(0)
        sequence_len = x.size(2)

        x = x.transpose(1, 2).contiguous()

        return x.view(
            batch_size,
            sequence_len,
            self.d_model
        )

    def forward(
        self,
        query,
        key,
        value,
        mask=None
    ):

        query = self.split_heads(
            self.w_q(query)
        )

        key = self.split_heads(
            self.w_k(key)
        )

        value = self.split_heads(
            self.w_v(value)
        )

        output, attention_weights = self.attention(
            query,
            key,
            value,
            mask
        )

        output = self.combine_heads(output)

        output = self.w_o(output)

        return output, attention_weights


# ============================================================
# Feed Forward Network
# ============================================================

class PositionwiseFeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff,
        dropout
    ):

        super().__init__()

        self.linear1 = nn.Linear(
            d_model,
            d_ff
        )

        self.linear2 = nn.Linear(
            d_ff,
            d_model
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.linear1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.linear2(x)

        return x


# ============================================================
# Encoder Layer
# ============================================================

class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout
    ):

        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model,
            num_heads,
            dropout
        )

        self.feed_forward = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)

        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x,
        source_mask
    ):

        attention_output, _ = self.self_attention(
            x,
            x,
            x,
            source_mask
        )

        x = self.norm1(
            x + self.dropout1(attention_output)
        )

        ff_output = self.feed_forward(x)

        x = self.norm2(
            x + self.dropout2(ff_output)
        )

        return x


# ============================================================
# Decoder Layer
# ============================================================

class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout
    ):

        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model,
            num_heads,
            dropout
        )

        self.cross_attention = MultiHeadAttention(
            d_model,
            num_heads,
            dropout
        )

        self.feed_forward = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.norm2 = nn.LayerNorm(d_model)

        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)

        self.dropout2 = nn.Dropout(dropout)

        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x,
        encoder_output,
        target_mask,
        source_mask
    ):

        # Masked self-attention

        attention_output, _ = self.self_attention(
            x,
            x,
            x,
            target_mask
        )

        x = self.norm1(
            x + self.dropout1(attention_output)
        )

        # Encoder-Decoder attention

        attention_output, _ = self.cross_attention(
            x,
            encoder_output,
            encoder_output,
            source_mask
        )

        x = self.norm2(
            x + self.dropout2(attention_output)
        )

        # Feed Forward Network

        ff_output = self.feed_forward(x)

        x = self.norm3(
            x + self.dropout3(ff_output)
        )

        return x


# ============================================================
# Encoder
# ============================================================

class Encoder(nn.Module):

    def __init__(
        self,
        embedding,
        positional_encoding,
        num_layers
    ):

        super().__init__()

        self.embedding = embedding

        self.positional_encoding = positional_encoding

        encoder_layer = EncoderLayer(
            config.D_MODEL,
            config.NUM_HEADS,
            config.D_FF,
            config.DROPOUT
        )

        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        source,
        source_mask
    ):

        x = self.embedding(source)

        x = x * math.sqrt(config.D_MODEL)

        x = self.positional_encoding(x)

        for layer in self.layers:

            x = layer(
                x,
                source_mask
            )

        return x


# ============================================================
# Decoder
# ============================================================

class Decoder(nn.Module):

    def __init__(
        self,
        embedding,
        positional_encoding,
        num_layers
    ):

        super().__init__()

        self.embedding = embedding

        self.positional_encoding = positional_encoding

        decoder_layer = DecoderLayer(
            config.D_MODEL,
            config.NUM_HEADS,
            config.D_FF,
            config.DROPOUT
        )

        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        target,
        encoder_output,
        target_mask,
        source_mask
    ):

        x = self.embedding(target)

        x = x * math.sqrt(config.D_MODEL)

        x = self.positional_encoding(x)

        for layer in self.layers:

            x = layer(
                x,
                encoder_output,
                target_mask,
                source_mask
            )

        return x


# ============================================================
# Transformer
# ============================================================

class Transformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            config.VOCAB_SIZE,
            config.D_MODEL,
            padding_idx=config.PAD_ID
        )

        self.positional_encoding = PositionalEncoding(
            config.D_MODEL,
            config.MAX_POSITIONAL_ENCODING,
            config.DROPOUT
        )

        self.encoder = Encoder(
            self.embedding,
            self.positional_encoding,
            config.NUM_ENCODER_LAYERS
        )

        self.decoder = Decoder(
            self.embedding,
            self.positional_encoding,
            config.NUM_DECODER_LAYERS
        )

        self.generator = nn.Linear(
            config.D_MODEL,
            config.VOCAB_SIZE,
            bias=False
        )

        # Tie output projection weights with token embedding
        self.generator.weight = self.embedding.weight

    def make_source_mask(self, source):

        # [B, 1, 1, S]

        return (
            source != config.PAD_ID
        ).unsqueeze(1).unsqueeze(2)

    def make_target_mask(self, target):

        batch_size, target_len = target.shape

        # Padding mask: [B, 1, 1, T]

        padding_mask = (
            target != config.PAD_ID
        ).unsqueeze(1).unsqueeze(2)

        # Causal mask: [1, 1, T, T]

        causal_mask = torch.tril(
            torch.ones(
                target_len,
                target_len,
                device=target.device,
                dtype=torch.bool
            )
        ).unsqueeze(0).unsqueeze(0)

        return padding_mask & causal_mask

    def encode(self, source):

        source_mask = self.make_source_mask(source)

        return self.encoder(
            source,
            source_mask
        )

    def decode(
        self,
        target,
        encoder_output,
        source_mask
    ):

        target_mask = self.make_target_mask(target)

        return self.decoder(
            target,
            encoder_output,
            target_mask,
            source_mask
        )

    def forward(
        self,
        source,
        target
    ):

        source_mask = self.make_source_mask(source)

        target_mask = self.make_target_mask(target)

        encoder_output = self.encoder(
            source,
            source_mask
        )

        decoder_output = self.decoder(
            target,
            encoder_output,
            target_mask,
            source_mask
        )

        logits = self.generator(
            decoder_output
        )

        return logits


def build_model():

    model = Transformer()

    return model