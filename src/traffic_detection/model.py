from __future__ import annotations

import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class PacketCNN(nn.Module):
    """Compact 1D CNN for 30 signed packet lengths.

    Packet magnitude, direction and position are represented separately. Zero is
    reserved for padding, so padded positions cannot be confused with TCP data.
    """

    def __init__(
        self,
        num_classes: int,
        max_packet_size: int = 2048,
        size_embedding_dim: int = 32,
        hidden_dim: int = 96,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.max_packet_size = max_packet_size
        self.size_embedding = nn.Embedding(
            max_packet_size + 2, size_embedding_dim, padding_idx=0
        )
        self.direction_embedding = nn.Embedding(3, 4, padding_idx=0)
        self.position_embedding = nn.Embedding(30, 8)

        input_dim = size_embedding_dim + 4 + 8 + 3
        self.projection = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            ResidualConvBlock(hidden_dim, kernel_size=3, dropout=dropout / 2),
            ResidualConvBlock(hidden_dim, kernel_size=5, dropout=dropout / 2),
            ResidualConvBlock(
                hidden_dim, kernel_size=3, dropout=dropout / 2, dilation=2
            ),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, packets: torch.Tensor) -> torch.Tensor:
        packets = packets.long()
        mask = packets.ne(0)
        magnitude = packets.abs().clamp_max(self.max_packet_size) + 1
        magnitude = magnitude.masked_fill(~mask, 0)
        # 0=padding, 1=server/negative, 2=client/positive.
        direction = torch.where(mask, packets.gt(0).long() + 1, 0)

        positions = torch.arange(30, device=packets.device).expand(packets.size(0), -1)
        size_emb = self.size_embedding(magnitude)
        direction_emb = self.direction_embedding(direction)
        position_emb = self.position_embedding(positions)
        normalized_size = (packets.float() / float(self.max_packet_size)).unsqueeze(-1)
        normalized_abs = normalized_size.abs()
        mask_channel = mask.float().unsqueeze(-1)

        x = torch.cat(
            [
                size_emb,
                direction_emb,
                position_emb,
                normalized_size,
                normalized_abs,
                mask_channel,
            ],
            dim=-1,
        )
        x = self.projection(x.transpose(1, 2))
        x = self.encoder(x)

        float_mask = mask.unsqueeze(1).float()
        denom = float_mask.sum(dim=2).clamp_min(1.0)
        avg_pool = (x * float_mask).sum(dim=2) / denom
        max_pool = x.masked_fill(~mask.unsqueeze(1), -1e4).amax(dim=2)
        positive = packets.gt(0)
        negative = packets.lt(0)
        positive_count = positive.sum(dim=1, keepdim=True).float() / 30.0
        negative_count = negative.sum(dim=1, keepdim=True).float() / 30.0
        positive_bytes = packets.clamp_min(0).sum(dim=1, keepdim=True).float()
        negative_bytes = -packets.clamp_max(0).sum(dim=1, keepdim=True).float()
        byte_scale = float(30 * self.max_packet_size)
        direction_changes = (
            mask[:, 1:]
            & mask[:, :-1]
            & packets[:, 1:].sign().ne(packets[:, :-1].sign())
        ).sum(dim=1, keepdim=True).float() / 29.0
        global_features = torch.cat(
            [
                denom / 30.0,
                positive_count,
                negative_count,
                positive_bytes / byte_scale,
                negative_bytes / byte_scale,
                direction_changes,
            ],
            dim=1,
        )
        return self.classifier(torch.cat([avg_pool, max_pool, global_features], dim=1))

