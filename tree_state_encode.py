import torch
import torch.nn as nn


class SimpleTreeStateEncoder(nn.Module):
    """
    融合骨架、点云的双路编码器
    """

    def __init__(self, pc_feat_dim=3, sk_feat_dim=6, out_dim=64, hidden_dim=64):
        super().__init__()
        self.pc_mlp = nn.Sequential(
            nn.Linear(pc_feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.sk_mlp = nn.Sequential(
            nn.Linear(sk_feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        # 融合层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, out_dim)
        )

    def forward(self, pc, sk):
        # pc: [B, N_pc, 3], sk: [B, N_sk, 6]
        pc_feat = self.pc_mlp(pc)  # [B, N_pc, H]
        sk_feat = self.sk_mlp(sk)  # [B, N_sk, H]

        # 全局特征
        pc_pool = torch.max(pc_feat, 1)[0]  # [B, H]
        sk_pool = torch.max(sk_feat, 1)[0]  # [B, H]

        x = torch.cat([pc_pool, sk_pool], dim=-1)  # [B, H*2]
        x = self.fc(x)  # [B, out_dim]
        return x