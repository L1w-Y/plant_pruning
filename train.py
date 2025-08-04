import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Model
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import csv
from scipy.ndimage import gaussian_filter1d
import os
# ==== 超参数 ====
STATE_DIM = 64
MAX_EPISODE_LEN = 40
ACTION_PAD_VALUE = -100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
N_EPOCH = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class LemonPruneTrajectoryDataset(Dataset):
    def __init__(self, episodes, max_len=MAX_EPISODE_LEN):
        self.episodes = episodes
        self.max_len = max_len

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode = self.episodes[idx]
        length = len(episode)
        # 修复：如果长度超过 max_len，截断
        if length > self.max_len:
            episode = episode[:self.max_len]
            length = self.max_len
        states = np.stack([step['state'] for step in episode])
        actions = np.array([step['action'] for step in episode])
        rewards = np.array([step['reward'] for step in episode])
        rtg = np.cumsum(rewards[::-1])[::-1]
        pad_len = self.max_len - length
        states = np.pad(states, ((0, pad_len), (0, 0)), 'constant')
        actions = np.pad(actions, (0, pad_len), 'constant', constant_values=ACTION_PAD_VALUE)
        rtg = np.pad(rtg, (0, pad_len), 'constant')
        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:length] = 1.0
        return {
            "states": torch.tensor(states, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "rtg": torch.tensor(rtg, dtype=torch.float32).unsqueeze(-1),
            "mask": torch.tensor(mask, dtype=torch.float32),
        }


# ==== 模型 (无需修改) ====
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, max_length, hidden_size=128, n_layer=3, n_head=4):
        super().__init__()
        self.state_embed = nn.Linear(state_dim, hidden_size)
        self.action_embed = nn.Embedding(act_dim, hidden_size)
        self.rtg_embed = nn.Linear(1, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length * 3, hidden_size))
        self.embed_ln = nn.LayerNorm(hidden_size)
        config = GPT2Config(
            n_embd=hidden_size, n_layer=n_layer, n_head=n_head,
            n_positions=max_length * 3, n_ctx=max_length * 3, vocab_size=1,
            attn_pdrop=0.1, resid_pdrop=0.1, embd_pdrop=0.1
        )
        self.transformer = GPT2Model(config)
        self.action_head = nn.Linear(hidden_size, act_dim)

    def forward(self, states, actions, rtg, mask=None):
        B, T, _ = states.shape
        state_embeddings = self.state_embed(states)
        action_embeddings = self.action_embed(torch.clamp(actions, 0))
        rtg_embeddings = self.rtg_embed(rtg)
        stacked_inputs = torch.stack((rtg_embeddings, state_embeddings, action_embeddings), dim=1) \
            .permute(0, 2, 1, 3).reshape(B, 3 * T, self.state_embed.out_features)
        stacked_inputs = self.embed_ln(stacked_inputs + self.pos_embed)
        if mask is not None:
            stacked_mask = torch.stack((mask, mask, mask), dim=1).permute(0, 2, 1).reshape(B, 3 * T)
        else:
            stacked_mask = None
        outputs = self.transformer(inputs_embeds=stacked_inputs, attention_mask=stacked_mask)
        x = outputs.last_hidden_state
        x = x.reshape(B, T, 3, self.state_embed.out_features).permute(0, 2, 1, 3)
        action_logits = self.action_head(x[:, 1])
        return action_logits


# ==== 训练主程序 ====
import csv  # ✅ 新增：用于保存 loss 日志
from scipy.ndimage import gaussian_filter1d  # ✅ 新增：用于平滑曲线

def train():
    # === 1. 加载两个数据集 ===
    print("Loading datasets...")
    with open("lemon_pruning_rl_episodes.pkl", "rb") as f:
        random_episodes = pickle.load(f)
    print(f"Loaded {len(random_episodes)} random episodes.")

    with open("lemon_pruning_EXPERT_episodes.pkl", "rb") as f:
        expert_episodes = pickle.load(f)
    print(f"Loaded {len(expert_episodes)} expert episodes.")

    episodes = random_episodes + expert_episodes
    print(f"Total episodes for training: {len(episodes)}")

    actions_all = [step['action'] for epi in episodes for step in epi]
    if not actions_all:
        print("错误：数据集中没有任何动作，无法继续训练。请检查数据生成过程。")
        return

    N_ACTIONS = max(actions_all) + 1  # 轨迹采集已经加 STOP_ACTION_ID
    print(f"Action space size (max branch id + 1): {N_ACTIONS}")

    dataset = LemonPruneTrajectoryDataset(episodes)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = DecisionTransformer(
        state_dim=STATE_DIM, act_dim=N_ACTIONS, max_length=MAX_EPISODE_LEN,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=ACTION_PAD_VALUE)

    total_steps = N_EPOCH * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # === 2. 准备记录 loss ===
    losses = []
    loss_csv_path = "loss_log.csv"
    with open(loss_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_loss"])  # 写入 CSV 表头

    print("\nStarting training...")

    # === 3. 训练循环 ===
    for epoch in range(N_EPOCH):
        model.train()
        total_loss = 0

        for batch in loader:
            states = batch["states"].to(DEVICE)
            actions = batch["actions"].to(DEVICE)
            rtg = batch["rtg"].to(DEVICE)

            logits = model(states, actions, rtg)
            loss = loss_fn(logits.reshape(-1, N_ACTIONS), actions.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)

        # === 4. 保存到 CSV 文件 ===
        with open(loss_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss])

        # 控制台打印
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{N_EPOCH}, Loss: {avg_loss:.4f}")

    # === 5. 模型保存 ===
    torch.save(model.state_dict(), "decision_transformer_final.pt")
    print("训练完成！模型已保存到 decision_transformer_final.pt")

    # === 6. 绘制 Loss 曲线 ===
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, N_EPOCH + 1), losses, label="Raw Loss", alpha=0.5)
    smoothed = gaussian_filter1d(losses, sigma=3)
    plt.plot(range(1, N_EPOCH + 1), smoothed, label="Smoothed Loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()



if __name__ == "__main__":
    train()
