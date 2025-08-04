import os
import numpy as np
import pickle
import torch
from tqdm import tqdm

from reward_fn import reward_fn, get_branch_properties  # 导入辅助函数
from tree_state_encode import SimpleTreeStateEncoder
from tree_utils import (
    read_pointclouds_ply, read_skeleton_txt, get_all_branch_indices,
    remove_branch_from_skeleton, remove_branch_from_pointcloud,
    collect_all_paths
)

# --- 超参数 ---
MAX_PRUNE_STEPS = 40
PC_MAX_POINTS = 2048
SK_MAX_POINTS = 256
STATE_DIM = 64


def find_worst_branch(sk):
    """专家规则：找到当前树上“最差”的一根分枝。"""
    branch_ids = get_all_branch_indices(sk)
    if branch_ids is None or len(branch_ids) == 0:  # 如果还可能是 None 的话
        return None

    worst_score = -np.inf
    worst_branch_id = None

    for bid in branch_ids:
        if bid == 0: continue  # 从不剪主干

        # 使用 reward_fn 的逻辑来评估单根枝条的“坏度”
        # 我们模拟剪掉它，但只看 action_reward 部分
        # 这是一种简化的启发式方法
        score = 0
        prop = get_branch_properties(sk, bid)
        if prop:
            tree_center = sk[:, :3].mean(axis=0)
            center_to_branch_start = prop['start_point'] - tree_center
            dot_inward = np.dot(prop['vector'],
                                center_to_branch_start / (np.linalg.norm(center_to_branch_start) + 1e-6))
            if dot_inward < -0.2 and prop['length'] > 0.15: score += 1.0

            dot_upright = np.dot(prop['vector'], np.array([0, 0, 1]))
            if dot_upright > np.cos(np.deg2rad(30)): score += 0.6
            if dot_upright < np.sin(np.deg2rad(-20)): score += 0.4

        if score > worst_score:
            worst_score = score
            worst_branch_id = bid

    # 如果没有找到明显坏的枝条，就随机选一个非主干的
    if worst_branch_id is None:
        available_branches = [b for b in branch_ids if b != 0]
        if not available_branches: return None
        return np.random.choice(available_branches)

    return worst_branch_id


def generate_expert_trajectory(pc_path, sk_path, encoder):
    try:
        pc_initial = read_pointclouds_ply(pc_path, max_points=PC_MAX_POINTS)
        sk_initial = read_skeleton_txt(sk_path, max_skeletons=SK_MAX_POINTS)
    except Exception as e:
        return None
    if pc_initial.shape[0] == 0 or sk_initial.shape[0] == 0: return None

    initial_branch_ids = get_all_branch_indices(sk_initial)
    if len(initial_branch_ids) < 5: return None
    initial_branch_count = len(initial_branch_ids)   # <<< 需要加这一行

    STOP_ACTION_ID = max(initial_branch_ids) + 1

    trajectory = []
    pc_current, sk_current = pc_initial.copy(), sk_initial.copy()

    for step in range(min(MAX_PRUNE_STEPS, MAX_PRUNE_STEPS)):
        action = find_worst_branch(sk_current)
        if action is None: break

        pc_tensor = torch.from_numpy(pc_current).float().unsqueeze(0)
        sk_tensor = torch.from_numpy(sk_current).float().unsqueeze(0)
        state_vec = encoder(pc_tensor, sk_tensor).squeeze(0).detach().numpy()

        sk_next = remove_branch_from_skeleton(sk_current, action)
        pc_next = remove_branch_from_pointcloud(pc_current, sk_current, action)

        reward = reward_fn(sk_current, pc_current, sk_next, pc_next, action)

        trajectory.append({"state": state_vec, "action": int(action), "reward": reward, "done": False})
        pc_current, sk_current = pc_next, sk_next

    # 结束后加 STOP_ACTION
    pc_tensor = torch.from_numpy(pc_current).float().unsqueeze(0)
    sk_tensor = torch.from_numpy(sk_current).float().unsqueeze(0)
    state_vec = encoder(pc_tensor, sk_tensor).squeeze(0).detach().numpy()
    trajectory.append({
        "state": state_vec,
        "action": int(STOP_ACTION_ID),
        "reward": reward_fn(
            sk_initial, pc_initial, sk_current, pc_current, STOP_ACTION_ID,
            stop_action_id=STOP_ACTION_ID, initial_branch_count=initial_branch_count
        ),
        "done": True
    })

    if not trajectory: return None
    return trajectory


if __name__ == "__main__":
    lemon_root = r"D:\DownLoad\baiduyunDownload\9lemon\00lemon"
    save_path = "lemon_pruning_EXPERT_episodes.pkl"  # 保存为新文件

    pc_files, sk_files, _ = collect_all_paths(lemon_root)
    encoder = SimpleTreeStateEncoder(out_dim=STATE_DIM)
    encoder.eval()

    all_episodes = []
    for i in tqdm(range(len(pc_files)), desc="Generating Expert Episodes"):
        episode = generate_expert_trajectory(pc_files[i], sk_files[i], encoder)
        if episode: all_episodes.append(episode)

    print(f"\n成功生成 {len(all_episodes)} 条专家轨迹。")
    with open(save_path, "wb") as f:
        pickle.dump(all_episodes, f)
    print(f"\n专家数据集已保存到 {save_path}")