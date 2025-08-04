import os
import numpy as np
import pickle
import torch
from tqdm import tqdm

from reward_fn import reward_fn
from tree_state_encode import SimpleTreeStateEncoder
from tree_utils import (
    read_pointclouds_ply,
    read_skeleton_txt,
    get_all_branch_indices,
    remove_branch_from_skeleton,
    remove_branch_from_pointcloud,
    collect_all_paths
)

MAX_TRAJECTORY_LEN = 40
PC_MAX_POINTS = 2048
SK_MAX_POINTS = 256
STATE_DIM = 64

def generate_dynamic_trajectory(pc_path, sk_path, encoder, is_expert=False):
    try:
        pc_initial = read_pointclouds_ply(pc_path, max_points=PC_MAX_POINTS)
        sk_initial = read_skeleton_txt(sk_path, max_skeletons=SK_MAX_POINTS)
    except Exception as e:
        return None

    if pc_initial.shape[0] == 0 or sk_initial.shape[0] == 0:
        return None

    initial_branch_ids = get_all_branch_indices(sk_initial)
    initial_branch_count = len(initial_branch_ids)
    if initial_branch_count < 5: return None

    target_prune_ratio = np.random.uniform(0.15, 0.35)
    target_pruned_count = int(initial_branch_count * target_prune_ratio)

    # STOP_ACTION_ID = 最大分枝 id + 1
    STOP_ACTION_ID = max(initial_branch_ids) + 1

    trajectory = []
    pc_current = pc_initial.copy()
    sk_current = sk_initial.copy()
    steps = 0
    while steps < target_pruned_count and steps < MAX_TRAJECTORY_LEN:
        if steps >= MAX_TRAJECTORY_LEN: break
        available_branches = get_all_branch_indices(sk_current)
        if len(available_branches) < 2: break

        pc_tensor = torch.from_numpy(pc_current).float().unsqueeze(0)
        sk_tensor = torch.from_numpy(sk_current).float().unsqueeze(0)
        state_vec = encoder(pc_tensor, sk_tensor).squeeze(0).detach().numpy()

        if is_expert:
            from create_expert_dataset import find_worst_branch
            action = find_worst_branch(sk_current)
            if action is None: break
        else:
            action = np.random.choice(available_branches)
            if action == 0 and len(available_branches) > 1:
                action = np.random.choice([b for b in available_branches if b != 0])

        sk_next = remove_branch_from_skeleton(sk_current, action)
        pc_next = remove_branch_from_pointcloud(pc_current, sk_current, action)
        reward = reward_fn(sk_current, pc_current, sk_next, pc_next, action)

        trajectory.append({
            "state": state_vec, "action": int(action), "reward": reward, "done": False
        })

        pc_current, sk_current = pc_next, sk_next
        steps += 1

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
    save_path = "lemon_pruning_rl_episodes.pkl"  # 随机数据集

    pc_files, sk_files, _ = collect_all_paths(lemon_root)
    print(f"共找到 {len(pc_files)} 棵柠檬树的数据")

    encoder = SimpleTreeStateEncoder(out_dim=STATE_DIM)
    encoder.eval()

    all_episodes = []
    for i in tqdm(range(len(pc_files)), desc="Generating Random Dynamic Episodes"):
        episode = generate_dynamic_trajectory(pc_files[i], sk_files[i], encoder, is_expert=False)
        if episode:
            all_episodes.append(episode)

    print(f"\n成功生成 {len(all_episodes)} 条随机动态轨迹。")
    with open(save_path, "wb") as f:
        pickle.dump(all_episodes, f)
    print(f"\n数据集已保存到 {save_path}")