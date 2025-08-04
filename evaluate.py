import torch
import numpy as np
import os
import pickle
import open3d as o3d
from tqdm import tqdm

# 导入项目模块
from reward_fn import get_pruning_kpis  # 我们可能需要稍微调整它
from tree_state_encode import SimpleTreeStateEncoder
from train import DecisionTransformer
from tree_utils import (
    read_pointclouds_ply, read_skeleton_txt, get_all_branch_indices,
    collect_all_paths, build_parent_to_children_graph,
    find_all_descendant_branches_optimized
)

# --- 配置参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_DIM = 64
MAX_EPISODE_LEN = 40
PC_MAX_POINTS = 2048
SK_MAX_POINTS = 256
MODEL_PATH = "decision_transformer_final.pt"
LEMON_ROOT = r"D:\DownLoad\baiduyunDownload\9lemon\00lemon"
TEST_TREE_INDEX = 15
OUTPUT_DIR = "pruning_visualization_skeleton"  # 新的输出目录


def save_skeleton_as_colored_ply(skeleton_points, color, file_path):
    """将骨架坐标点保存为带颜色的PLY点云文件。"""
    if skeleton_points.shape[0] == 0:
        print(f"  - No points to save for {os.path.basename(file_path)}.")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(skeleton_points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array(color), (skeleton_points.shape[0], 1)))
    o3d.io.write_point_cloud(file_path, pcd)
    print(f"  - Saved skeleton visualization to {file_path}")


def run_skeleton_evaluation():
    """
    全新的评估流程，只操作骨架，不涉及网格修改。
    """
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # --- 1. 加载模型 (与之前相同) ---
    print("Loading model and calculating action space...");
    try:
        with open("lemon_pruning_rl_episodes.pkl", "rb") as f:
            random_episodes = pickle.load(f)
        with open("lemon_pruning_EXPERT_episodes.pkl", "rb") as f:
            expert_episodes = pickle.load(f)
        episodes_for_action_calc = random_episodes + expert_episodes
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found."); return
    actions_all = [step['action'] for epi in episodes_for_action_calc for step in epi]
    N_ACTIONS = max(actions_all) + 1 if actions_all else 0
    model = DecisionTransformer(state_dim=STATE_DIM, act_dim=N_ACTIONS, max_length=MAX_EPISODE_LEN).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE));
    model.eval()
    encoder = SimpleTreeStateEncoder(out_dim=STATE_DIM).to(DEVICE);
    encoder.eval()

    # --- 2. 加载数据 ---
    pc_files, sk_files, obj_files = collect_all_paths(LEMON_ROOT)
    if TEST_TREE_INDEX >= len(sk_files): print("Error: Test index out of bounds."); return
    pc_path, sk_path = pc_files[TEST_TREE_INDEX], sk_files[TEST_TREE_INDEX]
    print(f"Loading test tree: {os.path.dirname(os.path.dirname(sk_path))}")

    sk_initial_raw = np.loadtxt(sk_path)
    pc_initial_raw = read_pointclouds_ply(pc_path)  # 点云仅用于状态编码

    parent_to_children_graph = build_parent_to_children_graph(sk_initial_raw)

    # --- 3. 剪枝前KPI评分 ---
    print("\n--- Initial Skeleton KPIs ---")
    # 注意：这里的KPIs是基于骨架的，更精确
    initial_kpis = get_pruning_kpis(sk_initial_raw, pc_initial_raw)
    for key, val in initial_kpis.items(): print(f"  - {key}: {val:.3f}")

    # --- 4. 纯骨架模拟剪枝 ---
    STOP_ACTION_ID = int(np.max(get_all_branch_indices(sk_initial_raw))) + 1
    print(f"\nDetermined STOP_ACTION_ID: {STOP_ACTION_ID}")

    target_rtg = torch.tensor([[20.0]], dtype=torch.float32, device=DEVICE)
    states = torch.zeros(1, MAX_EPISODE_LEN, STATE_DIM, device=DEVICE)
    actions = torch.zeros(1, MAX_EPISODE_LEN, dtype=torch.long, device=DEVICE)
    rtgs = torch.zeros(1, MAX_EPISODE_LEN, 1, device=DEVICE)

    sk_current_raw = sk_initial_raw.copy()
    removed_branch_ids = set()

    print("\n--- Starting Skeleton-based Pruning Simulation ---")
    for t in tqdm(range(MAX_EPISODE_LEN), desc="Pruning Steps"):
        available_actions = get_all_branch_indices(sk_current_raw).tolist()
        if t > 0: available_actions.append(STOP_ACTION_ID)
        if not available_actions: print("No more branches to prune."); break

        state_vec = encoder(
            torch.from_numpy(pc_initial_raw).float().unsqueeze(0).to(DEVICE),
            torch.from_numpy(sk_current_raw[:SK_MAX_POINTS]).float().unsqueeze(0).to(DEVICE)
        ).squeeze(0)
        states[0, t], rtgs[0, t] = state_vec, target_rtg

        with torch.no_grad():
            action_logits = model(states, actions, rtgs)
        pred_probs = torch.softmax(action_logits[0, t], dim=0)
        mask = torch.zeros_like(pred_probs);
        mask[available_actions] = 1
        pred_probs *= mask

        if pred_probs.sum() == 0: print("No valid action predicted."); break
        pred_action = torch.argmax(pred_probs).item()

        if pred_action == STOP_ACTION_ID:
            print(f"\nStep {t + 1}: Model chose to STOP. Halting pruning.")
            break

        # --- 精确的骨架操作 ---
        descendants = find_all_descendant_branches_optimized(parent_to_children_graph, pred_action)
        all_branches_to_remove = {pred_action}.union(descendants)
        removed_branch_ids.update(all_branches_to_remove)

        sk_mask = ~np.isin(sk_current_raw[:, 5], list(all_branches_to_remove))
        sk_current_raw = sk_current_raw[sk_mask]
        target_rtg -= 0.5

    sk_final_raw = sk_current_raw

    # --- 5. 剪枝后KPI评分 ---
    print("\n--- Final Skeleton KPIs ---")
    final_kpis = get_pruning_kpis(sk_final_raw, pc_initial_raw)  # 点云还是用原始的，因为我们没有生成新点云
    for key, val in final_kpis.items():
        change = val - initial_kpis.get(key, 0)
        print(f"  - {key}: {val:.3f} (Change: {change:+.3f})")

    # --- 6. 生成骨架可视化文件 ---
    print("\n--- Generating Skeleton Visualizations ---")

    # 找到被移除的骨架点
    removed_mask = np.isin(sk_initial_raw[:, 5], list(removed_branch_ids))
    sk_removed_part = sk_initial_raw[removed_mask]

    save_skeleton_as_colored_ply(sk_initial_raw, [0.5, 0.5, 0.5],
                                 os.path.join(OUTPUT_DIR, "skeleton_original_GRAY.ply"))
    save_skeleton_as_colored_ply(sk_final_raw, [0, 1, 0], os.path.join(OUTPUT_DIR, "skeleton_final_GREEN.ply"))
    save_skeleton_as_colored_ply(sk_removed_part, [1, 0, 0], os.path.join(OUTPUT_DIR, "skeleton_pruned_part_RED.ply"))

    print(f"\nVisual evaluation finished. Check the '{OUTPUT_DIR}' folder.")


if __name__ == "__main__":
    run_skeleton_evaluation()