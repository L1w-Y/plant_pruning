import torch
import numpy as np
import os
import pickle
import trimesh
from tqdm import tqdm
from scipy.spatial.distance import cdist  # 导入正确的距离计算工具

# 导入项目模块
from reward_fn import get_pruning_kpis
from tree_state_encode import SimpleTreeStateEncoder
from train import DecisionTransformer
from tree_utils import (
    read_pointclouds_ply, read_skeleton_txt, get_all_branch_indices,
    collect_all_paths, remove_branch_from_mesh,
    build_parent_to_children_graph, find_all_descendant_branches_optimized
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
OUTPUT_DIR = "pruning_visualization_mesh"


def create_color_coded_trunk_visualization(original_trunk_mesh, removed_branch_indices, skeleton, output_path):
    """在原始树干上，将被移除的分枝系统染成红色。"""
    print("\nCreating color-coded trunk visualization...")
    if not removed_branch_indices:
        print("  - Diagnostic: No branches were removed, so visualization file was not created.")
        return

    all_removed_points = []
    all_search_radii = []
    for branch_id in removed_branch_indices:
        branch_mask = skeleton[:, 5] == branch_id
        if np.any(branch_mask):
            all_removed_points.append(skeleton[branch_mask][:, :3])
            avg_radius = np.mean(skeleton[branch_mask][:, 3])
            all_search_radii.append(max(avg_radius * 2.5, 0.02))

    if not all_removed_points:
        print("  - Diagnostic: Could not find skeleton points for any removed branches.")
        return

    removed_skeleton_points = np.vstack(all_removed_points)
    search_radius = np.mean(all_search_radii)
    print(f"  - Using a dynamic search radius of: {search_radius:.3f}m")

    # --- 终极修复：使用 scipy.spatial.cdist 计算正确的距离 ---
    # 1. 计算每个网格顶点(N)到每个被移除骨架点(M)的距离矩阵 (NxM)
    distance_matrix = cdist(original_trunk_mesh.vertices, removed_skeleton_points)

    # 2. 对每一行（每个顶点）找到到所有骨架点的最小距离
    min_distances_to_skeleton = np.min(distance_matrix, axis=1)

    # 3. 现在 min_distances_to_skeleton 的长度与顶点数完全匹配
    close_vertices_mask = min_distances_to_skeleton < search_radius

    num_colored_vertices = np.sum(close_vertices_mask)
    if num_colored_vertices == 0:
        print("  - Diagnostic: No mesh vertices found near the removed skeleton points.");
        return
    print(f"  - Found {num_colored_vertices} vertices to color red.")

    colors = np.full((len(original_trunk_mesh.vertices), 4), [128, 128, 128, 255], dtype=np.uint8)
    colors[close_vertices_mask] = [255, 0, 0, 255]

    color_coded_mesh = original_trunk_mesh.copy()
    color_coded_mesh.visual.vertex_colors = colors
    color_coded_mesh.export(output_path)
    print(f"  - Color-coded visualization saved to {output_path}")


def run_visual_evaluation():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # --- 1. 加载模型 ---
    print("Calculating action space size...");
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
    pc_path, sk_path, obj_path_dict = pc_files[TEST_TREE_INDEX], sk_files[TEST_TREE_INDEX], obj_files[TEST_TREE_INDEX]
    print(f"Loading test tree: {os.path.dirname(os.path.dirname(pc_path))}")

    # --- 3. 加载初始数据和网格 ---
    sk_initial_raw = np.loadtxt(sk_path)
    pc_initial_raw = read_pointclouds_ply(pc_path)
    leaf_mesh_original = trimesh.load(obj_path_dict['leaf'], force='mesh')
    trunk_mesh_original = trimesh.load(obj_path_dict['trunk'], force='mesh')
    leaf_mesh_original.export(os.path.join(OUTPUT_DIR, "leaf_original.ply"))
    trunk_mesh_original.export(os.path.join(OUTPUT_DIR, "trunk_original.ply"))

    parent_to_children_graph = build_parent_to_children_graph(sk_initial_raw)

    # --- 4. 剪枝前KPI评分 ---
    print("\n--- Initial Tree KPIs (Before Pruning) ---")
    initial_kpis = get_pruning_kpis(sk_initial_raw, pc_initial_raw)
    for key, val in initial_kpis.items(): print(f"  - {key}: {val:.3f}")

    # --- 5. 模拟剪枝 ---
    STOP_ACTION_ID = int(np.max(get_all_branch_indices(sk_initial_raw))) + 1
    print(f"\nDetermined STOP_ACTION_ID: {STOP_ACTION_ID}")

    target_rtg = torch.tensor([[20.0]], dtype=torch.float32, device=DEVICE)
    states = torch.zeros(1, MAX_EPISODE_LEN, STATE_DIM, device=DEVICE)
    actions = torch.zeros(1, MAX_EPISODE_LEN, dtype=torch.long, device=DEVICE)
    rtgs = torch.zeros(1, MAX_EPISODE_LEN, 1, device=DEVICE)

    sk_current_raw = sk_initial_raw.copy()
    leaf_mesh_pruned = leaf_mesh_original.copy()
    trunk_mesh_pruned = trunk_mesh_original.copy()
    removed_branch_ids_history = set()

    print("\n--- Starting Intelligent Pruning ---")
    for t in tqdm(range(MAX_EPISODE_LEN), desc="Pruning Steps"):
        available_actions = get_all_branch_indices(sk_current_raw).tolist()
        if t > 0: available_actions.append(STOP_ACTION_ID)
        if not available_actions: print("No more branches to prune."); break

        state_vec = encoder(
            torch.from_numpy(pc_initial_raw).float().unsqueeze(0).to(DEVICE),
            torch.from_numpy(sk_current_raw[:SK_MAX_POINTS]).float().unsqueeze(0).to(DEVICE)
        ).squeeze(0)
        states[0, t] = state_vec
        rtgs[0, t] = target_rtg

        with torch.no_grad():
            action_logits = model(states, actions, rtgs)
        pred_probs = torch.softmax(action_logits[0, t], dim=0)
        mask = torch.zeros_like(pred_probs);
        mask[available_actions] = 1
        pred_probs *= mask

        if pred_probs.sum() == 0: print("No valid action predicted."); break
        pred_action = torch.argmax(pred_probs).item()

        print(f"\nStep {t + 1}: Model chose action {pred_action}.")
        if pred_action == STOP_ACTION_ID:
            print("  - Action is STOP. Halting pruning.")
            break

        descendants = find_all_descendant_branches_optimized(parent_to_children_graph, pred_action)
        all_branches_to_remove = {pred_action}.union(descendants)
        removed_branch_ids_history.update(all_branches_to_remove)

        leaf_mesh_pruned = remove_branch_from_mesh(leaf_mesh_pruned, sk_current_raw, pred_action)
        trunk_mesh_pruned = remove_branch_from_mesh(trunk_mesh_pruned, sk_current_raw, pred_action)

        sk_mask = ~np.isin(sk_current_raw[:, 5], list(all_branches_to_remove))
        sk_current_raw = sk_current_raw[sk_mask]
        target_rtg -= 0.5

    # --- 6. 保存与评估 ---
    print("\n--- Pruning Finished ---")
    print(f"Total branches removed: {len(removed_branch_ids_history)}")
    leaf_mesh_pruned.export(os.path.join(OUTPUT_DIR, "leaf_final_pruned.ply"))
    trunk_mesh_pruned.export(os.path.join(OUTPUT_DIR, "trunk_final_pruned.ply"))
    print("Saved final pruned leaf and trunk meshes.")

    print("\n--- Final Tree KPIs (After Pruning) ---")
    final_full_mesh = trimesh.util.concatenate(leaf_mesh_pruned, trunk_mesh_pruned)
    pc_final_raw = final_full_mesh.sample(pc_initial_raw.shape[0]) if not final_full_mesh.is_empty else np.array([])
    final_kpis = get_pruning_kpis(sk_current_raw, pc_final_raw)
    for key, val in final_kpis.items():
        change = val - initial_kpis.get(key, 0)
        print(f"  - {key}: {val:.3f} (Change: {change:+.3f})")

    # --- 7. 创建可视化文件 ---
    create_color_coded_trunk_visualization(
        trunk_mesh_original,
        list(removed_branch_ids_history),
        sk_initial_raw,
        os.path.join(OUTPUT_DIR, "trunk_pruning_visualization.ply")
    )

    print(f"\nVisual evaluation finished. Check the '{OUTPUT_DIR}' folder.")


if __name__ == "__main__":
    run_visual_evaluation()