import numpy as np
from scipy.spatial.distance import pdist, squareform


# --- 辅助函数：用于计算各种量化指标 ---

def get_branch_properties(sk, branch_id):
    """获取单个分枝的详细属性：点、长度、平均半径、向量等。"""
    mask = sk[:, 5] == branch_id
    points = sk[mask][:, :3]
    if points.shape[0] < 2:
        return None

    radii = sk[mask][:, 3]
    vector = points[-1] - points[0]
    length = np.linalg.norm(vector)

    return {
        "points": points,
        "vector": vector / (length + 1e-6),
        "start_point": points[0],
        "end_point": points[-1],
        "length": length,
        "avg_radius": np.mean(radii)
    }


def get_structural_hierarchy_score(sk):
    """R3.1 - 计算结构层次得分。下层主枝应比上层更长更粗。"""
    branch_ids = np.unique(sk[:, 5])
    if len(branch_ids) < 2: return 0.5

    properties = [get_branch_properties(sk, bid) for bid in branch_ids]
    properties = [p for p in properties if p is not None and p['length'] > 0.1]  # 过滤掉无效或太小的分枝
    if len(properties) < 2: return 0.5

    # 按Z坐标（高度）排序
    properties.sort(key=lambda p: p['start_point'][2])

    violations = 0
    comparisons = 0
    for i in range(len(properties)):
        for j in range(i + 1, len(properties)):
            # 比较第i个（下层）和第j个（上层）分枝
            lower_branch = properties[i]
            upper_branch = properties[j]

            # 规则：下层应更长、更粗
            if lower_branch['length'] < upper_branch['length']: violations += 1
            if lower_branch['avg_radius'] < upper_branch['avg_radius']: violations += 1
            comparisons += 2

    if comparisons == 0: return 0.5
    return 1.0 - (violations / comparisons)


def get_crowding_score(sk, threshold_cm=10.0):
    """R3.2 - 计算拥挤度得分。分枝间距越小，得分越低。"""
    branch_ids = np.unique(sk[:, 5])
    if len(branch_ids) < 2: return 1.0

    branch_centers = np.array([sk[sk[:, 5] == bid][:, :3].mean(axis=0) for bid in branch_ids])
    dist_matrix = squareform(pdist(branch_centers))
    np.fill_diagonal(dist_matrix, np.inf)

    # 将厘米转换为米
    threshold_m = threshold_cm / 100.0

    # 距离小于阈值的分枝对被认为是拥挤的
    crowded_pairs = dist_matrix < threshold_m
    # 惩罚与拥挤对的数量成正比
    penalty = np.sum(crowded_pairs) / (len(branch_ids) ** 2)
    return 1.0 - penalty


def get_balance_score(pc, sk):
    """R3.3 - 计算树体平衡得分。"""
    if pc.shape[0] < 10 or sk.shape[0] == 0: return 0.5
    trunk_base = sk[np.argmin(sk[:, 2])][:2]
    center_of_mass = pc[:, :2].mean(axis=0)
    distance = np.linalg.norm(trunk_base - center_of_mass)
    bbox_size_2d = np.linalg.norm(pc[:, :2].max(axis=0) - pc[:, :2].min(axis=0))
    return 1.0 - np.clip(distance / (bbox_size_2d * 0.2 + 1e-6), 0, 1)


# --- 主奖励函数 ---

# (reward_fn.py 中的辅助函数保持不变)

def reward_fn(sk_before, pc_before, sk_after, pc_after, action_branch_id, stop_action_id=None, initial_branch_count=None):
    # --- 1. 对被剪掉的枝条本身进行评价 (规则2) ---
    w_action = 4.0
    reward_for_action = 0
    size_bonus = 0
    prop = get_branch_properties(sk_before, action_branch_id)
    if stop_action_id is not None and action_branch_id == stop_action_id:
        # STOP_ACTION奖励
        if initial_branch_count is not None:
            final_branch_count = len(np.unique(sk_after[:, 5]))
            pruned_ratio = (initial_branch_count - final_branch_count) / (initial_branch_count + 1e-6)
        else:
            initial_branch_count = len(np.unique(sk_before[:, 5]))
            final_branch_count = len(np.unique(sk_after[:, 5]))
            pruned_ratio = (initial_branch_count - final_branch_count) / (initial_branch_count + 1e-6)
        if 0.18 <= pruned_ratio <= 0.32:
            return 2.0
        elif pruned_ratio < 0.13:
            return -1.5
        else:
            return -2.0
    if prop:

        size_bonus = prop['length'] * prop['avg_radius'] * 100.0  # 乘以100进行缩放

        tree_center = sk_before[:, :3].mean(axis=0)
        center_to_branch_start = prop['start_point'] - tree_center

        # R1.3 - 内膛枝
        dot_inward = np.dot(prop['vector'], center_to_branch_start / (np.linalg.norm(center_to_branch_start) + 1e-6))
        if dot_inward < -0.2 and prop['length'] > 0.15:
            reward_for_action += 1.0  # 基础分提高到1.0

        # R1.4 - 直立枝
        dot_upright = np.dot(prop['vector'], np.array([0, 0, 1]))
        if dot_upright > np.cos(np.deg2rad(30)):
            reward_for_action += 0.6

        # R1.2 - 下垂枝
        if dot_upright < np.sin(np.deg2rad(-20)):
            reward_for_action += 0.4

    # 将尺寸奖励加到动作奖励上
    final_action_reward = w_action * (reward_for_action + size_bonus)

    # --- 2. 评估剪枝后状态的改善程度 (规则3) ---
    # 权重也进行调整
    delta_reward = 0
    delta_reward += (get_structural_hierarchy_score(sk_after) - get_structural_hierarchy_score(sk_before)) * 1.0  # 权重降低
    delta_reward += (get_crowding_score(sk_after) - get_crowding_score(sk_before)) * 2.5  # 权重提高
    delta_reward += (get_balance_score(pc_after, sk_after) - get_balance_score(pc_before, sk_before)) * 1.5  # 权重提高

    # --- 3. 约束与惩罚 (规则4) ---
    penalty = 0
    if prop and prop['avg_radius'] > np.mean(sk_before[:, 3]) * 1.5:
        penalty -= 0.5

    vol_before = pc_before.shape[0]
    vol_after = pc_after.shape[0]
    volume_reduction_ratio = (vol_before - vol_after) / (vol_before + 1e-6)
    if volume_reduction_ratio > 0.20:  # 容忍度放宽到20%
        penalty -= (volume_reduction_ratio - 0.20) * 2.0  # 惩罚力度减小

    penalty -= 0.05

    # --- 最终综合得分 ---
    final_reward = final_action_reward + delta_reward + penalty

    return float(np.clip(final_reward, -3.0, 3.0))  # 奖励范围也扩大

def get_pruning_kpis(sk, pc):
    """计算一棵树的所有关键性能指标(KPIs)"""
    if sk.shape[0] == 0 or pc.shape[0] == 0:
        return {"branch_count": 0, "volume": 0, "hierarchy": 0, "crowding": 0, "balance": 0}
    return {
        "branch_count": len(np.unique(sk[:, 5])),
        "volume": pc.shape[0],
        "hierarchy": get_structural_hierarchy_score(sk),
        "crowding": get_crowding_score(sk),
        "balance": get_balance_score(pc, sk)
    }


def get_branch_badness_report(sk, branch_id):
    """为一根指定的枝条生成一份“坏度”分析报告。"""
    report = {"is_bad": False, "reasons": []}
    prop = get_branch_properties(sk, branch_id)
    if not prop:
        return report


    # 检查是否为内膛枝
    tree_center = sk[:, :3].mean(axis=0)
    center_to_branch_start = prop['start_point'] - tree_center
    dot_inward = np.dot(prop['vector'], center_to_branch_start / (np.linalg.norm(center_to_branch_start) + 1e-6))
    if dot_inward < -0.2 and prop['length'] > 0.15:
        report["is_bad"] = True
        report["reasons"].append(f"Inward-growing (score: {dot_inward:.2f})")

    # 检查是否为直立枝
    dot_upright = np.dot(prop['vector'], np.array([0, 0, 1]))
    if dot_upright > np.cos(np.deg2rad(30)):
        report["is_bad"] = True
        report["reasons"].append(f"Upright (angle vs vertical < 30°)")

    # 检查是否为下垂枝
    if dot_upright < np.sin(np.deg2rad(-20)):
        report["is_bad"] = True
        report["reasons"].append(f"Drooping (angle vs horizontal < -20°)")

    # 检查是否为粗大骨干枝
    if prop['avg_radius'] > np.mean(sk[:, 3]) * 1.5:
        report["reasons"].append("Is a major structural branch")

    return report