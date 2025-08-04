import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import re
import os

def read_pointclouds_ply(file_path, max_points=2048):
    """从.ply文件读取点云，并进行下采样或填充。"""
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
    except Exception as e:
        print(f"Warning: Cannot read point cloud {file_path}. Error: {e}. Returning zeros.")
        return np.zeros((max_points, 3), dtype=np.float32)

    N = points.shape[0]
    if N == 0:
        return np.zeros((max_points, 3), dtype=np.float32)

    # 如果点数过多，使用Open3D的体素下采样，比随机采样更能保留形状
    if N > max_points:
        # 计算合适的体素大小以接近目标点数
        voxel_size = (np.max(points, axis=0) - np.min(points, axis=0)).max() / 50
        pcd = pcd.voxel_down_sample(voxel_size)
        points = np.asarray(pcd.points)
        N = points.shape[0]

    # 随机选择或填充
    if N >= max_points:
        idx = np.random.choice(N, max_points, replace=False)
        sampled = points[idx]
    else:
        sampled = np.zeros((max_points, 3), dtype=np.float32)
        sampled[:N] = points

    return sampled.astype(np.float32)


def read_skeleton_txt(file_path, max_skeletons=256):
    """读取骨架文件，格式：x, y, z, radius, parent_id, branch_id。"""
    try:
        # 第6列(索引5)是branch_id，需要是整数
        arr = np.loadtxt(file_path, dtype=np.float32)
        if arr.ndim == 1:  # 如果只有一行
            arr = arr.reshape(1, -1)
    except Exception as e:
        print(f"Warning: Cannot read skeleton {file_path}. Error: {e}. Returning zeros.")
        return np.zeros((max_skeletons, 6), dtype=np.float32)

    N = arr.shape[0]
    if N == 0:
        return np.zeros((max_skeletons, 6), dtype=np.float32)

    if N >= max_skeletons:
        idx = np.random.choice(N, max_skeletons, replace=False)
        sampled = arr[idx]
    else:
        sampled = np.zeros((max_skeletons, 6), dtype=np.float32)
        sampled[:N] = arr

    return sampled


def get_all_branch_indices(skeleton_arr):
    """从骨架数据中获取所有唯一的分枝ID。"""
    if skeleton_arr.shape[0] == 0:
        return np.array([], dtype=int)
    # branch_id 是第6列 (索引5)
    return np.unique(skeleton_arr[:, 5].astype(int))


def remove_branch_from_skeleton(skeleton_arr, branch_index):
    """从骨架数组中移除指定ID的分枝。"""
    mask = skeleton_arr[:, 5] != branch_index
    return skeleton_arr[mask]


def remove_branch_from_pointcloud(pc_arr, skeleton_arr, branch_index_to_remove, radius_multiplier=3.0):
    """
    从点云中移除与指定骨架分枝关联的点。
    策略：找到属于该分枝的骨架点，然后移除这些骨架点周围一定半径内的所有点云点。
    """
    if pc_arr.shape[0] == 0 or skeleton_arr.shape[0] == 0:
        return pc_arr

    # 1. 找到要移除的分枝的所有骨架点及其平均半径
    branch_skeleton_points = skeleton_arr[skeleton_arr[:, 5] == branch_index_to_remove]
    if branch_skeleton_points.shape[0] == 0:
        return pc_arr  # 该分枝不存在

    branch_coords = branch_skeleton_points[:, :3]
    # 使用分枝的平均半径作为基础搜索半径
    avg_radius = np.mean(branch_skeleton_points[:, 3])
    search_radius = avg_radius * radius_multiplier

    # 2. 使用KDTree高效查找点云中邻近的点
    # 创建点云的KDTree
    pc_tree = KDTree(pc_arr)

    # 为分枝的每个骨架点查找其球形半径内的点云点索引
    indices_to_remove = pc_tree.query_ball_point(branch_coords, r=search_radius, p=2.0)

    # 将找到的索引列表展平并去重
    flat_indices_to_remove = np.unique([item for sublist in indices_to_remove for item in sublist])

    if len(flat_indices_to_remove) == 0:
        return pc_arr

    # 3. 创建一个mask来移除这些点
    mask = np.ones(pc_arr.shape[0], dtype=bool)
    mask[flat_indices_to_remove] = False

    return pc_arr[mask]


def collect_all_paths(lemon_root):
    """
    使用正则表达式收集所有 "LemonXX_XX" 格式的树木数据路径。
    """
    pc_files, sk_files, obj_files = [], [], []

    # 正则表达式匹配 "Lemon" + 两位数字 + "_" + 任意数字
    pattern = re.compile(r'Lemon\d{2}_\d+')

    if not os.path.isdir(lemon_root):
        print(f"Error: Root directory not found at {lemon_root}")
        return [], [], []

    for subdir in sorted(os.listdir(lemon_root)):
        if pattern.match(subdir):
            tree_dir = os.path.join(lemon_root, subdir)

            pc_path = os.path.join(tree_dir, "pointcloud", "mergePointClouds.ply")
            sk_path = os.path.join(tree_dir, "skeleton", "skeleton.txt")

            # 收集mesh文件路径
            leaf_obj_path = os.path.join(tree_dir, "meshmodel", "leavesmesh.obj")
            trunk_obj_path = os.path.join(tree_dir, "meshmodel", "trunkmesh.obj")

            if os.path.exists(pc_path) and os.path.exists(sk_path):
                pc_files.append(pc_path)
                sk_files.append(sk_path)

                # 确保两个obj文件都存在才添加
                if os.path.exists(leaf_obj_path) and os.path.exists(trunk_obj_path):
                    obj_files.append({'leaf': leaf_obj_path, 'trunk': trunk_obj_path})
                else:
                    obj_files.append(None)  # 用None占位，保持列表对齐

    return pc_files, sk_files, obj_files

def save_point_cloud(pc_arr, file_path):
    """将numpy点云数组保存为.ply文件。"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_arr)
    o3d.io.write_point_cloud(file_path, pcd)
    print(f"Saved point cloud to {file_path}")

def find_all_descendant_branches(skeleton_arr, root_branch_id):
    """
    给定一个根分枝ID，找到其所有的后代分枝。
    这是一个图遍历问题。
    """
    if skeleton_arr.shape[0] == 0:
        return set()

    # 步骤1: 构建一个从父分枝到子分枝的图
    # 我们需要先知道每个点属于哪个分枝
    point_idx_to_branch_id = {i: int(row[5]) for i, row in enumerate(skeleton_arr)}
    parent_to_children_graph = {}

    for point_idx, point_data in enumerate(skeleton_arr):
        current_branch_id = int(point_data[5])
        parent_point_id = int(point_data[4])  # 父点的ID (基于1的索引)

        # parent_point_id > 0 表示它不是根节点
        if parent_point_id > 0:
            parent_point_idx = parent_point_id - 1  # 转换为基于0的数组索引

            # 确保父点索引有效
            if parent_point_idx in point_idx_to_branch_id:
                parent_branch_id = point_idx_to_branch_id[parent_point_idx]

                # 如果一个点的父点属于不同的分枝，那么就建立了一条连接
                if parent_branch_id != current_branch_id:
                    if parent_branch_id not in parent_to_children_graph:
                        parent_to_children_graph[parent_branch_id] = set()
                    parent_to_children_graph[parent_branch_id].add(current_branch_id)

    # 步骤2: 从给定的 root_branch_id 开始进行广度优先搜索(BFS)
    all_descendants = set()
    queue = [root_branch_id]

    visited = {root_branch_id}

    while queue:
        current_branch = queue.pop(0)

        # 查找当前分枝的所有子分枝
        if current_branch in parent_to_children_graph:
            children = parent_to_children_graph[current_branch]
            for child in children:
                if child not in visited:
                    all_descendants.add(child)
                    visited.add(child)
                    queue.append(child)

    return all_descendants

def remove_branch_from_mesh(mesh, skeleton_arr, branch_index_to_remove, radius_multiplier=2.5):
    """从 trimesh 网格中移除与指定骨架分枝关联的顶点和面。"""
    from scipy.spatial.distance import cdist  # 在函数内部导入，减少全局依赖

    descendant_branches = find_all_descendant_branches(skeleton_arr, branch_index_to_remove)
    all_branches_to_remove = {branch_index_to_remove}.union(descendant_branches)

    mask = np.isin(skeleton_arr[:, 5], list(all_branches_to_remove))
    branch_system_skeleton_points = skeleton_arr[mask]

    if branch_system_skeleton_points.shape[0] == 0:
        return mesh

    branch_coords = branch_system_skeleton_points[:, :3]
    main_branch_points = skeleton_arr[skeleton_arr[:, 5] == branch_index_to_remove]
    avg_radius = np.mean(main_branch_points[:, 3]) if main_branch_points.shape[0] > 0 else 0.05
    search_radius = avg_radius * radius_multiplier

    # 计算网格顶点到分枝骨架点的最小距离
    distance_matrix = cdist(mesh.vertices, branch_coords)
    min_distances = np.min(distance_matrix, axis=1)

    # 找到距离小于搜索半径的顶点
    vertices_to_remove_mask = min_distances < search_radius

    # 找到所有顶点都被标记为移除的面
    face_vertex_masks = vertices_to_remove_mask[mesh.faces]
    faces_to_remove_mask = np.all(face_vertex_masks, axis=1)

    # 更新网格，只保留不被移除的面
    mesh.update_faces(~faces_to_remove_mask)
    mesh.remove_unreferenced_vertices()

    return mesh


# --- 请将这两个函数添加到 tree_utils.py 文件末尾 ---

def build_parent_to_children_graph(skeleton_arr):
    """
    构建一个从父分枝到子分枝的映射图，用于高效查询子分枝。
    """
    parent_to_children_graph = {}
    if skeleton_arr.shape[0] == 0:
        return parent_to_children_graph

    point_idx_to_branch_id = {i: int(row[5]) for i, row in enumerate(skeleton_arr)}
    for point_idx, point_data in enumerate(skeleton_arr):
        current_branch_id = int(point_data[5])
        parent_point_id = int(point_data[4])
        if parent_point_id > 0:
            parent_point_idx = parent_point_id - 1
            if parent_point_idx in point_idx_to_branch_id:
                parent_branch_id = point_idx_to_branch_id[parent_point_idx]
                if parent_branch_id != current_branch_id:
                    if parent_branch_id not in parent_to_children_graph:
                        parent_to_children_graph[parent_branch_id] = set()
                    parent_to_children_graph[parent_branch_id].add(current_branch_id)
    return parent_to_children_graph


def find_all_descendant_branches_optimized(parent_to_children_graph, root_branch_id):
    """
    使用广度优先搜索（BFS）高效查找所有后代分枝。
    """
    all_descendants = set()
    queue = [root_branch_id]
    visited = {root_branch_id}

    while queue:
        current_branch = queue.pop(0)
        if current_branch in parent_to_children_graph:
            children = parent_to_children_graph[current_branch]
            for child in children:
                if child not in visited:
                    all_descendants.add(child)
                    visited.add(child)
                    queue.append(child)
    return all_descendants