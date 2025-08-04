import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入我们项目中的工具函数
from tree_utils import collect_all_paths, read_skeleton_txt, get_all_branch_indices


def analyze_dataset_branch_distribution(lemon_root):
    """
    分析数据集中所有树的可剪分枝数量，并生成报告和图表。
    """
    print(f"开始分析数据集，根目录: {lemon_root}")

    # 1. 收集所有骨架文件的路径
    pc_files, sk_files, _ = collect_all_paths(lemon_root)
    if not sk_files:
        print("错误：在指定目录下没有找到骨架文件。请检查路径。")
        return

    print(f"共找到 {len(sk_files)} 棵树的骨架数据。")

    branch_counts = []

    # 2. 遍历每一棵树，计算其可剪分枝数
    for sk_path in tqdm(sk_files, desc="Processing trees"):
        try:
            sk_data = read_skeleton_txt(sk_path)
            if sk_data.shape[0] == 0:
                continue

            all_ids = get_all_branch_indices(sk_data)

            # 定义“可剪分枝”：所有分枝 减去 主干（通常ID为0）
            # 我们检查0是否在列表中，如果在就移除，这样更健壮
            if 0 in all_ids:
                prunable_count = len(all_ids) - 1
            else:
                prunable_count = len(all_ids)

            if prunable_count > 0:
                branch_counts.append(prunable_count)

        except Exception as e:
            print(f"\n处理文件 {os.path.basename(sk_path)} 时出错: {e}")

    if not branch_counts:
        print("未能从任何文件中成功计算分枝数量。")
        return

    # 3. 将列表转换为NumPy数组以便进行统计分析
    branch_counts_np = np.array(branch_counts)

    # 4. 计算并打印统计报告
    print("\n--- 可剪分枝数量统计报告 ---")
    print(f"已成功分析树木数量: {len(branch_counts_np)}")
    print(f"最大分枝数: {np.max(branch_counts_np)}")
    print(f"最小分枝数: {np.min(branch_counts_np)}")
    print(f"平均分枝数: {np.mean(branch_counts_np):.2f}")
    print(f"中位数分枝数: {np.median(branch_counts_np)}")

    # 计算百分位数，这对于设定上限非常有帮助
    p75 = np.percentile(branch_counts_np, 75)
    p90 = np.percentile(branch_counts_np, 90)
    p95 = np.percentile(branch_counts_np, 95)
    p99 = np.percentile(branch_counts_np, 99)
    print(f"75%的树分枝数低于: {p75:.0f}")
    print(f"90%的树分枝数低于: {p90:.0f}")
    print(f"95%的树分枝数低于: {p95:.0f}")
    print(f"99%的树分枝数低于: {p99:.0f}")
    print("---------------------------------")

    # 5. 绘制直方图以实现可视化
    plt.figure(figsize=(12, 6))
    plt.hist(branch_counts_np, bins=range(int(np.min(branch_counts_np)), int(np.max(branch_counts_np)) + 2),
             edgecolor='black', alpha=0.7)
    plt.title('Distribution of Prunable Branch Counts Across the Dataset', fontsize=16)
    plt.xlabel('Number of Prunable Branches', fontsize=12)
    plt.ylabel('Number of Trees (Frequency)', fontsize=12)

    # 在图上标记出重要的统计数据
    plt.axvline(np.mean(branch_counts_np), color='red', linestyle='dashed', linewidth=2,
                label=f'Mean: {np.mean(branch_counts_np):.2f}')
    plt.axvline(np.median(branch_counts_np), color='green', linestyle='dashed', linewidth=2,
                label=f'Median: {np.median(branch_counts_np):.0f}')
    plt.axvline(p95, color='purple', linestyle='dashed', linewidth=2, label=f'95th Percentile: {p95:.0f}')

    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 保存图表
    save_fig_path = "branch_counts_distribution.png"
    plt.savefig(save_fig_path)
    print(f"\n分布图已保存到: {save_fig_path}")
    plt.show()


if __name__ == "__main__":
    # !!! 重要：请将这里的路径修改为你自己的柠檬树数据集根目录 !!!
    LEMON_DATASET_ROOT = r"D:\DownLoad\baiduyunDownload\9lemon\00lemon"

    analyze_dataset_branch_distribution(LEMON_DATASET_ROOT)
