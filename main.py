import os
import re
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import RadioButtons
from scipy.spatial import ConvexHull
from itertools import cycle
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# 设置绘图样式
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


class NASLogParser:
    def __init__(self):
        self.generations = []
        self.arch_count = 0
        self.current_gen = []

    def parse_directory(self, base_dir):
        """解析目录中的所有arch_x子文件夹"""
        try:
            arch_folders = sorted(
                [d for d in os.listdir(base_dir) if d.startswith('arch_')],
                key=lambda x: int(x.split('_')[1]))

            for folder in arch_folders:
                log_path = os.path.join(base_dir, folder, '')
                metrics = self._parse_log_file(log_path)
                self._add_to_generation(metrics)

            # 处理最后一组不足30个的情况
            if self.current_gen and len(self.current_gen) < 30:
                self.current_gen += [(0, 0, 0)] * (30 - len(self.current_gen))
                self.generations.append(self.current_gen)

            return self.generations
        except Exception as e:
            print(f"解析目录时出错: {e}")
            return []

    def _parse_log_file(self, folder_path):
        """解析单个日志文件"""
        try:
            # 假设日志文件命名格式为 log_arch_X.txt
            log_file = os.path.join(folder_path, f'log_arch_{self.arch_count+1}.txt')

            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f.readlines()[::-1]:  # 从后往前查找最新结果
                    if 'with result' in line:
                        # 修改正则表达式，处理中文冒号和英文冒号
                        match = re.search(
                            r'val_dice:([\d.]+)[，,]params ([\d.]+)[，,]robustness:([\d.]+)',
                            line
                        )
                        if match:
                            return (
                                float(match.group(1)),
                                float(match.group(2)),
                                float(match.group(3))
                            )
            return (0, 0, 0)
        except Exception as e:
            print(f"解析日志文件 {log_file} 时出错: {e}")
            return (0, 0, 0)

    def _add_to_generation(self, metrics):
        """将架构添加到当前代"""
        self.current_gen.append(metrics)
        self.arch_count += 1

        # 每30个架构为一组
        if self.arch_count % 30 == 0:
            self.generations.append(self.current_gen)
            self.current_gen = []


class NASVisualizer:
    def __init__(self, generations):
        self.generations = generations
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 设置颜色和标记
        self.colors = cycle(plt.cm.tab20.colors)
        self.gen_colors = {i: next(self.colors) for i in range(len(generations))}

        # 创建控件
        self._create_controls()

        # 初始化显示
        self.current_gen = 0
        self.update_plot()

    def _create_controls(self):
        """创建交互控件"""
        # 调整主图位置
        plt.subplots_adjust(left=0.2)

        # 代选择器
        rax = plt.axes([0.05, 0.3, 0.1, 0.4])
        self.radio = RadioButtons(
            rax,
            [f'第 {i + 1} 代' for i in range(len(self.generations))],
            active=0
        )
        self.radio.on_clicked(self.on_gen_change)

        # 添加说明文本
        self.fig.text(
            0.05, 0.15,
            '操作说明:\n'
            '1. 点击选择不同代\n'
            '2. 鼠标拖动旋转视图\n'
            '3. 滚轮缩放视图\n'
            '4. 右键拖动平移视图',
            fontsize=10
        )

    def on_gen_change(self, label):
        """处理代选择事件"""
        self.current_gen = int(label.split()[1]) - 1
        self.update_plot()

    def calculate_pareto_front(self, points):
        """计算3D帕累托前沿，调整优化方向"""
        points = np.array(points)
        if len(points) < 3:
            return points

        # 移除无效点
        valid_points = points[~np.isnan(points).any(axis=1)]
        if len(valid_points) < 3:
            return valid_points

        # 调整优化方向
        # - Dice 系数：越大越好（保持不变）
        # - 参数量：越小越好（取反）
        # - 鲁棒性：越小越好（取反）
        adjusted_points = valid_points.copy()
        adjusted_points[:, 1] = -adjusted_points[:, 1]  # 参数量取反
        adjusted_points[:, 2] = -adjusted_points[:, 2]  # 鲁棒性取反
        # Dice 系数（[:, 0]）保持不变

        # 标准化处理
        mins = adjusted_points.min(axis=0)
        maxs = adjusted_points.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1  # 避免除以零
        normalized = (adjusted_points - mins) / ranges

        print("标准化后的数据：", normalized)

        # 计算凸包
        try:
            hull = ConvexHull(normalized)

            # 提取帕累托前沿点
            pareto_indices = set()
            for simplex in hull.simplices:
                for idx in simplex:
                    pareto_indices.add(idx)

            return valid_points[list(pareto_indices)]
        except Exception as e:
            print(f"计算帕累托前沿时出错: {e}")
            return valid_points

    def update_plot(self):
        """更新3D绘图，修正墙角平面投影方向为最差方向"""
        self.ax.clear()

        # 获取当前代数据
        gen_data = np.array(self.generations[self.current_gen])
        valid_mask = ~np.isnan(gen_data).any(axis=1)
        gen_data = gen_data[valid_mask]

        if len(gen_data) == 0:
            self.ax.text(0.5, 0.5, 0.5, "无有效数据", ha='center')
            plt.draw()
            return

        # 计算帕累托前沿
        pareto_points = self.calculate_pareto_front(gen_data)

        # 检查是否有有效的帕累托前沿点
        if len(pareto_points) == 0:
            self.ax.text(0.5, 0.5, 0.5, "无帕累托前沿点", ha='center')
            plt.draw()
            return

        # 绘制帕累托前沿点
        self.ax.scatter(
            pareto_points[:, 1], pareto_points[:, 0], pareto_points[:, 2],
            c='gold', s=120, edgecolors='black',
            marker='*', label='最优架构'
        )

        # 减少平面数量：只为部分点绘制平面（例如每隔一个点）
        step = max(1, len(pareto_points)//len(pareto_points))  # 控制绘制平面的点数量，最多绘制5个点
        selected_points = pareto_points[::step]

        # 绘制每个选定帕累托点的“墙角”平面
        for point in selected_points:
            x, y, z = point[1], point[0], point[2]  # params, val_dice, robustness

            # 定义平面的边界（向“最差”方向延伸）
            # - 参数量：越小越好，最差值是最大值 x=2.0
            # - Dice 系数：越大越好，最差值是最小值 y=0.8
            # - 鲁棒性：越小越好，最差值是最大值 z=0.3
            x_worst, y_worst, z_worst = 2.0, 0.8, 0.3  # 坐标轴的“最差”值

            # 降低网格密度以提高性能
            grid_size = 10

            # 平面 1：平行于 yz 平面 (x = x_i)，从 x_i 向 x_worst 延伸
            y_range = np.linspace(y, y_worst, grid_size)  # Dice 从 y_i 向 0.8 延伸
            z_range = np.linspace(z, z_worst, grid_size)  # 鲁棒性从 z_i 向 0.3 延伸
            Y, Z = np.meshgrid(y_range, z_range)
            X = np.full_like(Y, x)
            self.ax.plot_surface(X, Y, Z, color='#FF9999', alpha=0.05, rstride=1, cstride=1)

            # 平面 2：平行于 xz 平面 (y = y_i)，从 y_i 向 y_worst 延伸
            x_range = np.linspace(x, x_worst, grid_size)  # 参数量从 x_i 向 2.0 延伸
            z_range = np.linspace(z, z_worst, grid_size)  # 鲁棒性从 z_i 向 0.3 延伸
            X, Z = np.meshgrid(x_range, z_range)
            Y = np.full_like(X, y)
            self.ax.plot_surface(X, Y, Z, color='#99FF99', alpha=0.05, rstride=1, cstride=1)

            # 平面 3：平行于 xy 平面 (z = z_i)，从 z_i 向 z_worst 延伸
            x_range = np.linspace(x, x_worst, grid_size)  # 参数量从 x_i 向 2.0 延伸
            y_range = np.linspace(y, y_worst, grid_size)  # Dice 从 y_i 向 0.8 延伸
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.full_like(X, z)
            self.ax.plot_surface(X, Y, Z, color='#9999FF', alpha=0.05, rstride=1, cstride=1)

        # 设置坐标轴范围
        self.ax.set_xlim([0, 2])  # 参数量 (MB) ↓
        self.ax.set_ylim([0.8, 1])  # Dice系数 ↑
        self.ax.set_zlim([0, 0.3])  # 鲁棒性 ↓

        # 设置坐标轴标签，确保方向正确
        self.ax.set_xlabel('参数量 (MB) ↓', fontsize=12)
        self.ax.set_ylabel('Dice系数 ↑', fontsize=12)
        self.ax.set_zlabel('鲁棒性 ↓', fontsize=12)
        self.ax.set_title(
            f'第 {self.current_gen + 1} 代帕累托前沿 (共{len(self.generations)}代)',
            fontsize=14, pad=20
        )

        # 添加统计信息
        stats_text = (
            f'帕累托前沿点数: {len(pareto_points)}\n'
            f'绘制平面点数: {len(selected_points)}\n'
        )
        self.ax.text2D(
            0.02, 0.98, stats_text,
            transform=self.ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7),
            fontsize=10
        )

        # 添加图例
        self.ax.legend(loc='upper right')

        plt.draw()


def main():
    # 配置参数
    base_dir = "D:/PycharmProjects/Muti-Obj-Visu/search-GA-BiObj-micro-20250330-205519"  # 替换为实际路径

    # 解析日志
    print("正在解析日志文件...")
    parser = NASLogParser()
    generations = parser.parse_directory(base_dir)

    if not generations:
        print("未找到有效数据，请检查路径和日志格式")
        return

    print(f"成功解析 {len(generations)} 代数据")

    # 可视化
    print("正在创建可视化界面...")
    visualizer = NASVisualizer(generations)
    plt.show()


if __name__ == "__main__":
    main()