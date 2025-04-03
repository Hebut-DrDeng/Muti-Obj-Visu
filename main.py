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
        """计算3D帕累托前沿"""
        points = np.array(points)
        if len(points) < 3:
            return points

        # 移除无效点
        valid_points = points[~np.isnan(points).any(axis=1)]
        print("有效数据：", valid_points)
        if len(valid_points) < 3:
            return valid_points

        # 标准化处理
        mins = valid_points.min(axis=0)
        maxs = valid_points.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1  # 避免除以零
        normalized = (valid_points - mins) / ranges

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
        """更新3D绘图"""
        self.ax.clear()

        # 获取当前代数据
        gen_data = np.array(self.generations[self.current_gen])
        valid_mask = ~np.isnan(gen_data).any(axis=1)
        gen_data = gen_data[valid_mask]

        if len(gen_data) == 0:
            self.ax.text(0.5, 0.5, 0.5, "无有效数据", ha='center')
            plt.draw()
            return

        val_dice = gen_data[:, 0]
        params = gen_data[:, 1]
        robustness = gen_data[:, 2]

        # 计算帕累托前沿
        pareto_points = self.calculate_pareto_front(gen_data)

        # 绘制普通点
        self.ax.scatter(
            params, val_dice, robustness,
            c=[self.gen_colors[self.current_gen]] * len(val_dice),
            s=60, alpha=0.6, label='普通架构'
        )

        # 检查有效的帕累托前沿点
        if len(pareto_points) > 1:
            # 按参数排序以便连线
            order = np.lexsort((pareto_points[:, 2], pareto_points[:, 1]))

            # 生成网格数据
            grid_x, grid_y = np.meshgrid(np.linspace(pareto_points[:, 1].min(), pareto_points[:, 1].max(), 50),
                                         np.linspace(pareto_points[:, 0].min(), pareto_points[:, 0].max(), 50))

            # 使用griddata进行插值
            grid_z = griddata((pareto_points[:, 1], pareto_points[:, 0]), pareto_points[:, 2], (grid_x, grid_y),
                              method='linear')

            # 检查是否有NaN值
            if np.isnan(grid_z).any():
                print("插值结果中包含NaN值，可能需要调整插值参数或数据范围")
                # 将NaN值替换为0
                grid_z = np.nan_to_num(grid_z)

            # 绘制曲面
            surface = self.ax.plot_surface(grid_x, grid_y, grid_z, color='gray', alpha=0.5, rstride=100, cstride=100)

            # 突出前沿点
            self.ax.scatter(
                pareto_points[:, 1], pareto_points[:, 0], pareto_points[:, 2],
                c='gold', s=120, edgecolors='black',
                marker='*', label='最优架构'
            )

        # 设置坐标轴范围
        self.ax.set_xlim([0, 2])  # 参数量 (MB) ↓
        self.ax.set_ylim([0.8, 1])  # Dice系数 ↑
        self.ax.set_zlim([0, 0.3])  # 鲁棒性 ↑

        # 设置坐标轴
        self.ax.set_xlabel('参数量 (MB) ↓', fontsize=12)
        self.ax.set_ylabel('Dice系数 ↑', fontsize=12)
        self.ax.set_zlabel('鲁棒性 ↑', fontsize=12)
        self.ax.set_title(
            f'第 {self.current_gen + 1} 代架构评估 (共{len(self.generations)}代)',
            fontsize=14, pad=20
        )

        # 添加统计信息
        stats_text = (
            f'平均值:\n'
            f'Dice: {val_dice.mean():.3f}\n'
            f'参数: {params.mean():.3f} MB\n'
            f'鲁棒: {robustness.mean():.3f}\n'
            f'有效点: {len(gen_data)}/30'
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
    base_dir = "C:/Users/11659/Desktop/search-GA-BiObj-micro-20250330-205519"  # 替换为实际路径

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