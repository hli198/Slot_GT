# 效果很差的一个版本
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 定义已知的点和距离关系
distances = {
    "O1_P1": 534,
    "P1_P2": 250,
    "O1_O3": 501,
    "O3_P4": 473,
    "O1_P4": 697,
    "O3_P2": 686,
    "O1_P2": 475,
    "O3_P3": 535,
    "O1_P3": 540,
    "O3_P5": 538,
    "P4_P5": 250,
    "O3_O5": 518,
    "O5_P5": 545,
    "O3_P6": 705,
    "O5_P6": 476,
    "O5_P7": 530,
    "P6_P7": 233,
    "T5_O3": 256,
    "T5_O5": 273,
    "T3_O3": 254,
    "T3_O1": 259,
    "T7_O5": 242,
}

# 更新点的索引，确保所有点都在其中
points = {
    "O1": 0,
    "O3": 1,
    "O5": 2,
    "P1": 3,
    "P2": 4,
    "P3": 5,
    "P4": 6,
    "P5": 7,
    "P6": 8,
    "P7": 9,
    "T3": 10,
    "T5": 11,
    "T7": 12,
}

# 定义点的初始位置
initial_positions = np.zeros((len(points), 2))

# 定义计算距离的函数
def calculate_distance(pos, p1, p2):
    return np.linalg.norm(pos[p1] - pos[p2])

# 定义目标函数
def objective_function(pos):
    total_error = 0.0
    for (key, true_distance) in distances.items():
        p1, p2 = key.split('_')
        p1_index = points[p1]
        p2_index = points[p2]
        calculated_distance = calculate_distance(pos, p1_index, p2_index)
        total_error += (calculated_distance - true_distance) ** 2
    return total_error

# 最小化目标函数
result = minimize(objective_function, initial_positions.flatten())

# 获取最终位置
final_positions = result.x.reshape((len(points), 2))

# 输出结果
for point, position in zip(points.keys(), final_positions):
    print(f"{point}: (x={position[0]:.2f}, y={position[1]:.2f})")

# 可视化结果
plt.figure(figsize=(10, 8))

# 绘制点
for point, position in zip(points.keys(), final_positions):
    plt.scatter(position[0], position[1], label=point)

# 绘制连线
for (key, true_distance) in distances.items():
    p1, p2 = key.split('_')
    p1_index = points[p1]
    p2_index = points[p2]
    plt.plot([final_positions[p1_index][0], final_positions[p2_index][0]],
             [final_positions[p1_index][1], final_positions[p2_index][1]], 'k--', alpha=0.5)

plt.title("Point Positions and Distances")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid()
plt.axis('equal')  # 使x和y轴比例相同
plt.show()
