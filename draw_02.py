# 实现输入O3和O1坐标，求出所有坐标出来。
# 接下来要做的是输入O3和O5坐标，求出所有坐标来
import matplotlib.pyplot as plt
import numpy as np

# 测量距离
distances = {
    "O1_P1": 534,
    "P1_P2": 250,
    # "O1_O3": 501,
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

# # No.122-127 外宽度真值
# width_truth = {
#     "P1_P2_122": 250,
#     "P2_P3_123": 250,
#     "P3_P4_124": 256, #待确认
#     "P4_P5_125": 250,
#     "P5_P6_126": 267,
#     "P6_P7_127": 233,
# }


# 由已知两点，以及两边距离，计算第三点的坐标
def find_point_from_distances(ref_point, second_point, d_ref, d_second, return_negative=False):
    direction = (second_point - ref_point) / np.linalg.norm(second_point - ref_point)
    x = (d_ref**2 - d_second**2 + np.linalg.norm(second_point - ref_point)**2) / (2 * np.linalg.norm(second_point - ref_point))
    
    # 计算 y 的两个可能值
    y_positive = np.sqrt(d_ref**2 - x**2)
    y_negative = -y_positive

    # 默认一个结果
    if return_negative:
        return ref_point + x * direction + y_negative * np.array([-direction[1], direction[0]])
    else:
        return ref_point + x * direction + y_positive * np.array([-direction[1], direction[0]])

# 计算所有点的函数
def calculate_points(O3, O1):
    O1_O3_distance = np.linalg.norm(O1 - O3)  # 计算 O1 和 O3 的距离
    P4 = find_point_from_distances(O3, O1, distances["O3_P4"], distances["O1_P4"])
    P2 = find_point_from_distances(O3, O1, distances["O3_P2"], distances["O1_P2"])
    P1_negative = find_point_from_distances(O1, P2, distances["O1_P1"], distances["P1_P2"], return_negative=True)
    P3 = find_point_from_distances(O3, O1, distances["O3_P3"], distances["O1_P3"])
    P5 = find_point_from_distances(O3, P4, distances["O3_P5"], distances["P4_P5"])
    O5 = find_point_from_distances(O3, P5, distances["O3_O5"], distances["O5_P5"])
    P6_negative = find_point_from_distances(O3, O5, distances["O3_P6"], distances["O5_P6"], return_negative=True)
    P7 = find_point_from_distances(O5, P6_negative, distances["O5_P7"], distances["P6_P7"])
    T3_negative = find_point_from_distances(O3, O1, distances["T3_O3"], distances["T3_O1"], return_negative=True)
    T5 = find_point_from_distances(O3, O5, distances["T5_O3"], distances["T5_O5"])
    
    return {
        "O3": O3,
        "O1": O1,
        "P1": P1_negative,
        "P2": P2,
        "P3": P3,
        "P4": P4,
        "P5": P5,
        "O5": O5,
        "P6": P6_negative,
        "P7": P7,
        "T3": T3_negative,
        "T5": T5,
    }

# 设置O3和O1的坐标（可以修改为任意值）
# O3 = np.array([102, 502])  
# O1 = np.array([457.76, 857.76])  

O3 = np.array([0, 0])  # 任意坐标, 但要保证O1_O3准备
O1 = np.array([502, 0])  

# 计算所有点的坐标
points = calculate_points(O3, O1)

# print
for point_name, coord in points.items():
    print(f"{point_name}: {coord}") 

# ====== 绘图 ======
plt.figure(figsize=(10, 8))

plt.plot([O3[0], O1[0]], [O3[1], O1[1]], 'ro-', label='O3-O1')
plt.plot([O3[0], points["P4"][0]], [O3[1], points["P4"][1]], 'go-', label='O3-P4')
plt.plot([O1[0], points["P4"][0]], [O1[1], points["P4"][1]], 'bo-', label='O1-P4')
plt.plot([O3[0], points["P2"][0]], [O3[1], points["P2"][1]], 'mo-', label='O3-P2')
plt.plot([O1[0], points["P2"][0]], [O1[1], points["P2"][1]], 'co-', label='O1-P2')
plt.plot([O1[0], points["P1"][0]], [O1[1], points["P1"][1]], 'co-', label='O1-P1')
plt.plot([points["P1"][0], points["P2"][0]], [points["P1"][1], points["P2"][1]], 'co-', label='P1-P2')
plt.plot([O3[0], points["P3"][0]], [O3[1], points["P3"][1]], 'yo-', label='O3-P3')
plt.plot([O1[0], points["P3"][0]], [O1[1], points["P3"][1]], 'ko-', label='O1-P3')
plt.plot([O3[0], points["P5"][0]], [O3[1], points["P5"][1]], 'ro--', label='O3-P5')
plt.plot([points["P4"][0], points["P5"][0]], [points["P4"][1], points["P5"][1]], 'go--', label='P4-P5')
plt.plot([O3[0], points["O5"][0]], [O3[1], points["O5"][1]], 'bo--', label='O3-O5')
plt.plot([points["P5"][0], points["O5"][0]], [points["P5"][1], points["O5"][1]], 'co--', label='P5-O5')
plt.plot([O3[0], points["P6"][0]], [O3[1], points["P6"][1]], 'mo--', label='O3-P6')
plt.plot([points["O5"][0], points["P6"][0]], [points["O5"][1], points["P6"][1]], 'yo--', label='O5-P6')
plt.plot([points["P7"][0], points["O5"][0]], [points["P7"][1], points["O5"][1]], 'bo--', label='P7-O5')
plt.plot([points["P7"][0], points["P6"][0]], [points["P7"][1], points["P6"][1]], 'co--', label='P7-P6')
plt.plot([points["T5"][0], points["O5"][0]], [points["T5"][1], points["O5"][1]], 'bo--', label='T5-O5')
plt.plot([points["T5"][0], O3[0]], [points["T5"][1], O3[1]], 'bo--', label='T5-O3')
plt.plot([points["T3"][0], O3[0]], [points["T3"][1], O3[1]], 'bo-', label='T3-O3')
plt.plot([points["T3"][0], O1[0]], [points["T3"][1], O1[1]], 'bo-', label='T3-O1')

# 
for point_name, coord in points.items():
    plt.text(coord[0], coord[1], point_name, fontsize=12, ha='right')

# 
plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.title('Point Diagram Based on I-beam')
plt.grid(True)
plt.axis('equal')
# plt.gca().invert_yaxis()  # Y轴翻转
plt.legend()
plt.show()



