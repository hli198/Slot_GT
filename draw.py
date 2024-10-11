# 初版，绘制角点基于工字钢的点位关系，以及分析真值的误差。
import matplotlib.pyplot as plt
import numpy as np

# 测量距离
distances = {
    "O1_P1": 534,
    "P1_P2": 250,
    "O1_P1": 534,
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
    # "T7_P7": 
}

# No.122-127 外宽度真值
width_truth = {
    "P1_P2_122": 250,
    "P2_P3_123": 250,
    "P3_P4_124": 256, #待确认
    "P4_P5_125": 250,
    "P5_P6_126": 267,
    "P6_P7_127": 233,
}

# Base line O1_O3, O3(0,0)
O3 = np.array([0, 0])
O1 = np.array([distances["O1_O3"], 0])

# def find_third_point(A, B, d_AC, d_BC):
#     AB_vector = B - A
#     AB_length = np.linalg.norm(AB_vector)
#     AB_unit = AB_vector / AB_length
#     x = (d_AC**2 - d_BC**2 + AB_length**2) / (2 * AB_length)
#     y = np.sqrt(d_AC**2 - x**2)
#     return A + x * AB_unit + y * np.array([-AB_unit[1], AB_unit[0]])

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


P4 = find_point_from_distances(O3, O1, distances["O3_P4"], distances["O1_P4"])
P2 = find_point_from_distances(O3, O1, distances["O3_P2"], distances["O1_P2"])
P1_negative = find_point_from_distances(O1, P2, distances["O1_P1"], distances["P1_P2"], return_negative=True)
P3 = find_point_from_distances(O3, O1, distances["O3_P3"], distances["O1_P3"])
P5 = find_point_from_distances(O3, P4, distances["O3_P5"], distances["P4_P5"])
O5 = find_point_from_distances(O3, P5, distances["O3_O5"], distances["O5_P5"])
P6_neggative = find_point_from_distances(O3, O5, distances["O3_P6"], distances["O5_P6"], return_negative=True)
P7 = find_point_from_distances(O5, P6_neggative, distances["O5_P7"], distances["P6_P7"])
T3_negative = find_point_from_distances(O3, O1, distances["T3_O3"], distances["T3_O1"], return_negative=True)
T5 = find_point_from_distances(O3, O5, distances["T5_O3"], distances["T5_O5"])
# T7 = find_point_from_distances(P7, O5, distances["T7_P7"], distances["T7_O5"])

# print
print("O3:", O3)
print("O1:", O1)
print("P3:", P3)
print("P4:", P4)
print("P2:", P2)
print("P5:", P5)
print("O5:", O5)
print("P6:", P6_neggative)
print("P7:", P7)
print("P1:", P1_negative)
print("T3:", T3_negative)
print("T5:", T5)
# print("T7:", T7)

# ======verify======
# 计算 slot外侧两点间距离，对比推算值与测量值
distance_P6_P5 = np.linalg.norm(P6_neggative - P5)
print("P5_P6_126:", distance_P6_P5)
distance_P2_P3 = np.linalg.norm(P3 - P2)
print("P2_P3_123:", distance_P2_P3)
distance_P4_P3 = np.linalg.norm(P3 - P4)
print("P3_P4_124:", distance_P4_P3)


# ========拟合P点，分析======
points = np.array([P1_negative, P2, P3, P4, P5, P6_neggative, P7])
x_coords = points[:, 0]
y_coords = points[:, 1]

# 使用最小二乘法拟合线性方程 y = mx + b
m, b = np.polyfit(x_coords, y_coords, 1)
fit_line = m * x_coords + b
distances_to_fit = np.abs(y_coords - fit_line) / np.sqrt(1 + m**2)

print(f"Fitting equation: y = {m:.2f}x + {b:.2f}")
print("Vertical distance to fitted line:", distances_to_fit)

# plot fitted line and P
plt.figure(figsize=(10, 8))
plt.plot(x_coords, y_coords, 'ro-', label='P')
plt.plot(x_coords, fit_line, 'b-', label='fitted line')
plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.title('Least squares fitting')
plt.grid(True)
plt.axis('equal')
plt.gca().invert_yaxis()  # Y轴翻转
plt.legend()
plt.show(block=False)


# =====plot point diagram========
plt.figure(figsize=(10, 8))
plt.plot([O3[0], O1[0]], [O3[1], O1[1]], 'ro-', label='O3-O1')
plt.plot([O3[0], P4[0]], [O3[1], P4[1]], 'go-', label='O3-P4')
plt.plot([O1[0], P4[0]], [O1[1], P4[1]], 'bo-', label='O1-P4')
plt.plot([O3[0], P2[0]], [O3[1], P2[1]], 'mo-', label='O3-P2')
plt.plot([O1[0], P2[0]], [O1[1], P2[1]], 'co-', label='O1-P2')
plt.plot([O1[0], P1_negative[0]], [O1[1], P1_negative[1]], 'co-', label='O1-P1')
plt.plot([P1_negative[0], P2[0]], [P1_negative[1], P2[1]], 'co-', label='P1-P2')
plt.plot([O3[0], P3[0]], [O3[1], P3[1]], 'yo-', label='O3-P3')
plt.plot([O1[0], P3[0]], [O1[1], P3[1]], 'ko-', label='O1-P3')
plt.plot([O3[0], P5[0]], [O3[1], P5[1]], 'ro--', label='O3-P5')
plt.plot([P4[0], P5[0]], [P4[1], P5[1]], 'go--', label='P4-P5')
plt.plot([O3[0], O5[0]], [O3[1], O5[1]], 'bo--', label='O3-O5')
plt.plot([P5[0], O5[0]], [P5[1], O5[1]], 'co--', label='P5-O5')
plt.plot([O3[0], P6_neggative[0]], [O3[1], P6_neggative[1]], 'mo--', label='O3-P6')
plt.plot([O5[0], P6_neggative[0]], [O5[1], P6_neggative[1]], 'yo--', label='O5-P6')
plt.plot([P7[0], O5[0]], [P7[1], O5[1]], 'bo--', label='P7-O5')
plt.plot([P7[0], P6_neggative[0]], [P7[1], P6_neggative[1]], 'co--', label='P7-P6')
plt.plot([T5[0], O5[0]], [T5[1], O5[1]], 'bo--', label='T5-O5')
plt.plot([T5[0], O3[0]], [T5[1], O3[1]], 'bo--', label='T5-O3')
plt.plot([T3_negative[0], O3[0]], [T3_negative[1], O3[1]], 'bo-', label='T3-O3')
plt.plot([T3_negative[0], O1[0]], [T3_negative[1], O1[1]], 'bo-', label='T3-O1')
# plt.plot([T7[0], P7[0]], [T7[1], P7[1]], 'go--', label='T7-P7')
# plt.plot([T7[0], O5[0]], [T7[1], O5[1]], 'go--', label='T7-O5')


#to do: 添加T7点，需要测量T7到P7的距离。


for point, label in zip([O3, O1, P3, P4, P2, P5, O5, P6_neggative, P7, P1_negative, T5, T3_negative], ['O3', 'O1', 'P3', 'P4', 'P2', 'P5', 'O5', 'P6', 'P7', 'P1','T5','T3']):
    plt.text(point[0], point[1], label, fontsize=12, ha='right')

plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.title('Point diagram based I-beam')
plt.grid(True)
plt.axis('equal')
plt.gca().invert_yaxis()  # Y轴翻转
plt.legend()
plt.show(block=False)

plt.show() 
