import matplotlib.pyplot as plt
import numpy as np

# 测量距离
distances = {
    "O1_P1": 534,
    "P1_P2": 250,
    "O3_P4": 473,
    "O1_P4": 697,
    "O3_P2": 686,
    "O1_P2": 475,
    "O3_P3": 535,
    "O1_P3": 540,
    "O3_P5": 538,
    "O1_O3": 501,
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

# 计算所有点的函数 (O3, O1)
def calculate_points_from_O3_O1(O3, O1):
    P4 = find_point_from_distances(O3, O1, distances["O3_P4"], distances["O1_P4"])
    P2 = find_point_from_distances(O3, O1, distances["O3_P2"], distances["O1_P2"])
    P1 = find_point_from_distances(O1, P2, distances["O1_P1"], distances["P1_P2"], return_negative=True)
    P3 = find_point_from_distances(O3, O1, distances["O3_P3"], distances["O1_P3"])
    P5 = find_point_from_distances(O3, P4, distances["O3_P5"], distances["P4_P5"])
    O5 = find_point_from_distances(O3, P5, distances["O3_O5"], distances["O5_P5"])
    P6 = find_point_from_distances(O3, O5, distances["O3_P6"], distances["O5_P6"], return_negative=True)
    P7 = find_point_from_distances(O5, P6, distances["O5_P7"], distances["P6_P7"])
    T3 = find_point_from_distances(O3, O1, distances["T3_O3"], distances["T3_O1"], return_negative=True)
    T5 = find_point_from_distances(O3, O5, distances["T5_O3"], distances["T5_O5"])
    
    return {
        "O3": O3,
        "O1": O1,
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "P4": P4,
        "P5": P5,
        "O5": O5,
        "P6": P6,
        "P7": P7,
        "T3": T3,
        "T5": T5,
    }

# 计算所有点的函数 (O3, O5)
def calculate_points_from_O3_O5(O3, O5):
    P5 = find_point_from_distances(O3, O5, distances["O3_P5"], distances["O5_P5"], return_negative=True)
    P6 = find_point_from_distances(O3, O5, distances["O3_P6"], distances["O5_P6"], return_negative=True)
    P7 = find_point_from_distances(O5, P6, distances["O5_P7"], distances["P6_P7"])
    P4 = find_point_from_distances(O3, P5, distances["O3_P4"], distances["P4_P5"], return_negative=True)
    O1 = find_point_from_distances(O3, P4, distances["O1_O3"], distances["O1_P4"], return_negative=True)
    P3 = find_point_from_distances(O3, O1, distances["O3_P3"], distances["O1_P3"])
    P2 = find_point_from_distances(O3, O1, distances["O3_P2"], distances["O1_P2"])
    P1 = find_point_from_distances(O1, P2, distances["O1_P1"], distances["P1_P2"], return_negative=True)
    T3 = find_point_from_distances(O3, O1, distances["T3_O3"], distances["T3_O1"], return_negative=True)
    T5 = find_point_from_distances(O3, O5, distances["T5_O3"], distances["T5_O5"])
    
    return {
        "O3": O3,
        "O1": O1,
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "P4": P4,
        "P5": P5,
        "O5": O5,
        "P6": P6,
        "P7": P7,
        "T3": T3,
        "T5": T5,
    }

# 判断输入是 O1 还是 O5，并计算所有点
def calculate_points(O3, O1=None, O5=None):
    if O1 is not None:
        return calculate_points_from_O3_O1(O3, O1)
    elif O5 is not None:
        return calculate_points_from_O3_O5(O3, O5)
    else:
        raise ValueError("需要输入O1或O5的坐标")

# 设置两点的坐标（可以修改为任意值）,但要保证两点的距离准确
O3 = np.array([378.83857701708274, 488.9395625553959])  
# O1 = np.array([502, 0]) 
O5 = np.array([39, 98])  

# points = calculate_points(O3, O1=O1)  
points = calculate_points(O3, O5=O5)  

# print all points
for point_name, point_coords in points.items():
    print(f"{point_name}: {point_coords}")


# 绘制点函数
def plot_points(points):
    fig, ax = plt.subplots()
    
    # 绘制点及标注
    for point_name, point_coords in points.items():
        ax.plot(point_coords[0], point_coords[1], 'o')  # 绘制点
        ax.text(point_coords[0] + 10, point_coords[1] + 10, point_name)  # 标注点名称
    
    ax.set_aspect('equal')  # 保持x和y轴比例一致
    plt.grid(True)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Points Plot')
    plt.show()

# 调用绘图函数
plot_points(points)