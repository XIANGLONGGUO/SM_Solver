from node import *
import numpy
import matplotlib.pyplot as plt
from node import *
def add_node_utils(node_list):
    I=0
    print('结束输入：end')
    while True:
        x = input("请输入节点的x y z坐标,或者退出：")
        if x=="end":
            break
        if len(x.split(" "))!=3:
            print("输入错误")
            raise ValueError
        if x.split(' ') in node_list:
            print("节点已存在")
            raise ValueError
        node_list=add_node(node_list,x.split(" "))
        print("节点添加成功")
        print('node_list')
        for i in range(len(node_list)):
            print(i)
            print(node_list[i])

    return node_list

def remove_node_utils(node_list):
    I=0
    print('结束输入输入：end')
    while True:
        x = input("请输入节点的index：")
        print('node_list')
        for i in range(len(node_list)):
            print(i)
            print(node_list[i])
        node_list.pop(int(x))
        print('node_list')
        for i in range(len(node_list)):
            print(i)
            print(node_list[i])
        if x.pop()=="end":
            break
    return node_list

def add_connect_utils(node_list,connect_list):
    I=0
    print('结束输入：end')
    while True:
        print('没有此力请输入0')
        x = input("请输入节点的index index E I 前端点是否约束(0否，1简支，2固定) 后端点是否约束(0否，1简支，2固定) 左端点Fx Fy Fz Mx My Mz 右端点Fx Fy Fz Mx My Mz qx qy qz：")
        if x=="end":
            break
        if int(x.split(' ')[0])>len(node_list) or int(x.split(' ')[1])>len(node_list):
            print("节点不存在")
            raise ValueError
        print('node_list')
        for i in range(len(node_list)):
            print(i)
            print(node_list[i])
        connect_list=connect_node(connect_list,x.split(" "))
        print('connect_list')
        for i in range(len(connect_list)):
            print(i)
            print(connect_list[i])

    return connect_list

def add_point_utils(node_list,connect_list,dist_list):
    I=0
    print('结束输入：end')
    while True:
        x = input("请输入节点的连接序号 起始点距离占比 是否固定(0否，1简支，2固定) 端点Fx Fy Fz Mx My Mz：")
        if x=="end":
            print('node_list')
            for i in range(len(node_list)):
                print(i)
                print(node_list[i])
            print('connect_list')
            
            for i in range(len(connect_list)):
                print(i)
                print(connect_list[i])
            print('dist_list')
            for i in range(len(dist_list)):
                print(i)
                print(dist_list[i])
            break
        
        if int(x.split(' ')[0])>len(connect_list):
            print("连接不存在")
            raise ValueError
        print('node_list')
        for i in range(len(node_list)):
            print(i)
            print(node_list[i])
        print('connect_list')
        for i in range(len(connect_list)):
            print(i)
            print(connect_list[i])
        print('dist_list')
        dist_list=add_point(dist_list,x.split(" "))

        for i in range(len(dist_list)):
            print(i)
            print(dist_list[i])

    return dist_list

def remove_connect_utils(node_list,connect_list):
    I=0
    print('结束输入：end')
    while True:
        x = input("请输入节点的连接序号：")
        for i in range(len(node_list)):
            print(i)
            print(node_list[i])
        
        for i in range(len(connect_list)):
            print(i)
            print(connect_list[i])
        connect_list.pop(int(x))
        if x=="end":
            for i in range(len(node_list)):
                print(i)
            print(node_list[i])
            
            for i in range(len(connect_list)):
                print(i)
                print(connect_list[i])
            break
    return connect_list

def remove_point_utils(node_list,connect_list,dist_list):
    I=0
    print('结束输入：end')
    while True:
        x = input("index")
        for i in range(len(dist_list)):
            print(i)
            print(dist_list[i])
        
        dist_list.pop(int(x))
        if x=="end":
            for i in range(len(node_list)):
                print(i)
            print(node_list[i])
            
            for i in range(len(connect_list)):
                print(i)
                print(connect_list[i])
            break
    return dist_list
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def plot(node_list,connect_list):
    # 你提供的三维点的列表
    points =node_list
    # 给定两个点在列表中的索引
    connections = []# 两个点的索引连接
    for i in range(len(connect_list)):
        connections.append([int(connect_list[i][0]),int(connect_list[i][1])])
    print(points)
    print(connections)
    # 提取点的坐标
    x = [float(point[0]) for point in points]
    y = [float(point[1]) for point in points]
    z = [float(point[2]) for point in points]

    # 画出点
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')  # 画出红色的点

    # 画出连线
    for connection in connections:
        point1_index, point2_index = connection
        ax.plot([x[point1_index], x[point2_index]],
                [y[point1_index], y[point2_index]],
                [z[point1_index], z[point2_index]], c='b')

    # 设置图表标题和坐标轴标签
    ax.set_title('3D Scatter Plot with Line')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # 显示图表
    plt.show()




