import numpy as np
import math
'''同学们，下面是计算固体力学编程作业：
编写求解三维桁架或三维刚架静位移、应力以及固有模态的程序。程序语言不限，用A4纸正反面打印，12月14号交。
程序模块包括：
1结点，单元，材料性质和外载荷输入模块；
2 单元矩阵计算和组装模块；
3 考虑位移边界条件，求解静力问题，并画出结构变形图模块；
4 考虑位移边界条件，求解广义本征值方程，并画出前5阶模态图模块。@所有人 '''
# 1结点，单元，材料性质和外载荷输入模块；
def add_node(node_list, list):#添加一个节点
    node_list.append(list)
    return node_list

def connect_node(connect_node ,list):#连接两个节点，返回两个节点的坐标和材料性质
    connect_node.append(list)
    return connect_node

def add_point(dist_list,list):#添加一个节点间的节点
    dist_list.append(list)
    return dist_list