import numpy as np
import numpy.linalg as la
import math
'''
numpy.linalg.det(a)： 计算矩阵的行列式。
numpy.linalg.inv(a)： 计算矩阵的逆矩阵。
numpy.linalg.eig(a)： 计算方阵的特征值和特征向量。
numpy.linalg.solve(a, b)： 解线性方程组 Ax = b，其中 A 是系数矩阵，b 是右侧的常数向量。
numpy.linalg.norm(x, ord=None)： 计算向量或矩阵的范数。
numpy.linalg.svd(a, full_matrices=True)： 计算奇异值分解。
'''
def get_dist(node_list,idx1,idx2):
    return la.norm(np.array(node_list[idx1])-np.array(node_list[idx2]))
#def get_angle(node_list,idx1,idx2):

def get_angle_x(node_list,idx1,idx2):
    vector = np.array(node_list[idx1])-np.array(node_list[idx2])
    
    # 计算向量的模
    magnitude = np.linalg.norm(vector)
    
    # 计算夹角（弧度）
    angle_rad = np.arccos(vector[0] / magnitude)
    
    # 将弧度转换为度
    return angle_rad
def get_angle_y(node_list,idx1,idx2):
    vector = np.array(node_list[idx1])-np.array(node_list[idx2])
    
    # 计算向量的模
    magnitude = np.linalg.norm(vector)
    
    # 计算夹角（弧度）
    angle_rad = np.arccos(vector[1] / magnitude)
    
    # 将弧度转换为度
    return angle_rad
def get_angle_z(node_list,idx1,idx2):
    vector = np.array(node_list[idx1])-np.array(node_list[idx2])
    
    # 计算向量的模
    magnitude = np.linalg.norm(vector)
    
    # 计算夹角（弧度）
    angle_rad = np.arccos(vector[2] / magnitude)
    
    # 将弧度转换为度
    return angle_rad