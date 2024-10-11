## 演示内容（代码在后面）
### 此程序主要用于对桁架结构求解，代码中部分是废弃的
此例子模型对左下角点加全方向约束，右下只有x方向没有约束，上方点约束为z方向，
此为振动模态图和特征频率（此处可能由于python求解广义特征值精度有一定误差）（此模型只有三个模态）
![[Pasted image 20231211152706.png]]![[Pasted image 20231211152717.png]]![[Pasted image 20231211152727.png]]![[Pasted image 20231211152741.png]]
对于此模型进行静力分析：施加y方向1N的力在上方节点：
这是位移场：
![[Pasted image 20231211153419.png]]
这是放大10000的形变图
![[Pasted image 20231211153522.png]]

## 代码
下面是代码：
主函数:主要是用于导入：此处数据已经被我写入了上面的默认测试样例
main.py:

```python
import math
import numpy as np
from utils import *
#from node import *
from calculate import *

if __name__=="__main__":
    print('Welcome to SM_Solver')
    print('author:GUOxianglong')
    node_list=[[0.0, 0.0, 0.0],[0.01, 0.0, 0.0],[0.005, 0.00866025404, 0.010]]####
    #node_list=[[0.0, 0.0, 0.0],[0.0, 0.33333, 0.0],[0.0,0.666666,0.0],[0.0,1.0,0.0]]
    connect_node=[[0,1],[1,2],[0,2]]####
    dist_list=[]
    l=[]

    Us=[[0,0,0],[0,1,0],[0,2,0],[1,1,0],[1,2,0],[2,2,0]]
    Metal=[70.0, 2750.0, 7.854e-07, 0.0, 0.0, 0.0, 0.0]
    '''connect_node=[[0,1],[1,2],[2,3]]####
    dist_list=[]
    l=[]

    Us=[[0,0,0],[0,1,0],[0,2,0],[3,0,0],[3,1,0],[3,2,0],[1,0,0],[1,2,0],[2,0,0],[2,2,0]]
    Metal=[1, 1e9, 1  , 0.0, 0.0, 0.0, 0.0]'''
    Dict_disp={}
    Lam = {}
    Ke = {}
    Me = {}
    mode=0
    R = []
    while True:
        node_list=convert(node_list)
        connect_node=convert1(connect_node)
        #dist_list=convert(dist_list)
        print('输入calculate来计算')
        print('输入add_node来添加节点')
        print('输入add_connect来添加连接')
        print('输入remove_node来删除节点')
        print('输入add_point来添加节点间的节点')
        print('输入remove_connect来删除连接')
        print('输入remove_point来删除节点间的节点')
        print('输入plot来绘图')
        print('输入mat来输入材料参数')
        print('输入force来添加载荷')
        print('输入c来计算')
        print('输入hz来计算频率')
        print('输入l来输入节点约束')
        print('exit退出')
        #######################################################################
        #Edit the shape of structure

        imput=input("输入操作：")
        if imput=="exit":
            break
        elif imput=="add_node" or imput=="an":
            print("开始添加节点")
            node_list=add_node_utils(node_list)
            for i in range(len(node_list)):
                print(i)
                print(node_list[i])
        elif imput=="remove_node" or imput=="rn":
            print("开始删除节点")
            node_list=remove_node_utils(node_list)
        elif imput=="add_connect" or imput=="ac":
            print("开始添加连接")
            #print('相同节点添加一次力即可')
            connect_node=add_connect_utils(node_list,connect_node)
        elif imput=="add_point" or imput=="ap":
            print("开始添加节点间的节点")
            print('不推荐使用这个功能，推荐多次建立节点，然后连接')
            dist_list=add_point_utils(node_list,connect_node,dist_list)
        elif imput=="remove_connect" or imput=="rc":
            print("开始删除连接")
            connect_node=remove_connect_utils(node_list,connect_node)
        elif imput=="remove_point" or imput=="rp":
            print("开始删除节点间的节点")
            dist_list=remove_point_utils(node_list,connect_node,dist_list)
        elif imput=="plot" or imput=="p":
            print("开始绘图")
            plot(node_list,connect_node)
        #######################################################################
        #create displcements and angle displacements
        elif imput=="mat" or imput=="m":
            mode = int(input('请选择求解桁架还是刚架，桁架请输入0，刚架请输入1\n'))
            Metal = input('请输入材料杨氏模量E，密度p，杆或梁的横截面积A，惯性矩Iz（杆输入0）,惯性矩Iy（杆输入0）,极惯性矩Ip（杆输入0）,剪切模量G（杆输入0）\n格式为E(GPa)，p(kg/m^3)，A(m^2)，Iz(m^4)，Iy(m^4)，Ip(m^4),G(GPa)\n').split(',')
            Metal = list(map(float,Metal))
            R = []
            while True:
                if mode == 0:
                    if Metal[3] != 0 or Metal[4] != 0 or Metal[5] != 0 or Metal[6] != 0:
                        print('Iz或Iy或Ip不等于0!!!!!')
                        Metal = input('请输入材料杨氏模量E，密度p，杆或梁的横截面积A，惯性矩Iz（杆输入0）,惯性矩Iy（杆输入0）,极惯性矩Ip（杆输入0）,剪切模量G（杆输入0）\n格式为E(GPa)，p(kg/m^3)，A(m^2)，Iz(m^4)，Iy(m^4)，Ip(m^4),G(GPa)\n').split(',')
                        Metal = list(map(float,Metal))
                    else:
                        print('设置完毕，桁架模式，材料参数如下',Metal)
                        break
                elif mode == 1:
                    print('设置完毕，刚架模式，材料参数如下',Metal)
                    break
                else:
                    print('模式选择错误，请重新选择')
                    mode = int(input('请选择求解桁架还是刚架，桁架请输入0，刚架请输入1\n'))
                    Metal = input('请输入材料杨氏模量E，密度p，杆或梁的横截面积A，惯性矩Iz（杆输入0）,惯性矩Iy（杆输入0）,极惯性矩Ip（杆输入0）,剪切模量G（杆输入0）\n格式为E(GPa)，p(kg/m^3)，A(m^2)，Iz(m^4)，Iy(m^4)，Ip(m^4),G(GPa)\n').split(',')
                    Metal = list(map(float,Metal))
        elif imput=='force' or imput=='f':
            R=add_force(R,node_list,connect_node,mode)
        elif imput=='l':
            l=get_l(l)
            Us=convert(l)
        elif imput=='calculate' or imput=='c':

            E,D,A,Iz,Iy,Ix,G = (Metal[0],Metal[1],Metal[2],Metal[3],Metal[4],Metal[5],Metal[6])
            for i in range(len(connect_node)):
                node = connect_node[i]
                op = node_list[node[0]]
                ed = node_list[node[1]]
                ele_array = np.array(op)-np.array(ed)
                lenth = np.linalg.norm(ele_array)
            
                if mode == 0:
                    print('衔架')
                    Lam[i] = cal_lam(ele_array,mode)
                    k = np.matrix([[1,-1],[-1,1]],dtype=float)
                    m = np.matrix([[2,1],[1,2]],dtype=float)#一致质量矩阵
                    Ke[i] = np.mat((E*10**9*A/lenth)*k)
                    Me[i] = D*A*lenth*m/6
                elif mode == 1:
                    print('钢架')
                    print(op,ed)
                    enode = np.array(list(map(float,input('请输入任意主平面xy内点的坐标，格式为0,0,0\n').split(','))))-np.array(op)
                    Lam[i] = cal_lam(ele_array,mode,enode)
                    k = np.matrix(np.zeros((12,12),dtype=float))
                    a1,a2,a3,a4,a5,a6,a7,a8 = (E*A/lenth,12*E*Iz/lenth**3,6*E*Iz/lenth**2,12*E*Iy/lenth**3,-6*E*Iy/lenth**2,G*Ix/lenth,4*E*Iy/lenth,4*E*Iz/lenth)
                    k[0,0],k[1,1],k[5,1],k[2,2],k[4,2],k[3,3],k[4,4],k[5,5] = (a1,a2,a3,a4,a5,a6,a7,a8)
                    k[6:12,6:12] = k[0:6,0:6]
                    k[11,7],k[10,8] = (-a3,-a5)
                    k[6,0],k[7,1],k[11,1],k[8,2],k[10,2],k[9,3],k[8,4],k[10,4],k[7,5],k[11,5] = (-a1,-a2,a3,-a4,a5,-a6,-a5,-a5/3,-a3,a8/2)
                    k += k.T - np.diag(k.diagonal())
                    Ke[i] = np.matrix(k,dtype=float)
                    m = np.matrix(np.zeros((12,12),dtype=float))
                    m[0,0],m[1,1],m[5,1],m[2,2],m[4,2],m[3,3],m[4,4],m[5,5] = (140,156,22*lenth,156,-22*lenth,140*Ix/A,4*lenth**2,4*lenth**2)
                    m[6:12,6:12] = m[0:6,0:6]
                    m[11,7],m[10,8] = (-22*lenth,22*lenth)
                    m[6,0],m[7,1],m[11,1],m[8,2],m[10,2],m[9,3],m[8,4],m[10,4],m[7,5],m[11,5] = (70,54,-13*lenth,54,13*lenth,70*Ix/A,-13*lenth,-3*lenth**2,13*lenth,-3*lenth**2)
                    m += m.T - np.diag(m.diagonal())
                    Me[i] = np.matrix(D*A*lenth*m/420,dtype=float)
            u_and_fw_out = u_and_fw(mode,node_list,R,connect_node,Lam,Ke,Me)
            Uw = u_and_fw_out[0]#位移列向量
            Fw = u_and_fw_out[1]#载荷列向量
            K_an_M_out = cal_k_or_m(mode,node_list,connect_node,Lam,Ke,Me)
            Kw = K_an_M_out[0]
            Mw = K_an_M_out[1]
            
            Uw = cal(Fw,Kw,mode,node_list,Us)
            print(Uw)
            s=int(input('请输入放大倍数：'))
            #s=50000000
            draw_strain(node_list,connect_node,Uw,mode,s)
        
        elif imput=='hz':

            E,D,A,Iz,Iy,Ix,G = (Metal[0],Metal[1],Metal[2],Metal[3],Metal[4],Metal[5],Metal[6])
            for i in range(len(connect_node)):
                node = connect_node[i]
                op = node_list[node[0]]
                ed = node_list[node[1]]
                ele_array = np.array(op)-np.array(ed)
                lenth = np.linalg.norm(ele_array)
            
                if mode == 0:
                    print('衔架')
                    Lam[i] = cal_lam(ele_array,mode)
                    k = np.matrix([[1,-1],[-1,1]],dtype='float64')
                    m = np.matrix([[2,1],[1,2]],dtype='float64')#一致0
                    #质量矩阵
                    Ke[i] = np.mat((E*10**9*A/lenth)*k)
                    Me[i] = D*A*lenth*m/6

                elif mode == 1:
                    print('钢架')
                    print(op,ed)
                    enode = np.array(list(map(float,input('请输入任意主平面xy内点的坐标，格式为0,0,0\n').split(','))))-np.array(op)
                    Lam[i] = cal_lam(ele_array,mode,enode)
                    k = np.matrix(np.zeros((12,12),dtype='float64'))
                    a1,a2,a3,a4,a5,a6,a7,a8 = (E*A/lenth,12*E*Iz/lenth**3,6*E*Iz/lenth**2,12*E*Iy/lenth**3,-6*E*Iy/lenth**2,G*Ix/lenth,4*E*Iy/lenth,4*E*Iz/lenth)
                    k[0,0],k[1,1],k[5,1],k[2,2],k[4,2],k[3,3],k[4,4],k[5,5] = (a1,a2,a3,a4,a5,a6,a7,a8)
                    k[6:12,6:12] = k[0:6,0:6]
                    k[11,7],k[10,8] = (-a3,-a5)
                    k[6,0],k[7,1],k[11,1],k[8,2],k[10,2],k[9,3],k[8,4],k[10,4],k[7,5],k[11,5] = (-a1,-a2,a3,-a4,a5,-a6,-a5,-a5/3,-a3,a8/2)
                    k += k.T - np.diag(k.diagonal())
                    Ke[i] = np.matrix(k,dtype=float)
                    m = np.matrix(np.zeros((12,12),dtype='float64'))
                    m[0,0],m[1,1],m[5,1],m[2,2],m[4,2],m[3,3],m[4,4],m[5,5] = (140,156,22*lenth,156,-22*lenth,140*Ix/A,4*lenth**2,4*lenth**2)
                    m[6:12,6:12] = m[0:6,0:6]
                    m[11,7],m[10,8] = (-22*lenth,22*lenth)
                    m[6,0],m[7,1],m[11,1],m[8,2],m[10,2],m[9,3],m[8,4],m[10,4],m[7,5],m[11,5] = (70,54,-13*lenth,54,13*lenth,70*Ix/A,-13*lenth,-3*lenth**2,13*lenth,-3*lenth**2)
                    m += m.T - np.diag(m.diagonal())
                    Me[i] = np.matrix(D*A*lenth*m/420,dtype='float64')
            u_and_fw_out = u_and_fw(mode,node_list,R,connect_node,Lam,Ke,Me)
            Uw = u_and_fw_out[0]#位移列向量
            Fw = u_and_fw_out[1]#载荷列向量
            K_an_M_out = cal_k_or_m(mode,node_list,connect_node,Lam,Ke,Me)
            Kw = K_an_M_out[0]
            Mw = K_an_M_out[1]
            
            Uss=[None]*len(node_list)
            for i in range(len(node_list)):
                Uss[i]=[0,0,0]
            for j in Us:
                i=int(j[0])
                if j[1]==0:
                    Uss[i][0]=1
                elif j[1]==1:
                    Uss[i][1]=1
                elif j[1]==2:
                    Uss[i][2]=1
            #import numpy as np
            import scipy.linalg as sl
            def delete(K, M, NP=len(node_list), NRR=Uss, NF=3):
                DK = K.copy()
                DM = M.copy()
                count = []

                for i in range(NP):
                    for j in range(NF):
                        print(i,j)
                        if NRR[i][j] == 1:
                            count.append(i * NF + j)

                # 将 count 转换为 NumPy 数组以便于索引
                count = np.array(count)

                # 删除行和列
                DK = np.delete(DK, count, axis=0)
                DK = np.delete(DK, count, axis=1)

                DM = np.delete(DM, count, axis=0)
                DM = np.delete(DM, count, axis=1)

                return DK, DM

            # 示例用法:
            # 假设在调用函数之前已经定义了 Node、Uss 和 NP 的值
            # K 和 M 是你的二维 NumPy 数组
            # DK, DM = delete(K, M, len(Node), Uss, NP)

            DK,DM=delete(Kw, Mw)
            print('-------------------------Kw-------------------------')
            print(Kw)
            print('-------------------------Mw-------------------------')
            print(Mw)
            e,v = sl.eig(DK,DM)

            #e=np.sqrt(e)
            print('-------------------------e-------------------------')
            print(e)
            print('-------------------------v-------------------------')
            print(v)
            #57924 105610 139080
            #print(la.norm(DK-DM*e[0]))
            

            print('-------------------------omega-------------------------')
            e=np.sqrt(e)
            print(e)
            print('-------------------------Uw-------------------------')
            m=0
            for i in range(len(v)):
                Uw=[None]*3*len(node_list)
                n=0
                for i in range(len(node_list)):
                        for j in range(3):
                            #print(i,j)
                            if Uss[i][j] == 1:
                                Uw[i * 3 + j]=0
                for i in range(len(Uw)):
                    if Uw[i] == None:
                        Uw[i]=v[m][n]
                        n+=1
                m+=1
                s=int(input('请输入放大倍数：'))
                #s=0.001
                draw_strain(node_list,connect_node,Uw,mode,s)
        else:
            print("输入错误")
````

utils.py主要存储main.py中的一些方法
```python

from node import *
import numpy
import matplotlib.pyplot as plt
from node import *
def convert(list1):
    for i in range(len(list1)):
        list2=list1[i]
        for j in range(len(list2)):
            k = list1[i][j]
            list1[i][j]=float(k)
    return  list1
def convert1(list1):
    for i in range(len(list1)):
        list2=list1[i]
        for j in range(len(list2)):
            k = list1[i][j]
            list1[i][j]=int(k)
    return  list1
def add_node_utils(node_list):
    I=0
    print('结束输入：end')
    while True:
        x = input("请输入节点的x y z坐标,或者退出：")
        #print(node_list)
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
        #print('没有此力请输入0')
        x = input("请输入节点的index index:")
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
        x = input("请输入节点的连接序号 起始点距离占比：")
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



def add_force(R,node_list,connect_list,mode):
#输入外载荷，单元内的外力平均分配到两结点，

    print('载荷施加在结点上，用列表表示其性质：[作用结点号，类型(力为0，力矩为1)，x方向分量，y方向分量，z方向分量]\n节点编号与结构图如下')
    plot(node_list,connect_list)
    force_input = input('请输入\n作用结点号，类型(力为0，力矩为1,杆只能输入力)，x方向分量，y方向分量，z方向分量\n输入完毕请输入0结束此步骤\n').split(',')
    force_input = list(map(float,force_input))
    while True:
        if force_input == [0]:
            print('以下是在载荷')
            for i in range(0,len(R)):
                F = R[i]
                print('结点'+str(F[0]),'类型'+str(F[1]),'x方向分量'+str(F[2]),'y方向分量'+str(F[3]),'z方向分量'+str(F[4]))
            break
        else:
            if len(force_input) != 5:
                print('参数不全!!!!!')
                force_input = input('请输入\n作用结点号，类型(力为0，力矩为1)，x方向分量，y方向分量，z方向分量\n输入完毕请输入0结束此步骤\n').split(',')
                force_input = list(map(float,force_input))
            elif mode == 0 and force_input[1] != 0:
                print('桁架模式下只能输入力!!!!!')
                force_input = input('请输入\n作用结点号，类型(力为0，力矩为1)，x方向分量，y方向分量，z方向分量\n输入完毕请输入0结束此步骤\n').split(',')
                force_input = list(map(float,force_input))
            else:
                R.append(force_input)
                print('输入成功，请输入下一载荷')
                force_input = input('请输入\n作用结点号，类型(力为0，力矩为1)，x方向分量，y方向分量，z方向分量\n输入完毕请输入0结束此步骤\n').split(',')
                force_input = list(map(float,force_input))
    return R
#单元矩阵计算
def cal_lam(ary,mode,ex_node = 0):
    ele_x = ary/np.linalg.norm(ary)
    if mode == 0:
        #---------------------------------------------------------------------------------------------------------------------
        lam = np.matrix([[ele_x[0],ele_x[1],ele_x[2],0,0,0],[0,0,0,ele_x[0],ele_x[1],ele_x[2]]],dtype=float)
        #---------------------------------------------------------------------------------------------------------------------
        return lam
    elif mode == 1:
        lx,mx,nx = (ele_x[0],ele_x[1],ele_x[2])
        enx = ex_node/np.linalg.norm(ex_node)
        p1,p2,p3 = (enx[0],enx[1],enx[2])
        s = np.sqrt((mx*p3-nx*p2)**2+(nx*p1-lx*p3)**2+(lx*p2-mx*p1)**2)
        lmn = lx*p1+mx*p2+nx*p3
        ly,my,ny,lz,mz,nz = ((p1-lx*lmn)/s,(p2-mx*lmn)/s,(p3-nx*lmn)/s,(mx*p3-nx*p2)/s,(nx*p1-lx*p3)/s,(lx*p2-mx*p1)/s)
        lam = np.matrix([[lx,ly,lz],[mx,my,mz],[nx,ny,nz]])
        return lam

def cal_k_or_m(m,Node,Rod,Lam,Ke,Me):
    kw = np.matrix(np.zeros((3*(m+1)*len(Node),3*(m+1)*len(Node)),dtype=float))
    mw = np.matrix(np.zeros((3*(m+1)*len(Node),3*(m+1)*len(Node)),dtype=float))
    t0 = np.matrix(np.zeros((3,3),dtype=float))
    for i in range(len(Rod)):
        start_num = Rod[i][0]
        end_num = Rod[i][1]
        if m==0:
            te = Lam[i].T
        elif m==1:
            te = np.vstack((np.hstack((Lam[i],t0,t0,t0)),np.hstack((t0,Lam[i],t0,t0)),np.hstack((t0,t0,Lam[i],t0)),np.hstack((t0,t0,t0,Lam[i]))))
        ke = np.dot(np.dot(te,Ke[i]),te.T)
        me = np.dot(np.dot(te,Me[i]),te.T)
        start1,start2,end1,end2 = (int(3*(m+1)*start_num),int(3*(m+1)*(start_num+1)),int(3*(m+1)*end_num),int(3*(m+1)*(end_num+1)))
        n1,n2 = (int(3*(m+1)),int(6*(m+1)))
        kw[start1:start2,start1:start2] += ke[0:n1,0:n1]
        kw[start1:start2,end1:end2] += ke[0:n1,n1:n2]
        kw[end1:end2,start1:start2] += ke[n1:n2,0:n1]
        kw[end1:end2,end1:end2] += ke[n1:n2,n1:n2]
        mw[start1:start2,start1:start2] += me[0:n1,0:n1]
        mw[start1:start2,end1:end2] += me[0:n1,n1:n2]
        mw[end1:end2,start1:start2] += me[n1:n2,0:n1]
        mw[end1:end2,end1:end2] += me[n1:n2,n1:n2]
    return [kw,mw]
def u_and_fw(m,Node,R,Rod,Lam,Ke,Me):
    u = np.zeros(((m+1)*3*len(Node),1),dtype=float)
    fw = np.zeros(((m+1)*3*len(Node),1),dtype=float)
    for i in range(0,len(R)):
        force_inf = R[i]
        for j in range(0,3):
            fw[(int(3*force_inf[0]+3*force_inf[1]+j),0)] += force_inf[j+2]
    return [u,fw]
def cal(Fw,Kw,mode,Node,Us):
    #处理约束
    #Us = [[0,0,0],[0,1,0],[1,0,0],[1,1,0]]#输入约束，元素为列表，每个代表一项约束，其中第一项为结点号，第二项为位移分量编号，第三项为位移值
    #处理刚度矩阵与载荷列向量
    for i in range(0,len(Us)):
        us = Us[i]
        Fw[int(3*(mode+1)*us[0]+us[1]),0] = us[2]
        Kw[int(3*(mode+1)*us[0]+us[1]),:] = 0
        Kw[:,int(3*(mode+1)*us[0]+us[1])] = 0
        Kw[int(3*(mode+1)*us[0]+us[1]),int(3*(mode+1)*us[0]+us[1])] = 1

    Z = []
    for i in range(len(Node)):
        Z.append(float(Node[i][-1]))
    if Z==[0.0]*len(Node):
        for i in range(len(Node)):
            Fw[int(3*(mode+1)*i+2),0] = 0
            Kw[int(3*(mode+1)*i+2),:] = 0
            Kw[:,int(3*(mode+1)*i+2)] = 0
            Kw[int(3*(mode+1)*i+2),int(3*(mode+1)*i+2)] = 1
            if mode==1:
                Fw[int(3*(mode+1)*i+4),0] = 0
                Kw[int(3*(mode+1)*i+4),:] = 0
                Kw[:,int(3*(mode+1)*i+4)] = 0
                Kw[int(3*(mode+1)*i+4),int(3*(mode+1)*i+4)] = 1
                Fw[int(3*(mode+1)*i+5),0] = 0
                Kw[int(3*(mode+1)*i+5),:] = 0
                Kw[:,int(3*(mode+1)*i+5)] = 0
                Kw[int(3*(mode+1)*i+5),int(3*(mode+1)*i+5)] = 1
    return np.dot(Kw.I,Fw)

# Drawing strains with scaling
def draw_strain(dic, rod, U,mode, scale_factor=1000):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Original structure
    for i in range(len(rod)):
        rod_node = rod[i]
        node_coord1 = dic[rod_node[0]]
        node_coord2 = dic[rod_node[1]]
        x, y, z = [node_coord1[0], node_coord2[0]], [node_coord1[1], node_coord2[1]], [node_coord1[2], node_coord2[2]]
        ax.scatter(x, y, z, c='red', s=10)
        ax.plot(x, y, z, color='black')

    # Deformed structure with scaling
    for i in range(len(rod)):
        rod_node = rod[i]
        node_coord1 = dic[rod_node[0]]
        node_coord2 = dic[rod_node[1]]
        x = [node_coord1[0] + scale_factor * U[int(3 * (mode + 1) * rod_node[0])], 
             node_coord2[0] + scale_factor * U[int(3 * (mode + 1) * rod_node[1])]]
        y = [node_coord1[1] + scale_factor * U[int(3 * (mode + 1) * rod_node[0] + 1)], 
             node_coord2[1] + scale_factor * U[int(3 * (mode + 1) * rod_node[1] + 1)]]
        z = [node_coord1[2] + scale_factor * U[int(3 * (mode + 1) * rod_node[0] + 2)], 
             node_coord2[2] + scale_factor * U[int(3 * (mode + 1) * rod_node[1] + 2)]]

        # Flatten the arrays before passing to scatter and plot functions
        x, y, z = np.array(x).flatten(), np.array(y).flatten(), np.array(z).flatten()

        ax.scatter(x, y, z, c='blue', s=10)
        ax.plot(x, y, z, color='black')

    plt.show()
def get_l(l):
    print('输入约束，元素为列表，每个代表一项约束，其中第一项为结点号，第二项为位移分量编号，第三项为位移值')
    while True:
        x = input('请输入约束，输入0结束此步骤\n').split(',')
        print(x)
        if x == ['0']:
            break
        else:
            l.append(x)
    return l
```

下面是node.py主要用于单元和节点的添加：
```python
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
```
下面是calculate.py（已废弃）：
```python
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
    # 计算角（弧度）
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
```









