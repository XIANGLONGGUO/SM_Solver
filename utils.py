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
        x = input("请输入节点的index index")
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
            print('输入完毕，外载荷如下')
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
#单元矩阵计算
def cal_lam(ary,mode,ex_node = 0):
    ele_x = ary/np.linalg.norm(ary)
    if mode == 0:
        if ele_x[0]==0 and ele_x[1]==0:
            ele_y = np.array([1,0,0])
            ele_z = np.array([0,1,0])
        else:
            ele_y = np.array([-ele_x[1],ele_x[0],0])
            cross = np.cross(ele_x, ele_y)
            ele_z = cross/np.linalg.norm(cross)

        cos_xx,cos_xy,cos_xz,cos_yx,cos_yy,cos_yz,cos_zx,cos_zy,cos_zz = (ele_x[0],ele_x[1],ele_x[2],ele_y[0],ele_y[1],ele_y[2],ele_z[0],ele_z[1],ele_z[2])
        lam = np.matrix([[cos_xx,cos_yx,cos_zx],[cos_xy,cos_yy,cos_zy],[cos_xz,cos_yz,cos_zz]])
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
    for i in len(Rod):
        start_num = Rod[i][0]
        end_num = Rod[i][1]
        if m==0:
            te = np.vstack((np.hstack((Lam[i],t0)),np.hstack((t0,Lam[i]))))
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
def cal(Fw,Kw,mode,Node):
    #求解静力位移
    #输入位移约束
    Us = [[0,0,0],[0,1,0],[1,0,0],[1,1,0]]#输入约束，元素为列表，每个代表一项约束，其中第一项为结点号，第二项为位移分量编号，第三项为位移值
    #处理刚度矩阵与载荷列向量
    for i in range(0,len(Us)):
        us = Us[i]
        Fw[int(3*(mode+1)*us[0]+us[1]),0] = us[2]
        Kw[int(3*(mode+1)*us[0]+us[1]),:] = 0
        Kw[:,int(3*(mode+1)*us[0]+us[1])] = 0
        Kw[int(3*(mode+1)*us[0]+us[1]),int(3*(mode+1)*us[0]+us[1])] = 1

    Z = []
    for i in len(Node):
        Z.append(float(Node[i][-1]))
    if Z==[0.0]*len(Node):
        for i in Node.keys():
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
    for i in len(rod):
        rod_node = rod[i]
        node_coord1 = dic[rod_node[0]]
        node_coord2 = dic[rod_node[1]]
        x, y, z = [node_coord1[0], node_coord2[0]], [node_coord1[1], node_coord2[1]], [node_coord1[2], node_coord2[2]]
        ax.scatter(x, y, z, c='red', s=10)
        ax.plot(x, y, z, color='black')

    # Deformed structure with scaling
    for i in len(rod):
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
