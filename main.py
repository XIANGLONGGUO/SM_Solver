import math
import numpy as np
from utils import *
#from node import *
from calculate import *
from limitation import *#不需要的，直接在连接时候约束了

if __name__=="__main__":
    print('Welcome to SM_Solver')
    print('author:GUOxianglong')
    node_list=[]
    connect_node=[]
    dist_list=[]
    soild_limitation=[]
    limitatation=[]
    Dict_disp={}
    Lam = {}
    Ke = {}
    Me = {}
    while True:
        print('输入calculate来计算')
        print('输入add_node来添加节点')
        print('输入add_connect来添加连接')
        print('输入remove_node来删除节点')
        print('输入add_point来添加节点间的节点')
        print('输入remove_connect来删除连接')
        print('输入remove_point来删除节点间的节点')
        print('输入plot来绘图')
        print('输入mat来输入材料参数')
        print('exit退出')
        #######################################################################
        #Edit the shape of structure

        imput=input("输入操作：")
        if imput=="exit":
            exit(0)
        if imput=="calculate" or imput=="c":
            print("开始计算")
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
            print('相同节点添加一次力即可')
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
        elif imput=='calculate' or imput=='c':
            

            E,D,A,Iz,Iy,Ix,G = (Metal[0],Metal[1],Metal[2],Metal[3],Metal[4],Metal[5],Metal[6])
            for i in len(connect_node):
                node = connect_node[i]
                op = node_list[node[0]]
                ed = node_list[node[1]]
                ele_array = np.array(op)-np.array(ed)
                lenth = np.linalg.norm(ele_array)
            
                if mode == 0:
                    Lam[i] = cal_lam(ele_array)
                    k = np.zeros((6,6))
                    k[0,0],k[3,3] = (1,1)
                    k[3,0],k[0,3] = (-1,-1)
                    m = np.matrix(2*np.eye(6,dtype=float))
                    m[3:6,0:3],m[0:3,3:6] = (np.matrix(np.eye(3,dtype=float)),np.matrix(np.eye(3,dtype=float)))#一致质量矩阵
                    Ke[i] = np.mat((E*10**9*A/lenth)*k)
                    Me[i] = D*A*lenth*m/6
                elif mode == 1:
                    print(op,ed)
                    enode = np.array(list(map(float,input('请输入任意主平面xy内点的坐标，格式为0,0,0\n').split(','))))-np.array(op)
                    Lam[i] = cal_lam(ele_array,enode)
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
            u_and_fw_out = u_and_fw(mode,node_list,connect_node,dist_list,R,Lam,Ke,Me)
            Uw = u_and_fw_out[0]#位移列向量
            Fw = u_and_fw_out[1]#载荷列向量
            K_an_M_out = cal_k_or_m(mode,node_list,connect_node,Lam,Ke,Me,Uw,Fw)
            Kw = K_an_M_out[0]
            Mw = K_an_M_out[1]
            
            Uw = cal(Fw,Kw,mode,node_list)
            print(Uw)
            s=50000000
            draw_strain(node_list,connect_node,Uw,mode,s)
        else:
            print("输入错误")