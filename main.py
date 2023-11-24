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
    while True:
        print('输入calculate来计算')
        print('输入add_node来添加节点')
        print('输入add_connect来添加连接')
        print('输入remove_node来删除节点')
        print('输入add_point来添加节点间的节点')
        print('输入remove_connect来删除连接')
        print('输入remove_point来删除节点间的节点')
        print('输入plot来绘图')
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
        
        else:
            print("输入错误")