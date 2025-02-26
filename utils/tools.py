import os
import sys
import datetime
import pickle
import random
def savefile(obj,path):
    with open(path,'w+') as f:
        pickle.dump(obj,f)
def split_trn_tst(actlis,train_ratio=0.8):

    total_size = len(actlis)
    train_size = int(total_size * train_ratio)
    
    # 随机选择训练集的元素索引
    train_indices = random.sample(range(total_size), train_size)
    
    # 创建训练集和测试集
    train_list = [actlis[i] for i in train_indices]
    test_list = [item for i, item in enumerate(actlis) if i not in train_indices]
    
    return train_list, test_list




def get_root_directory(path):
    # 将相对路径转换为绝对路径
    absolute_path = os.path.abspath(path)
    root_dir = absolute_path
   
    return root_dir


