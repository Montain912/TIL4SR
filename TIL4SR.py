import torch
from torch.optim import Adam
import json,math,sys,os,random
sys.path.append('/home/duyuxiang/Code/mycode/NGCF-PyTorch/NGCF')
sys.path.append('/home/duyuxiang/Code/mycode/NGCF-PyTorch/NGCF/utility')
from utils.batch_test import *

import warnings
import time
import itertools
from log_save import save_log
import json,math,sys,os,random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from collections import Counter
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
torch.manual_seed(2024)
import setting

# ALPHA_2 = setting.alpha_2
# ALPHA_3 = setting.alpha_3
# ALPHA_4 = setting.alpha_4
 

class PeroidCluster(nn.Module):
    def __init__(self, k,device,dimension=64, tempture=0.5, ):
        super(PeroidCluster, self).__init__()
        self.k = k
        hidden_dim = 64
        self.device = device
        self.tempture = tempture
        self.time_encond_layer = nn.Linear(1, dimension).to(device)
        self.concat_layer = nn.Linear( dimension*3 ,64).to(device)
        self.cluster_embs = nn.Parameter(torch.empty((k,hidden_dim))).to(self.device)

        nn.init.xavier_uniform_(self.cluster_embs)
    def initialize(self,Eu,Ei,times,dimension):
        self.user_embs = Eu 
        self.item_embs = Ei
        self.dim = dimension
        self.times = times
        times_np = np.array(times,dtype=int)
        max_time = times_np.max()
        min_time = times_np.min()
        self.timedis_threshold = (max_time - min_time) / self.k
        
        cos_times = np.cos(times_np)  # 计算cosine值
        cos_tensor = torch.Tensor(cos_times).unsqueeze(0).to(self.device) 
        self.time_embs = self.time_encond_layer(cos_tensor.unsqueeze(-1)).squeeze(0).to(self.device)
        # self.cluster_info = self.__init_emb__(dimension)
        self.cluster_info = {key:self.cluster_embs[key] for key in range(self.k)}


    def create_time_embs(self,time):
        cos_times = np.cos([int(t) for t in time])  # 计算cosine值
        # cos_tensor = torch.Tensor(cos_times).unsqueeze(0).to(self.device)
        cos_tensor = torch.tensor(cos_times).unsqueeze(0).to(self.time_encond_layer.weight.dtype).to(self.device)
        # print(cos_tensor)
        time_emb = self.time_encond_layer(cos_tensor.unsqueeze(-1)).squeeze(0).to(self.device)  
        return time_emb
        
    def __init_emb__(self, d):
        times = np.array(self.times,dtype=int)
        max_time = times.max()
        min_time = times.min()
        self.timedis_threshold = (max_time - min_time) / self.k

        cos_times = np.cos(times)  # 计算cosine值
        cos_tensor = torch.Tensor(cos_times).unsqueeze(0).to(self.device) 

        self.time_embs = self.time_encond_layer(cos_tensor.unsqueeze(-1)).squeeze(0).to(self.device)
        #类簇中心
        cluster_indices = self.uniform_sampling_timestamps(times,2)

        cluster_embs = []
        cluster_info = {}
        i = 0 
        for indice in cluster_indices:
         
            emb_cat = torch.cat((self.user_embs[indice].unsqueeze(0),self.itrm_embs[indice].unsqueeze(0),self.time_embs[indice].unsqueeze(0)),dim=1)     
            x = self.concat_layer(emb_cat).squeeze()
            cluster_embs.append(x)
            cluster_info[i] = cluster_embs[i]
            i += 1
        self.cluster_embs.data = torch.stack(cluster_embs).clone
        self.cluster_embs.to(self.device)
        return cluster_info
    
    def uniform_sampling_timestamps(self,timestamps, n_samples):
        sorted_indices = np.argsort(timestamps)
        sampled_indices = np.linspace(0, len(sorted_indices) - 1, n_samples, dtype=int)
        sampled_indices = sorted_indices[sampled_indices]
        return sampled_indices
    
    def select_pos_nev(self, cluster_max_sim_index, times):
        pos_dic = {}
        neg_dic = {}
        for key in range(self.k):
            pos_dic[key] = []
            neg_dic[key] = []
        for key, value in cluster_max_sim_index.items():
            mid_time = times[value]
            for indices in self.cluster_dic[key]:
                # print("time chazhi:",abs(times[indices] - mid_time))
                if abs(int(times[indices]) - int(mid_time)) < self.timedis_threshold:
                    pos_dic[key].append(indices)
                else:
                    neg_dic[key].append(indices)
        return pos_dic, neg_dic

    def getdata(self):
        time_embs = self.timeencode()
        self.x = (0.5 * self.x) + (0.5 * time_embs)

    def calc_clusterLoss(self, cluster_emb, positive_pairs, negative_pairs):
        positive_similarities = F.cosine_similarity(positive_pairs, cluster_emb.unsqueeze(0), dim=1) / self.tempture
        negative_similarities = F.cosine_similarity(negative_pairs, cluster_emb.unsqueeze(0), dim=1) / self.tempture
        denom = torch.exp(positive_similarities) + torch.exp(negative_similarities).sum()
        loss = -torch.log(torch.exp(positive_similarities) / denom)
        return loss

    def forward(self, lr=0.01):
        with torch.no_grad():
            # print(self.user_embs.shape,self.item_embs.shape,self.time_embs.shape)
            self.x = self.concat_layer(torch.cat((self.user_embs,self.item_embs,self.time_embs),dim=1)).to(self.device)  
        dot_product = torch.mm(self.x, self.cluster_embs.t())
        norm_c_k = torch.norm(self.cluster_embs, dim=1, keepdim=True)
        norm_h_i = torch.norm(self.x, dim=1)
        norms_product = torch.mm(norm_c_k, norm_h_i.unsqueeze(0)).t()
        cosine_similarity = dot_product / norms_product
        normalized_similarity = (cosine_similarity + 1) / 2
        self.cluster_dic = {}
        _, closest_clusters = torch.max(normalized_similarity, dim=1)
        for index, cluster_id in enumerate(closest_clusters):
            if cluster_id.item() not in self.cluster_dic:
                self.cluster_dic[cluster_id.item()] = []
            self.cluster_dic[cluster_id.item()].append(index)
        cluster_max_sim_index = {}
        
        
        for cluster in range(self.cluster_embs.shape[0]):
            cluster_mask = (closest_clusters == cluster)
            if cluster_mask.any():
                cluster_similarities = normalized_similarity[:, cluster]
                max_sim_value, max_sim_index = torch.max(cluster_similarities[cluster_mask], dim=0)
                global_indices = torch.arange(self.x.shape[0]).to(self.device)
                global_max_sim_index = global_indices[cluster_mask][max_sim_index]
                cluster_max_sim_index[cluster] = global_max_sim_index.item()
        pos_point, neg_point = self.select_pos_nev(cluster_max_sim_index, self.times)
        losses = []
        for i in pos_point.keys():
            if len(pos_point[i]) == 0 or len(neg_point[i]) == 0:
                # print(f'Cluster {i} has no positive or negative samples, skipping...')
                continue
            pos_indices = torch.tensor(pos_point[i]).to(self.device).long()
            neg_indices = torch.tensor(neg_point[i]).to(self.device).long()
            positive = torch.index_select(self.x, 0, pos_indices)
            negatives = torch.index_select(self.x, 0, neg_indices)
            anchor = self.cluster_embs[i].unsqueeze(0)
            loss = self.calc_clusterLoss(anchor, positive, negatives)
            losses.append(loss.mean())
        cl = self.clloss(self.cluster_embs)
        total_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)

        return total_loss + cl
    def clloss(self,cluster_embeddings):
        num_clusters = cluster_embeddings.size(0)
        loss = 0.0
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                # Calculate the Euclidean distance between cluster i and cluster j
                distance = torch.norm(cluster_embeddings[i] - cluster_embeddings[j], p=2)
                # We want to maximize distance, thus minimize negative distance
                loss -= distance
        # Normalize the loss by the number of pairs
        return loss / (num_clusters * (num_clusters - 1) / 2)


    def labels(self):
        label_ = np.zeros(len(self.x), dtype=int)
        for key, values in self.cluster_dic.items():
            for index in values:
                label_[index] = int(key)
        return label_


class GMM(nn.Module):
    def __init__(self, k,dimension=64, tempture=0.5, device="cpu"):
        super(GMM, self).__init__()
        self.k = k
        hidden_dim = 64
        self.device = device
        self.tempture = tempture
        self.time_encond_layer = nn.Linear(1, dimension).to(device)
        self.gmm = GaussianMixture(n_components=k, random_state=42)
       
    
       
    def initialize(self,Eu,Ei,times,dimension):
        self.user_embs = Eu 
        self.item_embs = Ei
        self.dim = dimension
        self.times = times
        times_np = np.array(times,dtype=int)
        max_time = times_np.max()
        min_time = times_np.min()
        self.timedis_threshold = (max_time - min_time) / self.k
        
        cos_times = np.cos(times_np)  # 计算cosine值
        cos_tensor = torch.Tensor(cos_times).unsqueeze(0).to(self.device) 
        self.time_embs = self.time_encond_layer(cos_tensor.unsqueeze(-1)).squeeze(0).to(self.device)
        # self.cluster_info = self.__init_emb__(dimension)


    def create_time_embs(self,time):
        cos_times = np.cos(time)  # 计算cosine值
        # cos_tensor = torch.Tensor(cos_times).unsqueeze(0).to(self.device)
        cos_tensor = torch.tensor(cos_times).unsqueeze(0).to(self.time_encond_layer.weight.dtype).to(self.device)
        # print(cos_tensor)
        time_emb = self.time_encond_layer(cos_tensor.unsqueeze(-1)).squeeze(0).to(self.device)  
        return time_emb
        
    
    def getdata(self):
        time_embs = self.timeencode()
        self.x = (0.5 * self.x) + (0.5 * time_embs)



    def forward(self, lr=0.01):
        with torch.no_grad():
            # print(self.user_embs.shape,self.item_embs.shape,self.time_embs.shape)
            # self.x = self.concat_layer(torch.cat((self.user_embs,self.item_embs,self.time_embs),dim=1)).to(self.device)  
            self.x = ((self.user_embs+self.item_embs+self.time_embs) / 3 ).to(self.device)  
        self.gmm.fit(self.x)
        self.cluster_embs = self.gmm.means_

    def labels(self):
        label_ = np.zeros(len(self.x), dtype=int)
        for key, values in self.cluster_dic.items():
            for index in values:
                label_[index] = int(key)
        return label_





os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
dataset = args.dataset
datapath = '/home/duyuxiang/Code/mycode/TIL4SR/Data/'


def TimeEmbedding(timestamp,size):
    time_Embedding = torch.zeros((size))
    
    for i in range(size):
        if i % 2 == 0:
            time_Embedding[i] = math.sin((int(timestamp) / ((10000.0)**(2*i/size))))
        else:
            time_Embedding[i] = math.cos((int(timestamp) / ((10000.0)**(2*i/size))))
    return time_Embedding


def _shorted_period_sample(M, k):
    interactions = {}

    with open(M, 'r+') as f:
        for m in f.readlines():
            inter = m.split(" ")
            inter[-1] = inter[-1][0:-1]
            user = inter[0]

            if len(inter) <= k:
                items = inter[1:]
            else:
                items = inter[len(inter)-k:]
            interactions[user] = items

    return interactions

def shorted_period_sample(interaction_dic,k):
    interactions_shorted = {}

    for user,items in interaction_dic.items():
        
        if len(items) <= k:
            interactions_shorted[user] = items
        else:
            interactions_shorted[user] = items[len(items) - k:]
    return interactions_shorted


def construct_bpr_Triplet(dataset, item_num):
    data_file = dataset + "/train.txt"
    bpr_Triplets = []
    with open(data_file, "r+") as f:
        for m in f.readlines():
            inter = m.split(" ")
            inter[-1] = inter[-1][0:-1]
            user = inter[0]

            for pos in inter[1:]:
                while True:
                    nev = random.randint(0, item_num)
                    if str(nev) in inter:
                        continue
                    else:
                        break
                bpr_Triplets.append((user, int(pos), nev))
    return bpr_Triplets


class Filter_long(nn.Module):
    def __init__(self, k, dimension, interactions,dataset,path="/home/duyuxiang/Code/mycode/TIL4SR",device="cpu"):
        super(Filter_long, self).__init__()
        self.device = device
        self.k = 10
        self.user_embedding = None  
        self.item_embedding = None
      
        self.cluster_emb = None

        self.decay = 1e-5
        self.batch_size = 1000
        # self.user_shorted_intent = torch.zeros_like(user_embedding,requires_grad=Ture).to(device)
        # self.user_shorted_intent = [0 for i in range(user_embedding.shape[0])]
        try:
            self.w_filters =  torch.load(path + f"/save/{dataset}/Filter_model/w_filter_k={k}.pth")
            print("load sucess...")
        except:
            print("renew create...")
            self.w_filters = self.init_para(interactions, k, dimension).to(device)
            torch.save(self.w_filters,path + f"/save/{dataset}/Filter_model/w_filter_k={k}.pth")
        print("init WQ WK WV Parameter")
        self.Wq = nn.Parameter(torch.empty(dimension, dimension, device=device))
        self.Wk = nn.Parameter(torch.empty(dimension, dimension, device=device))
        self.Wv = nn.Parameter(torch.empty(dimension, dimension, device=device))
        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wv)
        print("init WQ WK WV Parameter end")

        
    def compute_bpr_loss(self, user_embeddings, item_embeddings, users,pos_items,neg_items):
 
        user_ids = users
        pos_item_ids = pos_items
        neg_item_ids = neg_items

        user_emb = user_embeddings[user_ids]
        pos_item_emb = item_embeddings[pos_item_ids]
        neg_item_emb = item_embeddings[neg_item_ids]

        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)

        # BPR Loss
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        return loss

    
    def filter_long_interest(self, interactions, pc_model,interactions_times, user_embedding, item_embedding,alpha1,alpha2,alpha3):
       
        index = 0
        self.user_shorted_intent = [torch.zeros_like(user_embedding[0]) for _ in range(user_embedding.shape[0])]
        for user, items in interactions.items():
            user_index = int(user)
            if len(items) == 0:
                continue
            with torch.no_grad():
                time_embs = pc_model.create_time_embs(interactions_times[str(user_index)]).to(self.device)
                
############1
                items = torch.tensor([int(item) for item in items],requires_grad=False)
                items_indices = items.to(item_embedding.device)  # 确保 items 在同一设备上
                short_items_embs_ = item_embedding[items_indices].detach().float()
            
                user_emb = user_embedding[user_index].unsqueeze(0).detach().repeat(short_items_embs_.shape[0], 1)  # 使用.detach()断开梯度
                
                # # 混合用户嵌入和
                # emb_cat = torch.cat([user_emb,short_items_embs_,time_embs],dim=1)     
                # short_items_embs = pc_model.concat_layer(emb_cat).squeeze()

                # short_items_embs = (user_emb + short_items_embs_+ time_embs) / 3
                short_items_embs = (user_emb + short_items_embs_+ time_embs) 


                ### short_items_embs_ = torch.tensor([item_embedding[torch.tensor(int(items_indices))].detach().float() for items_indices in items])

############1
            ##注意力机制
            Q = torch.matmul(short_items_embs,self.Wq).to(self.device)
            K = torch.matmul( self.cluster_emb,self.Wk).to(self.device)
            V = torch.matmul( self.cluster_emb,self.Wv).to(self.device)
            
            d_k = Q.size(-1)
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
            attention_weights = F.softmax(attention_scores, dim=-1)
            shorted_embs = torch.matmul(attention_weights, V)

            """不用乘法，用和"""
            # sum_tensor = torch.sum(shorted_embs, dim=0) / shorted_embs.shape[0]
            # print(torch.sum((sum_tensor - user_embedding[user_index]),dim=0))
            # sys.exit()
            #
            """用乘"""
            if shorted_embs.shape[0]!=64:
                # print("sdasdas:",type(self.w_filters))
                
                    
                matrix_filter = torch.matmul(shorted_embs.t(), self.w_filters[user].to(self.device))
                
                
            else:
                matrix_filter = torch.matmul(shorted_embs.unsqueeze(0).t(), self.w_filters[user].to(self.device))
                
            # 
            new_user_emb_ = torch.matmul(user_embedding[user_index].unsqueeze(0), matrix_filter).squeeze(0)
            # norm = torch.norm(new_user_emb_, p=2)

            # new_user_emb_ = new_user_emb_ / norm
            new_user_emb =  alpha1 * user_embedding[user_index]  +  alpha2 * new_user_emb_
           
          
            
            # normalized_user_embs = new_user_emb
            
            self.user_shorted_intent[user_index] = new_user_emb
            index += 1
        # print(self.user_shorted_intent[4000])

        user_shorted_intent = torch.stack(self.user_shorted_intent)
        return alpha3 * user_shorted_intent

        # return (1-ALPHA_3) *  user_shorted_intent + ALPHA_3 * user_shorted_intent

    def init_para(self, interaction, k, dim):
        # data_file = dataset + "/train.txt"
        w_filter_matrix = nn.ParameterDict()
        for user,items in interaction.items():
            
            if len(items) < k:
                w_filter_matrix[user] = nn.Parameter(torch.randn((len(items), dim)).float())
            else:
                w_filter_matrix[user] = nn.Parameter(torch.randn((k, dim)).float())

        return w_filter_matrix
    def forward(self, interactions, interactions_times,pc_model,users,pos_items,neg_items,alpha1,alpha2,alpha3,drop_flag=False):
        self.user_mix_embs = self.filter_long_interest(interactions, pc_model,interactions_times, self.user_embedding, self.item_embedding,alpha1,alpha2,alpha3)

        # loss = self.compute_bpr_loss(self.user_mix_embs, self.item_embedding, users,p_is,n_is)
        u_g_embeddings = self.user_mix_embs[users, :].to(self.device)
        pos_i_g_embeddings = self.item_embedding[pos_items, :].to(self.device)
        neg_i_g_embeddings = self.item_embedding[neg_items, :].to(self.device)

        return u_g_embeddings,pos_i_g_embeddings,neg_i_g_embeddings
    # def create_bpr_loss(self, users, pos_items, neg_items):
    #     pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
    #     neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

    #     maxi = nn.LogSigmoid()(pos_scores - neg_scores)

    #     mf_loss = -1 * torch.mean(maxi)

    #     # cul regularizer
    #     regularizer = (torch.norm(users) ** 2
    #                    + torch.norm(pos_items) ** 2
    #                    + torch.norm(neg_items) ** 2) / 2
    #     emb_loss = self.decay * regularizer / self.batch_size

    #     return mf_loss + emb_loss, mf_loss, emb_loss

    # def rating(self, u_g_embeddings, pos_i_g_embeddings):
    #     return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())
    
    def create_bpr_loss(self, users, pos_items, neg_items):
        
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        # emb_loss = self.decay * regularizer / self.batch_size
        emb_loss = self.decay * regularizer 

        # print("mf_loss:",mf_loss)
        # print("emb_loss:",emb_loss)
        return mf_loss + emb_loss, mf_loss, emb_loss
    



def interactions_dic(filepath):
    actual_interactions = {}
    with open(filepath, 'r+') as f:
        for m in f.readlines():
            inter = m.split(" ")
            inter[-1] = inter[-1][0:-1]
            user = inter[0]

            actual_interactions[user] = inter[1:]

    return actual_interactions
# 准备数据和模型
def dcg():
    pass
def idcg():
    pass
def ndcg():
    pass


def hit_k(r, p, k):
    r = set(r[:k])
    s = p[:k]
    hit = 0
    for i in s:
        if s in r:
            hit +=1
    return hit

def precision_k(r,p,k):
    return hit_k(r,p,k) / k



def dcg_at_k(r, k):
    r = r[:k]
    if r.size(0):
        discount = torch.log2(torch.arange(2, k + 2).float().to(r.device))
        return torch.sum((2**r - 1) / discount)
    return torch.tensor(0.0, device=r.device)

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(torch.sort(r, descending=True)[0], k)
    if dcg_max.item() == 0:
        return torch.tensor(0.0, device=r.device)
    return dcg_at_k(r, k) / dcg_max
def recall_at_k(actual, predicted, k):
    actual_set = set(actual.tolist())
    predicted_set = set(predicted[:k].tolist())
    return len(actual_set & predicted_set) / float(len(actual_set))

def user_pos_neg(interactions,n_items):
    users = []
    pos_items = []
    neg_items = []
    for key,items in interactions.items():
        users.append(int(key))
        item = random.choice(items)
        pos_items.append(int(item))
        while 1:
            neg_item = random.randint(0, n_items-1)
            if str(neg_item) not in items and neg_item not in neg_items:
                neg_items.append(neg_item)
                break
            else:
                continue
    return users,pos_items,neg_items

