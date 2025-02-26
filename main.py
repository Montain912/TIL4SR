'''

'''



import torch
import torch.optim as optim
import numpy as np

from TIL4SR import *

from utils.parser import parse_args
from GCN.NGCF.NGCF import NGCF 
from utils.helper import *
from utils.batch_test import *


import warnings
warnings.filterwarnings('ignore')
from time import time

args = parse_args()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(2020)
if args.init_model=="gnn":


    interactions =data_generator.interactions
    interactions_test = data_generator.interactions_test
    interactions_times = data_generator.interactions_times
    interactions_test_time = data_generator.interactions_test_time



    args.device = torch.device('cuda:' + str(args.gpu_id))
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)
    
    pc =PeroidCluster(args.cluster_k,dimension=64
                        ,device=args.device)

    long_filter = Filter_long(args.last_k, args.embed_size,interactions=interactions,dataset=args.dataset,device=args.device)

    t0 = time()
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer_1 = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_2 = optim.Adam(pc.parameters(), lr=args.lr)
    optimizer_3 = optim.Adam(long_filter.parameters(), lr=args.lr)
    e = 0
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

    #----------------------------------------------------------

   
   
    items_num = data_generator.n_items - 1


  
    #------------------------------------------------------------------------------------------------------------------------
    
    for epoch in range(args.epoch):
        t1 = time()
        loss_gcn, mf_loss, emb_loss ,cluster_loss = 0., 0., 0. , 0. 
        n_batch = data_generator.n_train // args.batch_size + 1
        #1
        for idx in range(n_batch):
            # users, pos_items, neg_items,pos_times = data_generator.sample() 

            users, pos_items, neg_items ,pos_times= data_generator.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            # optimizer_1.zero_grad()
            batch_loss.backward()
            optimizer_1.step()

            loss_gcn += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

            u_embs = model.state_dict()["embedding_dict.user_emb"][users]
            i_embs = model.state_dict()["embedding_dict.item_emb"][pos_items]
            #2
            # print("212:",u_embs.size(),i_embs.size())
            pc.initialize(u_embs,i_embs,pos_times,args.embed_size)
            batch_cluster_loss =  pc() / args.batch_size 

            
            # optimizer_2.zero_grad()
            batch_cluster_loss.backward()
            optimizer_2.step()
        # 3
        item_emb = model.state_dict()["embedding_dict.item_emb"]
        long_filter.user_embedding = model.state_dict()["embedding_dict.user_emb"]
        long_filter.item_embedding = item_emb
        long_filter.cluster_emb = pc.cluster_embs


        interaction_batches = [dict(list(interactions.items())[i:i + args.batch_size]) for i in range(0, len(interactions), args.batch_size)]
        interactions_time_batch = [dict(list(interactions_times.items())[i:i + args.batch_size]) for i in range(0, len(interactions_times), args.batch_size)]
        batch_losses = []
        losses = [] 
        users,pos_items,neg_items,_ = data_generator.sample()
        
        
        for v in range(len(interaction_batches)):
            users,pos_items,neg_items = user_pos_neg(interaction_batches[v],data_generator.n_items)
            u_g_embeddings, pos_i_g_embeddings,neg_i_g_embeddings= long_filter(interaction_batches[v], interactions_time_batch[v],pc,users,pos_items,neg_items,
                                                                               args.coefficient_alpha1,args.coefficient_alpha2,args.coefficient_alpha3)
            loss,_,_ = long_filter.create_bpr_loss(u_g_embeddings,pos_i_g_embeddings, neg_i_g_embeddings)
            batch_losses.append(loss)
            # optimizer_3.zero_grad()
            if e==0:
        
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            
            optimizer_3.step()
        
        # 计算并记录每个 epoch 的平均损失
        epoch_loss = torch.mean(torch.tensor(batch_losses))
        if e%20 == 0:
                print(f"[loss info]：现在是第{e}轮，损失为[{loss_gcn}]+[{batch_cluster_loss}]+[{epoch_loss}]")
                torch.save(long_filter.user_mix_embs, f"/home/duyuxiang/Code/mycode/TIL4SR/save/{args.dataset}/prompt_embs/user_long_embs_{epoch}.pth")
                torch.save(long_filter,  f"/home/duyuxiang/Code/mycode/TIL4SR/save/{args.dataset}/filter/{epoch}.pth")
        
        e +=1
        
        losses.append(epoch_loss.item())
        # if epoch_loss.item() > losses[-1]:
        #     break
        # if epoch % 20 == 0 :
            # top_k = 10
            # users_to_test = list(data_generator.test_set.keys())

            # scores_matrix = torch.matmul(long_filter.user_mix_embs, item_emb.t()).to("cpu")
            # # print("long_filter.user_mix_embs",long_filter.user_mix_embs)
            # _, recommendations = torch.topk(scores_matrix, top_k, dim=1)
            # # actual_interactions = interactions_test
         

            # #获取test 列表
            # actual_interactions = data_generator.interactions_test
            
           

            # ndcg_scores = []
            # recall_scores = []
            # for user_id in actual_interactions.keys():
            #     actual = actual_interactions[user_id]
            #     actual = [int(ac) for ac in actual]
            #     predicted = recommendations[int(user_id)-1]

                
            #     # 生成实际交互的相关性分数6
            #     # relevance = torch.tensor([1 if items.item() in actual else 0 for items in predicted])
            #     try:
            #         relevance = torch.tensor([1 if item.item() in actual else 0 for item in predicted], device="cpu")
                    
            #     except RuntimeError as e:
            #         print("RuntimeError:", e)
              
            # torch.save(long_filter.user_mix_embs, f"/home/duyuxiang/Code/mycode/TIL4SR/save/{args.dataset}/prompt_embs/user_long_embs_{epoch}.pth")
            # torch.save(long_filter,  f"/home/duyuxiang/Code/mycode/TIL4SR/save/{args.dataset}/filter/{epoch}.pth")
        

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(long_filter, users_to_test, pc_model=pc,tp=1,drop_flag=False,alphas=[args.coefficient_alpha1,args.coefficient_alpha2,args.coefficient_alpha3])
        # ret = test(model, users_to_test, pc_model=pc,tp=0,drop_flag=False)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])
        print(args.verbose )
        if args.verbose > 0:
            print("time:",t3-t2)
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][1],
                        ret['precision'][0], ret['precision'][1], ret['hit_ratio'][0], ret['hit_ratio'][1],
                        ret['ndcg'][0], ret['ndcg'][1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), args.weights_path + args.dataset + "/gcn/"+str(epoch) + '.pkl')
            print('save the weights in path: ', args.weights_path + args.dataset + "/gcn/"+str(epoch) + '.pkl')

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)
    

   


elif args.init_model =="sr":
    pass
else:
    print('initialization user&item embedding model not defined')